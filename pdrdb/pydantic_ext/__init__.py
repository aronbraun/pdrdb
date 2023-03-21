import datetime
import functools
import inspect
import json
import typing
import warnings
from decimal import Decimal
from enum import Enum
from typing import Union, Tuple, Dict, Any, Type, Optional, no_type_check, List, Callable, overload

import psycopg2.extensions
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel as _BaseModel, DateTimeError, ValidationError
from pydantic.error_wrappers import ErrorWrapper
from pydantic.fields import ModelField
from pydantic.json import ENCODERS_BY_TYPE
from pydantic.schema import encode_default, default_ref_template
from pydantic.typing import is_callable_type
from pydantic.validators import parse_datetime

from pdrdb.pydantic_ext.field import PhoneNumber, AdaptedType, CommaDelimitedList

if typing.TYPE_CHECKING:
    from pydantic.typing import DictStrAny
    from pydantic.typing import AbstractSetIntStr
    # noinspection PyProtectedMember
    from pydantic.typing import MappingIntStrAny
    from pydantic.typing import TupleGenerator
    from pydantic.typing import CallableGenerator

from pydantic.utils import ValueItems, lenient_issubclass

_missing = object()


class DBDefault:
    def __conform__(self, proto):
        if proto is psycopg2.extensions.ISQLQuote:
            return self

    def getquoted(self):
        return 'DEFAULT'

    def __str__(self):
        return 'Default'

    def __repr__(self):
        return self.__class__.__name__

    def __copy__(self):
        return self

    def __deepcopy__(self, memo=None):
        return self


DB_DEFAULT = DBDefault()


class BaseModel(_BaseModel):

    def _iter(
            self,
            to_dict: bool = False,
            by_alias: bool = False,
            include: Union['AbstractSetIntStr', 'MappingIntStrAny'] = None,
            exclude: Union['AbstractSetIntStr', 'MappingIntStrAny'] = None,
            exclude_unset: bool = False,
            exclude_defaults: bool = False,
            exclude_none: bool = False,
    ) -> 'TupleGenerator':
        # Merge field set excludes with explicit exclude parameter with explicit overriding field set options.
        # The extra "is not None" guards are not logically necessary but optimizes performance for the simple case.
        if exclude is not None or self.__exclude_fields__ is not None:
            exclude = ValueItems.merge(self.__exclude_fields__, exclude)

        if include is not None or self.__include_fields__ is not None:
            include = ValueItems.merge(self.__include_fields__, include, intersect=True)

        allowed_keys = self._calculate_keys(
            include=include, exclude=exclude, exclude_unset=exclude_unset  # type: ignore
        )
        if allowed_keys is None and not (to_dict or by_alias or exclude_unset or exclude_defaults or exclude_none):
            # huge boost for plain _iter()
            yield from self.__dict__.items()
            return

        value_exclude = ValueItems(self, exclude) if exclude is not None else None
        value_include = ValueItems(self, include) if include is not None else None

        for field_key, v in self.__dict__.items():
            if (
                    (allowed_keys is not None and field_key not in allowed_keys)
                    or (exclude_none and v is None)
                    or (exclude_defaults and getattr(self.__fields__.get(field_key), 'default', _missing) == v)
            ):
                continue
            if by_alias and field_key in self.__fields__:
                dict_key = self.__fields__[field_key].alias
            else:
                dict_key = field_key

            if v is DB_DEFAULT:
                yield dict_key, DB_DEFAULT
            if to_dict or value_include or value_exclude:
                v = self._get_value(
                    v,
                    to_dict=to_dict,
                    by_alias=by_alias,
                    include=value_include and value_include.for_element(field_key),
                    exclude=value_exclude and value_exclude.for_element(field_key),
                    exclude_unset=exclude_unset,
                    exclude_defaults=exclude_defaults,
                    exclude_none=exclude_none,
                )
            yield dict_key, v

    @classmethod
    @no_type_check
    def _get_value(
            cls,
            v: Any,
            to_dict: bool,
            by_alias: bool,
            include: Optional[Union['AbstractSetIntStr', 'MappingIntStrAny']],
            exclude: Optional[Union['AbstractSetIntStr', 'MappingIntStrAny']],
            exclude_unset: bool,
            exclude_defaults: bool,
            exclude_none: bool,
    ) -> Any:
        if to_dict and isinstance(v, AdaptedType):
            return v.adapt()
        return super()._get_value(
            v,
            to_dict=to_dict,
            by_alias=by_alias,
            include=include,
            exclude=exclude,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
        )

    @classmethod
    def _minimal(cls, obj: Union[Dict, List], is_schema_obj: bool = True):
        """ Leave out title and description fields when not needed """
        if isinstance(obj, dict):
            return {k: cls._minimal(v, not (k in ('properties', 'definitions') and is_schema_obj))
                    for k, v in obj.items() if not is_schema_obj or k not in ('title', 'description')}
        elif isinstance(obj, list):
            return [cls._minimal(item, True) for item in obj]
        return obj

    @classmethod
    def schema(
            cls, by_alias: bool = True, ref_template: str = default_ref_template, minimal: bool = False
    ) -> 'DictStrAny':
        s = super().schema(by_alias=by_alias, ref_template=ref_template)
        return cls._minimal(s) if minimal else s

    @classmethod
    def schema_json(
            cls,
            *,
            by_alias: bool = True,
            ref_template: str = default_ref_template,
            minimal: bool = False,
            **dumps_kwargs: Any
    ) -> str:
        from pydantic.json import pydantic_encoder

        return cls.__config__.json_dumps(
            cls.schema(by_alias=by_alias, ref_template=ref_template, minimal=minimal),
            default=pydantic_encoder, **dumps_kwargs,
        )

    class Config:
        json_encoders = {
            DBDefault: str,
            Decimal: str,
            PhoneNumber: str,
        }


ENCODERS_BY_TYPE[DBDefault] = str
ENCODERS_BY_TYPE[Decimal] = str


def get_field_info_schema(field: ModelField, schema_overrides: bool = False) -> Tuple[Dict[str, Any], bool]:
    # If no title is explicitly set, we don't set title in the schema for enums.
    # The behaviour is the same as `BaseModel` reference, where the default title
    # is in the definitions part of the schema.
    schema_: Dict[str, Any] = {}
    if field.field_info.title or not lenient_issubclass(field.type_, Enum):
        schema_['title'] = field.field_info.title or field.alias.title().replace('_', ' ')

    if field.field_info.title:
        schema_overrides = True

    if field.field_info.description:
        schema_['description'] = field.field_info.description
        schema_overrides = True

    if (
            not field.required
            and not field.field_info.const
            and field.default is not None
            and not is_callable_type(field.outer_type_)
            and field.default is not DB_DEFAULT
    ):
        schema_['default'] = encode_default(field.default)
        schema_overrides = True

    return schema_, schema_overrides


import pydantic.schema

pydantic.schema.get_field_info_schema = get_field_info_schema

# This reference to the actual pydantic field_type_schema method is only loaded once
pydantic_field_type_schema = pydantic.schema.field_type_schema


def patch_pydantic_field_type_schema() -> None:
    """
    This ugly patch fixes the serialization of models containing Optional in them.
    https://github.com/samuelcolvin/pydantic/issues/1270
    """
    warnings.warn("Patching fastapi.applications.get_openapi")
    import fastapi.applications
    from pdrdb.openapi.utils import get_openapi
    fastapi.applications.get_openapi = get_openapi

    warnings.warn("Patching pydantic.schema.field_type_schema")

    def field_type_schema(field: ModelField, **kwargs):
        f_schema, definitions, nested_models = pydantic_field_type_schema(field, **kwargs)
        if field.allow_none:
            s_type = f_schema.get('type')
            if s_type:
                # Hack to detect whether we are generating for openapi
                # fastapi sets the ref_prefix to '#/components/schemas/'.
                # When using for openapi, swagger does not seem to support an array
                # for type, so use anyOf instead.
                if kwargs.get('ref_prefix') == '#/components/schemas/':
                    f_schema = {'anyOf': [f_schema, {"type": "null"}]}
                else:
                    if not isinstance(s_type, list):
                        f_schema['type'] = [s_type]
                    f_schema['type'].append("null")
            elif "$ref" in f_schema:
                f_schema["anyOf"] = [
                    {**f_schema},
                    {"type": "null"},
                ]
                del f_schema["$ref"]

            elif "allOf" in f_schema:
                f_schema['anyOf'] = f_schema['allOf']
                del f_schema['allOf']
                f_schema['anyOf'].append({"type": "null"})

            elif "anyOf" in f_schema or "oneOf" in f_schema:
                one_or_any = f_schema.get('anyOf') or f_schema.get('oneOf')
                for item in one_or_any:
                    if item.get('type') == 'null':
                        break
                else:
                    one_or_any.append({"type": "null"})

        return f_schema, definitions, nested_models

    pydantic.schema.field_type_schema = field_type_schema


patch_pydantic_field_type_schema()

T = typing.TypeVar('T')


@overload
def optional(*fields: str) -> Callable[[T], T]:
    ...


@overload
def optional(func: T) -> T:
    ...


def optional(*fields):
    def dec(_cls: T) -> T:
        for f in fields:
            _cls.__fields__[f].required = False
            _cls.__fields__[f].allow_none = True
            _cls.__annotations__[f] = Optional[_cls.__fields__[f].type_]
        return _cls

    if fields and inspect.isclass(fields[0]) and issubclass(fields[0], BaseModel):
        cls = fields[0]
        fields = cls.__fields__
        return dec(cls)
    return dec


def exclude(*fields):
    def dec(cls: Type[BaseModel]):
        for f in fields:
            cls.__fields__.pop(f)
            cls.__annotations__.pop(f, None)
        return cls

    return dec


from asyncpg.types import Range


def parse_datetime_for_range(dt):
    dt = parse_datetime(dt)
    return dt


class DateTimeTZRange:
    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        prop_type = {'type': 'string', 'format': 'date-time'}
        field_schema.update(type='object', properties={'from': prop_type, 'to': prop_type})

    @classmethod
    def __get_validators__(cls) -> 'CallableGenerator':
        yield cls.validate

    @classmethod
    def validate(cls, value) -> Range:
        if isinstance(value, Range):
            if value.lower == datetime.datetime.min or value.upper == datetime.datetime.max:
                lower = value.lower if value.lower != datetime.datetime.min else None
                upper = value.upper if value.upper != datetime.datetime.max else None
                return Range(lower, upper)
            return value
        else:
            if isinstance(value, (list, tuple)):
                try:
                    lower, upper = value
                except ValueError:
                    raise ValueError('invalid array value for Range. Must be exactly 2 items [from, to].')
            elif isinstance(value, dict):
                lower, upper = value.get('from'), value.get('to')
            elif isinstance(value, str):
                # s = '["2021-06-03 16:19:12.379+00","2023-07-03 16:19:12.379+00")'
                s = value
                if s == 'empty':
                    return Range(empty=True)
                elif s[0] in '([' and s[-1] in ')]':
                    s = s[1:-1]
                    lower, upper = s.split(',')
                    if lower:
                        lower = json.loads(lower)
                    else:
                        lower = None
                    if upper:
                        upper = json.loads(upper)
                    else:
                        upper = None
                else:
                    raise ValueError('invalid value for type DateTimeRange')
            else:
                raise ValueError('invalid value for type DateTimeRange')

            errors = []

            if lower is not None:
                try:
                    lower = parse_datetime_for_range(lower)
                except DateTimeError as e:
                    errors.append(ErrorWrapper(e, loc=('from',)))
            if upper is not None:
                try:
                    upper = parse_datetime_for_range(upper)
                except DateTimeError as e:
                    errors.append(ErrorWrapper(e, loc=('to',)))
            if errors:
                raise ValidationError(errors, BaseModel)
            return Range(lower, upper)

    def __json__(self):
        return {"from": self}


def range_to_dict(r: Range):
    return {
        "from": jsonable_encoder(r.lower),
        "to": jsonable_encoder(r.upper),
    }


ENCODERS_BY_TYPE[Range] = range_to_dict


def return_as_is(r: T) -> T:
    return r


DB_CUSTOM_ENCODERS_BY_TYPE = {
    Range: return_as_is,
    datetime.datetime: return_as_is,
    datetime.date: return_as_is,
    datetime.time: return_as_is,
    datetime.timedelta: return_as_is,
}

dbable_encoder = functools.partial(jsonable_encoder, custom_encoder=DB_CUSTOM_ENCODERS_BY_TYPE)
