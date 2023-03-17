import typing
import typing as t
from functools import reduce

import phonenumbers
import pydantic
from pydantic import BaseModel, validator, ConstrainedList
from pydantic.fields import ModelField
from typing_extensions import Protocol

if typing.TYPE_CHECKING:
    from pydantic.typing import DictStrAny

__all__ = ["ValidatedValue", "WrappedValue", "PhoneNumber", "AdaptedType", "CommaDelimitedList"]

from pdrdb.pydantic_ext.validators import validate_phone_number

T = t.TypeVar("T")
Validator = t.Callable[[T], T]


class Validated(Protocol[T]):
    """Any type that implements __get_validators__"""

    __validators__: t.List[Validator[T]]

    @classmethod
    def __get_validators__(cls) -> t.Iterator[Validator[T]]:
        """Retrieve validators for this item."""
        ...


class ValidatedValue(t.Generic[T]):
    """A container for a validated value of some type."""

    __validators__: t.List[Validator[T]]

    @classmethod
    def __get_validators__(cls) -> t.Iterator[Validator[T]]:
        """Retrieve validators for this value's type."""
        yield from cls.__validators__

    def __init__(self, val: t.Union[T, 'ValidatedValue[T]']) -> None:
        if isinstance(val, type(self)):
            self.value: T = val.value
        else:
            self.value: T = validate(type(self), val)  # type: ignore


class WrappedValue(t.Generic[T]):
    """A container for a validated value of some type."""

    __validators__: t.List[Validator[T]]

    @classmethod
    def __get_validators__(cls) -> t.Iterator[Validator[T]]:
        """Retrieve validators for this value's type."""
        yield cls

    def __init__(self, val: t.Union[T, 'WrappedValue[T]']) -> None:
        if isinstance(val, type(self)):
            self.value: T = val.value
        else:
            self.value: T = validate(type(self), val)  # type: ignore


def validate(validated_type: t.Type[Validated[T]], val: T) -> T:
    """Validate the value against the Pydantic __get_validators__ method."""
    return reduce(lambda x, y: y(x), validated_type.__validators__, val)


class AdaptedType:
    def adapt(self):
        raise NotImplementedError('You should implement this')


class DelimitedList(
    # typing.Generic[_T],  #  this is not needed ?
    ConstrainedList
):
    """
    # pydantic creates new subclasses for any List type on the fly using ConstrainedList as the base
    # unless, it is already a ConstrainedList subclass. In order not to lose our custom
    # __get_validators__ on our class, we subclass from ConstrainedList.
    # note, if we need to set list constraints, that should be subclassed further and set
    # the class attributes max_length, min_length, unique_items accordingly.
    #
    # for more info, see  the following functions what they do
    # from fastapi.dependencies.utils import is_scalar_sequence_field, create_response_field, get_dependant
    # from fastapi.dependencies.utils import is_scalar_sequence_field, create_response_field, get_dependant
    """

    delimiter = ' '
    model = BaseModel

    @classmethod
    def __get_validators__(cls):
        yield cls.validate
        yield from ConstrainedList.__get_validators__()

    @classmethod
    # You don't need to add the "ModelField", but it will help your
    # editor give you completion and catch errors
    def validate(cls, v, field: ModelField):
        if v and len(v) == 1 and cls.delimiter in v[0]:
            v = v[0].split(cls.delimiter)

        errors = []
        values = []
        for i, val in enumerate(v):
            value_, error = field.sub_fields[0].validate(v[i], {}, loc=(i,))
            values.append(value_)
            if error:
                errors.append(error)
        if errors:
            raise pydantic.ValidationError(errors, cls.model)
        return values


class CommaDelimitedList(DelimitedList):
    delimiter = ','


class PhoneNumber(WrappedValue[str], AdaptedType):

    __validators__ = [validate_phone_number]
    value: phonenumbers.PhoneNumber

    def __str__(self):
        return self.e164()

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.e164())

    def for_db(self):
        return self.e164()

    def digits(self):
        """E164 without the + prefix"""
        return self.value.country_code + self.value.national_number

    def e164(self):
        return phonenumbers.format_number(self.value, phonenumbers.PhoneNumberFormat.E164)

    def national_format(self):
        return phonenumbers.format_number(self.value, phonenumbers.PhoneNumberFormat.NATIONAL)

    def international_format(self):
        return phonenumbers.format_number(self.value, phonenumbers.PhoneNumberFormat.INTERNATIONAL)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update({
            "type": "string",
            "format": "Phone Number E164 Format",
            "example": "+19015551212"
        })

    def adapt(self):
        return self.for_db()


class PhoneNumber2(BaseModel):
    __root__: phonenumbers.PhoneNumber

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update({
            "type": "string",
            "format": "Phone Number E164 Format",
        })

    @validator('__root__', pre=True)
    def validate_phone(cls, value, values):
        print(values)
        return validate_phone_number(value)

    class Config(BaseModel.Config):
        arbitrary_types_allowed = True

    def dict(
        self,
        **kwargs,
    ) -> 'DictStrAny':
        return phonenumbers.format_number(self.__root__, phonenumbers.PhoneNumberFormat.E164)

    def __str__(self):
        return self.e164()

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.e164())

    @property
    def value(self):
        return self.__root__

    @value.setter
    def value(self, val):
        self.__root__ = val

    def e164(self):
        return phonenumbers.format_number(self.value, phonenumbers.PhoneNumberFormat.E164)

    def national_format(self):
        return phonenumbers.format_number(self.value, phonenumbers.PhoneNumberFormat.NATIONAL)

    def international_format(self):
        return phonenumbers.format_number(self.value, phonenumbers.PhoneNumberFormat.INTERNATIONAL)
