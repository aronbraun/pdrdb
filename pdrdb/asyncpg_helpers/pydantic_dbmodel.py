from __future__ import annotations

import typing
from typing import Optional, List, Type, TypeVar, Any, ClassVar, Generic, Generator, Union

import asyncpg
import sqlalchemy
from sqlalchemy.dialects.postgresql import Insert
from sqlalchemy.sql import Update, Select, Delete
from sqlalchemy.sql.base import ColumnCollection

from pdrdb.db.implementation import get_db, DBIntegrity
from app.helpers.query_helpers import ALL_COLUMNS
from app.helpers.exceptions import HTTP404
from app.pydantic_ext import BaseModel, dbable_encoder

TSelf = TypeVar('TSelf')
T = TypeVar('T')
TableType = TypeVar('TableType', bound=sqlalchemy.Table)
ColumnCollectionType = TypeVar('ColumnCollectionType', bound=ColumnCollection)

KT = TypeVar('KT')
VT = TypeVar('VT')


class ColumnMapping(ColumnCollection):
    def __getitem__(self, key: str) -> sqlalchemy.Column:
        ...

    def __getattr__(self, item: str) -> sqlalchemy.Column:
        ...

    def __getattribute__(self, item) -> sqlalchemy.Column:
        ...


class ColumnAccessor:
    def __get__(self, instance, owner) -> ColumnMapping:
        c = owner.__table__.c
        return c


class TableAccessor:
    def __get__(self, instance, owner) -> TableType:
        t = owner.__table__
        return t


class ModelAccessor:
    def __get__(self, instance, owner) -> TableType:
        t = owner.__model__
        return t


class DBModel(BaseModel):
    __table__: TableType
    __model__ = None

    c: ClassVar[ColumnMapping] = ColumnAccessor()
    t: ClassVar[sqlalchemy.Table] = TableAccessor()
    m: ClassVar[Any] = ModelAccessor()

    @classmethod
    async def fetch(cls: Type[T], query, *args, timeout=None) -> List[T]:
        db = get_db()
        result = await db.fetch(query, *args, timeout=timeout)
        return db.proxy_rows(result)

    @classmethod
    async def fetchrow(cls: Type[T], query, *args, timeout=None) -> Optional[T]:
        db = get_db()
        result = await db.fetchrow(query, *args, timeout=timeout)
        return db.proxy_rows(result)

    @classmethod
    async def fetch_as_model(cls: Type[T], query, *args, timeout=None) -> List[T]:
        db = get_db()
        result = await db.fetch(query, *args, timeout=timeout)
        return [cls(**row) for row in result] if result else result

    @classmethod
    async def fetchrow_as_model(cls: Type[T], query, *args, timeout=None) -> Optional[T]:
        db = get_db()
        result = await db.fetchrow(query, *args, timeout=timeout)
        return cls(**result) if result is not None else result

    @classmethod
    async def fetchval(cls: Type[T], query, *args, timeout=None) -> Optional[Any]:
        db = get_db()
        result = await db.fetchval(query, *args, timeout=timeout)
        return result

    @classmethod
    async def execute(cls: Type[T], query, *args, timeout=None) -> Optional[Any]:
        db = get_db()
        result = await db.execute(query, *args, timeout=timeout)
        return result

    @classmethod
    async def executemany(cls: Type[T], query, args, *, timeout=None) -> Optional[Any]:
        db = get_db()
        result = await db.executemany(query, args, timeout=timeout)
        return result

    def insert(self: TSelf, exclude_unset=True, returning=ALL_COLUMNS) -> 'WrappedInsert[TSelf]':
        stmt = WrappedInsert(
            self.__table__, bound_model=self
        ).values(
            **self.dict(exclude_unset=exclude_unset)
        ).returning(returning)
        return stmt

    def update(
            self: TSelf,
            exclude: set = None,
            exclude_unset=True,
            include: set = None,
            returning=ALL_COLUMNS,
            auto_pk_where=True,
            update_values=None
    ) -> 'WrappedUpdate[TSelf]':

        stmt = WrappedUpdate(
            self.__table__, bound_model=self
        )
        exclude = exclude or set()
        if auto_pk_where:
            pk_cols = self.__model__.__table__.primary_key.columns
            wheres = []
            for col in pk_cols:
                w = col == getattr(self, col.name)
                wheres.append(w)
                exclude.add(col.name)
            if not wheres and auto_pk_where:
                raise AssertionError('Cannot update a value without primary key clause')

            stmt = stmt.where(*wheres)

        if update_values is None:
            update_values = self.dict(exclude_unset=exclude_unset, exclude=exclude, include=include)
        serialized = dbable_encoder(update_values)
        stmt = stmt.values(**serialized).returning(returning)

        return stmt

    def update_pk(
            self: TSelf,
            exclude: set = None,
            exclude_unset=True,
            include: set = None,
            returning=ALL_COLUMNS,
            update_values=None
    ) -> 'WrappedUpdate[TSelf]':
        stmt = self.update(
            exclude=exclude,
            exclude_unset=exclude_unset,
            include=include,
            returning=returning,
            auto_pk_where=False,
            update_values=update_values
        )
        return stmt

    @classmethod
    def select(cls: Type[T], *cols) -> 'WrappedSelect[T, asyncpg.Record | None]':
        stmt = WrappedSelect.create(*cols, _model=cls).select_from(cls.__table__)
        return stmt

    @classmethod
    def select_all(cls: Type[T]) -> 'WrappedSelect[T, asyncpg.Record | None]':
        stmt = WrappedSelect.create(ALL_COLUMNS, _model=cls).select_from(cls.__table__)
        return stmt

    @classmethod
    def delete_stmt(cls: Type[T], *where, **filter_by) -> WrappedDelete[T, asyncpg.Record | None]:
        stmt = WrappedDelete(cls.__table__).returning(ALL_COLUMNS)
        if where:
            stmt = stmt.where(*where)
        if filter_by:
            stmt = stmt.filter_by(**filter_by)
        return stmt


ModelType = TypeVar('ModelType', bound=DBModel)


class _WrappedStmt(Generic[T]):
    def __init__(
            self,
            table,
            bound_model: T = None,
            values=None,
            inline=False,
            bind=None,
            prefixes=None,
            returning=None,
            return_defaults=False,
            **dialect_kw
    ):
        super().__init__(
            table,
            values=values,
            inline=inline,
            bind=bind,
            prefixes=prefixes,
            returning=returning,
            return_defaults=return_defaults,
            **dialect_kw
        )
        self._bound_model = bound_model

    async def fetch(self) -> List[T]:
        if self._bound_model is not None:
            return await self._bound_model.fetch_as_model(self)
        return await get_db().fetch(self)

    async def fetchrow(self) -> T:
        if self._bound_model is not None:
            return await self._bound_model.fetchrow_as_model(self)
        return await get_db().fetchrow(self)

    async def fetchval(self):
        return await self._bound_model.fetchval(self)

    async def execute(self):
        return await self._bound_model.execute(self)

    def __await__(self) -> Generator[Any, None, T]:
        return self.fetchrow().__await__()

    def returning(self: TSelf, *cols) -> TSelf:
        self._returning = () # noqa
        return super().returning(*cols)


class WrappedInsert(_WrappedStmt, Insert):
    """Insert statements"""


class WrappedUpdate(_WrappedStmt, Update):
    """Update statements"""


BoundModelType = TypeVar('BoundModelType')
BoundRecordType = TypeVar('BoundRecordType')


class MyWrappedQuery(Generic[BoundModelType, BoundRecordType]):
    _bound_model: BoundModelType
    _as_model = False

    FetchRetType = Union[BoundModelType, BoundRecordType]

    if typing.TYPE_CHECKING:
        def __init__(self: TSelf, _model: BoundModelType):
            self._bound_model = _model

    async def fetch(self) -> List[FetchRetType]:
        res = await get_db().fetch(self)
        if self._as_model and res:
            return [self._bound_model(**row) for row in res]
        return res

    async def fetchrow(self) -> FetchRetType:
        res = await get_db().fetchrow(self)
        if self._as_model and res is not None:
            return self._bound_model(**res)
        return res

    async def fetchval(self):
        return await get_db().fetchval(self)

    async def execute(self):
        return await get_db().execute(self)

    def __await__(self) -> Generator[Any, None, FetchRetType]:
        return self.fetch().__await__()

    def as_model(self: TSelf) -> TSelf:
        # assert self._bound_model is not None, 'You can only get as_model when selecting all columns'
        self._as_model = True
        return self


class WrappedSelect(MyWrappedQuery, Select):
    @classmethod
    def create(cls, *entites, _model: Type[DBModel]):
        # noinspection PyUnresolvedReferences
        # noinspection PyProtectedMember
        instance = super()._create_select(*entites)
        instance._bound_model = _model
        return instance


class WrappedDelete(MyWrappedQuery, Delete):
    error_detail = 'Not Found'
    _force = False

    def with_error_detail(self, error_detail):
        self.error_detail = error_detail
        return self

    def __await__(self) -> Generator[Any, None, T]:
        return self._await().__await__()

    async def _await(self):
        if not self._force and not self._where_criteria:
            raise AssertionError('You missed a where clause!!!!!!! do not delete your entire space!!!!!')
        async with DBIntegrity({}) as sess:
            response = await sess.fetch(self)
            if not response:
                raise HTTP404(detail=self.error_detail)
            return response
