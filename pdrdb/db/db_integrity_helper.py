from typing import Union

import asyncpg
from asyncpg.exceptions import PostgresError
from pydantic import BaseModel

import pdrdb.db.sa.asyncpg as asyncpg_imp
from pdrdb.db import implementation
from pdrdb.db.exception_handlers import handle_db_exception


class DBIntegrityAsyncPG:
    def __init__(self, data: Union[dict, list, BaseModel], sess: asyncpg_imp.MySingleDBConnectionClass = None):
        if isinstance(data, BaseModel):
            self.data = data.dict()
        elif isinstance(data, (dict, list)):
            self.data = data
        else:
            raise AssertionError('data must be type of dict or list or BaseModel')
        self._sess: asyncpg_imp.MySingleDBConnectionClass = sess
        self._default_pool = implementation.get_db()
        self._managed = False
        self._trans = None
        self._levels = 0

    async def __aenter__(self):
        self._managed = True
        self._levels += 1
        if self._sess is None:
            self._sess = await self._default_pool.__aenter__()
        else:
            await self._sess.__aenter__()
        if self._trans is not None:
            raise asyncpg.InterfaceError('Cannot create transaction already running')
        self._trans: asyncpg.connection.transaction.Transaction = self._sess._conn.transaction()
        await self._trans.__aenter__()
        return self._sess

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._levels -= 1
        suppress = None
        # print(exc_type, exc_val, exc_tb)
        if self._trans is not None:
            assert self._sess is not None
            self._trans: asyncpg.connection.transaction.Transaction
            try:
                suppress = await self._trans.__aexit__(exc_type, exc_val, exc_tb)
            except PostgresError as on_commit_err:
                if not exc_val and not exc_tb:
                    exc_val = on_commit_err
                    exc_type = PostgresError
            if self._levels < 1:
                self._trans = None
                if self._managed:
                    await self._default_pool.__aexit__(exc_type, exc_val, exc_tb)
                    self._sess = None
        if exc_type and issubclass(exc_type, PostgresError):
            handle_db_exception(exc_val, self.data)
        return suppress

    @classmethod
    def wrap(cls, function):
        async def wrapped_function(*args, **kwargs):
            async with cls({}) as session:
                return await function(session, *args, **kwargs)

        return wrapped_function
