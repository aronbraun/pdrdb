import asyncio
import datetime
import decimal
import re
import uuid
import warnings
from typing import Union, Optional

import asyncpg
import sqlalchemy.dialects.postgresql.asyncpg
from sqlalchemy.engine import Dialect
from sqlalchemy.sql import ClauseElement

from pdrdb.db.sa import asyncpg_fixups  # noqa


def _get_dialect() -> Dialect:
    dialect = sqlalchemy.dialects.postgresql.asyncpg.dialect(paramstyle="pyformat")

    dialect.implicit_returning = True
    dialect.supports_native_enum = True
    dialect.supports_smallserial = True  # 9.2+
    dialect._backslash_escapes = False
    dialect.supports_sane_multi_rowcount = True  # psycopg 2.0.9+
    dialect._has_native_hstore = True
    dialect.supports_native_decimal = True

    return dialect


DEFAULT_DIALECT = _get_dialect()


class ExtendedConnection(asyncpg.Connection):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._trans: Optional[asyncpg.connection.transaction.Transaction] = None

    async def fetch(self, query, *args, timeout: float = None) -> list:
        with QueryContext(query, args) as (query, args):
            return await super().fetch(query, *args, timeout=timeout)

    async def fetchrow(self, query, *args, timeout: float = None) -> list:
        with QueryContext(query, args) as (query, args):
            return await super().fetchrow(query, *args, timeout=timeout)

    async def fetchval(self, query, *args, column=0, timeout: float = None):
        with QueryContext(query, args) as (query, args):
            return await super().fetchval(query, *args, column=0, timeout=timeout)

    async def execute(self, query, *args, timeout=None):
        with QueryContext(query, args) as (query, args):
            return await super().execute(query, *args, timeout=timeout)

    async def executemany(self, command: str, args, *, timeout: float = None):
        return await super().executemany(command, args, timeout=timeout)

    async def cursor(self, query, *args, prefetch=None, timeout=None):
        with QueryContext(query, args) as (query, args):
            return await super().cursor(query, *args, prefetch=prefetch, timeout=timeout)

    async def commit(self):
        if self._trans is None:
            raise asyncpg.InterfaceError('No transaction in progress')
        return await self._trans.commit()

    async def rollback(self):
        if self._trans is None:
            raise asyncpg.InterfaceError('No transaction in progress')
        return await self._trans.rollback()

    async def savepoint(self):
        if self._trans is None:
            raise asyncpg.InterfaceError('No transaction in progress')
        return await self._trans.start()

    async def count_estimate(self, query: ClauseElement):
        connection = self
        sql = await connection.compile_query_args_using_format(query)
        est_count_stmt = f"SELECT count_estimate( $func$ {sql} $func$ );"
        count_result = await connection.fetchrow(est_count_stmt)
        count_estimate = count_result[0]
        return count_estimate

    async def exec_driver_sql(self, query, params):

        if params and isinstance(params, list):
            params = params[0]
        if isinstance(params, dict):
            params = tuple(params.values())
        args_placeholders = tuple(
            "$" + str(i)
            for i, _ in enumerate(params, start=1)
        )
        compiled = query % args_placeholders
        result = await self.fetch(compiled, *params)
        return result

    async def mogrify(self, query, params):
        from asyncpg.utils import _mogrify  # noqa
        return await _mogrify(self, query, params)

    async def compile_query_args_using_format(self, query):
        with QueryContext(query, None) as (q, p):
            result = await self.mogrify(q, p)
            return result

    # @classmethod
    # def _compile(cls, query: ClauseElement, dialect=DEFAULT_DIALECT):
    #     compiled = query.compile(
    #         dialect=dialect, compile_kwargs={"render_postcompile": True}
    #     )
    #     compiled_params = sorted(compiled.params.items())
    #
    #     mapping = {
    #         key: "$" + str(i) for i, (key, _) in enumerate(compiled_params, start=1)
    #     }
    #
    #     compiled_query = compiled.string % mapping
    #
    #     processors = compiled._bind_processors  # noqa
    #     args = [
    #         processors[key](val) if key in processors else val
    #         for key, val in compiled_params
    #     ]
    #     return compiled_query, args
    #
    # @classmethod
    # def _process_stmt(cls, statement: Union[ClauseElement, str], params: Optional[Tuple] = None):
    #     if isinstance(statement, ClauseElement):
    #         if params:
    #             raise ValueError('Cannot provide params when query is a ClauseElement. Bind your params directly.')
    #         statement, params = cls._compile(statement)
    #     return statement, params


def get_sql_type(val):
    if val is None:
        return ''
    if isinstance(val, datetime.datetime):
        if val.tzinfo is not None:
            return 'timestamptz'
        else:
            return 'timestamp'
    if isinstance(val, datetime.time):
        if val.tzinfo is not None:
            return 'timetz'
        else:
            return 'time'
    if isinstance(val, uuid.UUID):
        return 'uuid'
    if isinstance(val, int):
        if val > 2 ** 32:
            return 'bigint'
        else:
            return 'int'
    if isinstance(val, decimal.Decimal):
        return 'numeric'
    if isinstance(val, dict):
        return 'jsonb'
    if isinstance(val, list):
        return 'jsonb'


def _quote_ident(ident):
    return '"{}"'.format(ident.replace('"', '""'))


def _quote_literal(string):
    return "'{}'".format(string.replace("'", "''"))


class QueryContext:
    def __init__(self, query: Union[ClauseElement, str], params=(), dialect=DEFAULT_DIALECT):
        if isinstance(query, ClauseElement):
            if params:
                raise ValueError(
                    'Cannot provide params when query is a ClauseElement. '
                    'Bind your params directly. query: {}'.format(query)
                )
            self.clause_element = query
            self.query = None
        else:
            self.clause_element = None
            self.query = query
        self.params = params
        self.dialect = dialect

        self.compiled_clause = None
        self.compiled_params = None
        self.mapping = None
        self.compiled_query = None

        self._compile()

    def _compile(self):
        if self.clause_element is None:
            return
        if self.query is not None:
            return

        self.compiled_clause = self.clause_element.compile(
            dialect=self.dialect, compile_kwargs={"render_postcompile": True}
        )
        self.compiled_params = self.compiled_clause.params.items()

        self.mapping = {
            key: "$" + str(i) for i, (key, _) in enumerate(self.compiled_params, start=1)
        }

        self.query = self.compiled_clause.string % self.mapping

        processors = self.compiled_clause._bind_processors  # noqa
        self.params = [
            processors[key](val) if key in processors else val
            for key, val in self.compiled_params
        ]

    def __enter__(self):
        return self.query, self.params

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None and issubclass(exc_type, asyncpg.PostgresError):
            if exc_val.sqlstate == '22000' and self.mapping:
                match = re.match(r'^invalid input for query argument (\$\d+):', exc_val.args[0])
                if match:
                    key = match.group(1)
                    param_name = {k: v for v, k in self.mapping.items()}[key]
                    msg = (exc_val.args[0] or '') + ' (key: {})'.format(param_name)
                    exc_val.args = (msg,) + exc_val.args[1:]
            exc_val.__dict__['query_context'] = {
                'clause_element': self.clause_element,
                'query': self.query,
                'params': self.params,
                'compiled_clause': self.compiled_clause,
                'compiled_params': self.compiled_params,
                'mapping': self.mapping,
                'compiled_query': self.compiled_query,
            }


server_settings = {
    'search_path': 'public',
    'timezone': 'UTC',
    'application_name': 'fastapi'
}

try:
    import ujson as my_json


    def json_encode(content):
        return my_json.dumps(content, ensure_ascii=False).encode('utf-8')

except ImportError:
    import json as my_json


    def json_encode(content):
        return my_json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        ).encode('utf-8')


def json_decode(string):
    return my_json.loads(string)


def jsonb_encode(x):
    return b'\x01' + json_encode(x)


def jsonb_decode(x):
    return my_json.loads(x[1:])


def decode_prefix_range(x):
    prefix = x[2:-1].decode('utf-8')
    first = x[0]
    last = x[1]

    if first or last:
        val = f'{prefix}[{chr(first)}-{chr(last)}]'
    else:
        val = prefix
    # print(f'decoded from {x} -> {val}', file=sys.stderr)
    return val


def encode_prefix_range(x):
    first = '\x00'
    last = '\x00'
    for o in x:
        if o > '\xff':
            raise ValueError('prefix_range does not support unicode bytes > 0xff')
    if '[' in x or ']' in x:
        try:
            prefix, bracket_part = x.split('[')
        except ValueError:
            raise ValueError(f'Too many [ in prefix_range value {x}')
        try:
            range_part, closing_part = bracket_part.split(']')
        except ValueError:
            raise ValueError(f'Too many or no ] in prefix_range value {x}')
        if closing_part:
            raise ValueError(f'Invalid prefix {x} should not contain anything after closing ]')

        if range_part:
            if len(range_part) != 3 or range_part[1] != '-':
                raise ValueError(f'Invalid prefix {x}. Value range must [x-y]')
            first = range_part[0]
            last = range_part[2]
            if first == last:
                prefix += first
                first = '\x00'
                last = '\x00'
    else:
        prefix = x
    codec = 'utf-8'
    val = first.encode(codec) + last.encode(codec) + prefix.encode(codec) + b'\x00'
    # print(f'encoded from {x} -> {val}', file=sys.stderr)
    return val


async def setup_asyncpg_connection(conn: ExtendedConnection):
    await conn.set_type_codec(
        'json',
        encoder=json_encode,
        decoder=json_decode,
        schema='pg_catalog',
        format='binary',
    )

    await conn.set_type_codec(
        'jsonb',
        encoder=jsonb_encode,
        decoder=jsonb_decode,
        schema='pg_catalog',
        format='binary',
    )

    # await conn.set_type_codec(
    #     'prefix_range',
    #     encoder=encode_prefix_range,
    #     decoder=decode_prefix_range,
    #     schema='public',
    #     format='binary',
    # )


class ResultException(Exception):
    pass


class ProxyRecord:
    def __init__(self, record):
        self._record = record

    def __getattr__(self, item):
        try:
            return self._record[item]
        except KeyError as e:
            raise AttributeError(*e.args)

    def __getitem__(self, item):
        return self._record[item]

    def keys(self):
        return self._record.keys()

    def values(self):
        return self._record.values()

    def items(self):
        return self._record.items()

    def dict(self, **kwargs):  # noqa
        return dict(self._record)


class CollectionWrapper(list):
    def dict(self, **kwargs):
        return [r.dict() for r in self]


class ResultWrapper:
    def __init__(self, rows):
        rows = MySingleDBConnectionClass.proxy_rows(rows)
        self.rows = rows

    def fetchall(self):
        return self.all()

    def fetchone(self):
        return self.one()

    def one(self):
        num = len(self.rows) if self.rows else 0
        if num != 1:
            raise ResultException('Invalid number of records returned. (required 1, got %d' % num)
        return self.rows[0]

    def all(self):
        return self.rows or []

    def first(self):
        return self.rows[0] if self.rows else None

    def __iter__(self):
        yield from self.rows


DSN: str = None


def init_db(dsn: str):
    global DSN
    DSN = dsn


class MySingleDBConnectionClass:
    _lock = asyncio.Lock()
    _pool = None

    @classmethod
    async def init_pool(cls):
        async with cls._lock:
            if cls._pool is None or cls._pool._closed:  # noqa
                cls._pool = asyncpg.create_pool(
                    DSN, ssl=False, min_size=1, max_size=20, init=setup_asyncpg_connection,
                    server_settings=server_settings, connection_class=ExtendedConnection
                )
            await cls._pool

    @property
    def pool(self):
        return self._pool

    def __init__(self, auto_commit=True):
        assert DSN is not None, 'Missing Database dsn'
        if not auto_commit:
            raise AssertionError('This should always be true?')
        self._levels = 0
        self._conn: Optional[ExtendedConnection] = None

    async def __aenter__(self) -> 'MySingleDBConnectionClass':
        self._levels += 1
        if self._conn is None:
            await self.init_pool()
            self._conn: ExtendedConnection = await self.pool.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._levels -= 1
        if self._levels < 1:
            await self.pool.release(self._conn)
            self._conn = None

    def begin_nested(self):
        warnings.warn('THIS METHOD IS DEPRECATED')
        assert self._conn is not None
        return self._conn.transaction()

    async def connection(self) -> ExtendedConnection:
        warnings.warn('THIS METHOD IS DEPRECATED')
        assert self._conn is not None
        return self._conn

    def transaction(self):
        assert self._conn is not None
        return self._conn.transaction()

    async def count_estimate(self, query: ClauseElement):
        async with self:
            return await self._conn.count_estimate(query)

    async def fetch(self, query, *args, timeout=None):
        async with self:
            return await self._conn.fetch(query, *args, timeout=timeout)

    async def fetchrow(self, query, *args, timeout=None):
        async with self:
            return await self._conn.fetchrow(query, *args, timeout=timeout)

    async def fetchval(self, query, *args, timeout=None):
        async with self:
            return await self._conn.fetchval(query, *args, timeout=timeout)

    async def execute(self, query, *args, timeout=None):
        async with self:
            return await self._conn.execute(query, *args, timeout=timeout)

    async def executemany(self, command, args, *, timeout=None):
        async with self:
            return await self._conn.executemany(command, args, timeout=timeout)

    async def execute_wrapped(self, query, *args, timeout=None):
        async with self:
            result = await self._conn.fetch(query, *args, timeout=timeout)
            return ResultWrapper(result)

    async def one(self, query, *args, timeout=None):
        return self.proxy_rows(await self.fetchrow(query, *args, timeout=timeout))

    async def all(self, query, *args, timeout=None):
        return self.proxy_rows(await self.fetch(query, *args, timeout=timeout))

    @staticmethod
    def proxy_rows(result):
        if result:
            if isinstance(result, list):
                return CollectionWrapper(result)
            return ProxyRecord(result)
        return result


MY_DB = MySingleDBConnectionClass
