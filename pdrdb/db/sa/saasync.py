import os
import sys
from collections import defaultdict
from ipaddress import IPv4Address
from typing import Optional, Mapping, Union, Dict
from typing import TYPE_CHECKING, Any, List, NamedTuple, Type
from uuid import UUID

from sqlalchemy import Column
from sqlalchemy import event
from sqlalchemy.types import UserDefinedType

if TYPE_CHECKING:
    # pip install sqlalchemy-stubs
    ColumnType = Column[Any]
else:
    ColumnType = Column

import psycopg2.extensions
import sqlalchemy.orm
from sqlalchemy import MetaData
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.dialects.postgresql.asyncpg import (
    AsyncAdapt_asyncpg_dbapi,
    AsyncAdapt_asyncpg_connection,
    AsyncpgJSONB as _AsyncpgJSONB,
    AsyncpgJSON as _AsyncpgJSON,
)
from sqlalchemy.engine.result import Result
from sqlalchemy.engine.row import Row
from sqlalchemy.ext.asyncio import create_async_engine, session
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.sql import Executable, ClauseElement
from sqlalchemy.util import EMPTY_DICT, greenlet_spawn

from .. import DATABASE, REFLECT_SCHEMAS
from ...pydantic_ext import PhoneNumber
from ...utils import classorinstancemethod


class _PGUUID(PGUUID):
    def bind_processor(self, dialect):
        if not self.as_uuid:

            def process(value):
                if value is not None:
                    value = UUID(value)
                return value

            return process

    def result_processor(self, dialect, coltype):
        if not self.as_uuid and dialect.use_native_uuid:

            def process(value):
                if value is not None:
                    value = str(value)
                return value

            return process


class AsyncpgJSON(_AsyncpgJSON):
    def bind_processor(self, dialect):
        return None


class AsyncpgJSONB(_AsyncpgJSONB):
    def bind_processor(self, dialect):
        return None


sqlalchemy.dialects.postgresql.asyncpg.dialect.colspecs.update({
    sqlalchemy.sql.sqltypes.JSON: AsyncpgJSON,
    sqlalchemy.dialects.postgresql.json.JSONB: AsyncpgJSONB,
})

sqlalchemy.dialects.postgresql.asyncpg.dialect.ischema_names.update({
    'prefix_range': sqlalchemy.sql.sqltypes.TEXT,
})

SQLALCHEMY_DATABASE_URL = DATABASE.DSN

engine = create_async_engine(
    SQLALCHEMY_DATABASE_URL.replace('postgresql://', 'postgresql+asyncpg://'),
    connect_args={"server_settings": {
        'search_path': 'public',
        'timezone': 'UTC',
        'application_name': 'fastapi'
    },
    },
    # echo=True,
)

# sync_engine = create_engine(
#     SQLALCHEMY_DATABASE_URL.replace('postgresql+asyncpg://', 'postgresql://'),
#     connect_args={"options": "-csearch_path=public"},
# )

metadata = MetaData(bind=engine)

TABLES = {}


@event.listens_for(metadata, 'column_reflect')
def receive_column_reflect(inspector, table, column_info):
    TABLES[table] = table
    if type(column_info['type']).__name__ == 'UUID':
        column_info['type'].as_uuid = True


Base: sqlalchemy.ext.automap.AutomapBase = automap_base(metadata=metadata)

_reflected = False

import threading

_LOCK = threading.Lock()


def reflect_load_composite_types(db_engine: sqlalchemy.engine.Engine, schemas: List[str] = None):
    with db_engine.connect() as conn:
        result = conn.execute(sqlalchemy.text("""
WITH types AS (
    SELECT
        n.nspname,
        pg_catalog.format_type ( t.oid, NULL ) AS obj_name,
        CASE
            WHEN t.typrelid != 0 THEN CAST ( 'tuple' AS pg_catalog.text )
            WHEN t.typlen < 0 THEN CAST ( 'var' AS pg_catalog.text )
            ELSE CAST ( t.typlen AS pg_catalog.text )
            END AS obj_type,
        coalesce ( pg_catalog.obj_description ( t.oid, 'pg_type' ), '' ) AS description
    FROM
        pg_catalog.pg_type t
        JOIN pg_catalog.pg_namespace n
            ON n.oid = t.typnamespace
    WHERE ( t.typrelid = 0
            OR ( SELECT c.relkind = 'c'
                    FROM pg_catalog.pg_class c
                    WHERE c.oid = t.typrelid ) )
        AND NOT EXISTS (
                SELECT 1
                    FROM pg_catalog.pg_type el
                    WHERE el.oid = t.typelem
                    AND el.typarray = t.oid )
        AND n.nspname <> 'pg_catalog'
        AND n.nspname <> 'information_schema'
        AND n.nspname !~ '^pg_toast'
),
cols AS (
    SELECT n.nspname::text AS schema_name,
            pg_catalog.format_type ( t.oid, NULL ) AS obj_name,
            a.attname::text AS column_name,
            pg_catalog.format_type ( a.atttypid, a.atttypmod ) AS data_type,
            a.attnotnull AS is_required,
            a.attnum AS ordinal_position,
            pg_catalog.col_description ( a.attrelid, a.attnum ) AS description
        FROM pg_catalog.pg_attribute a
        JOIN pg_catalog.pg_type t
            ON a.attrelid = t.typrelid
        JOIN pg_catalog.pg_namespace n
            ON ( n.oid = t.typnamespace )
        JOIN types
            ON ( types.nspname = n.nspname
                AND types.obj_name = pg_catalog.format_type ( t.oid, NULL ) )
        WHERE a.attnum > 0
            AND NOT a.attisdropped
)
SELECT
    cols.schema_name,
    cols.obj_name,
    cols.column_name,
    cols.data_type,
    cols.ordinal_position,
    cols.is_required,
    coalesce ( cols.description, '' ) AS description
    FROM
        cols
    WHERE
        (:schemas)::text[] IS NULL OR cols.schema_name = ANY((:schemas)::text[]) -- your schema here
    ORDER BY
        cols.schema_name,
        cols.obj_name,
        cols.ordinal_position

        """).bindparams(schemas=schemas))

    result = result.fetchall()
    type_defs = defaultdict(list)
    for row in result:
        type_defs[row.obj_name].append(dict(row))

    generated_types = []
    for name, columns in type_defs.items():
        typ_def_string = generate_custom_type(name, columns)
        generated_types.append(typ_def_string)

    import generated_models
    file_path = os.path.join(str(generated_models.__path__[0]), 'custom_generated_types.py')

    imports = [
        'from typing import NamedTuple, List, cast, Type',
        'from sqlalchemy import Column',
        'from pdrdb.db.sa.saasync import CompositeType',
        'from collections import namedtuple',
    ]

    new_content = ''.join([
        '\n'.join(imports),
        '\n\n\n',
        '\n\n'.join(generated_types),
    ])

    with open(file_path, 'r+') as f:
        content = f.read()
        if content != new_content:
            f.seek(0)
            f.write(new_content)

    from generated_models import custom_generated_types
    for name in type_defs:
        class_name = name.replace('.', '_').title()
        sqlalchemy.dialects.postgresql.asyncpg.dialect.ischema_names[name] = getattr(custom_generated_types, class_name)


class CompositeType(UserDefinedType):
    """Represents a PostgreSQL composite type."""

    python_type = tuple

    name: str
    columns: List[ColumnType] = []
    type_cls: Type[NamedTuple]

    def get_col_spec(self, **kw):
        return f'name.title(name={self.name!r}, columns={self.columns!r})'


# def composite_type_factory(name: str, columns: List[ColumnType]):
#     for column in columns:
#         column.key = column.name
#
#     # def init(self, *args, **kwargs):
#     #     pass
#
#     type_cls = namedtuple(name.replace('.', '_'), [str(c.name) for c in columns])
#
#     composite = type(
#         name.replace('.', '_'),
#         (CompositeType,),
#         {"name": name, "columns": columns, "type_cls": type_cls, },
#     )
#     return composite


custom_types = {}


def generate_custom_type(name: str, columns: List[Dict]):
    # for column in columns:
    #     column.key = column.name

    col_defs = [
        f"""Column({col['column_name']!r}, type_={col['data_type']!r})"""
        for col in columns
    ]
    col_defs_str = ',\n        '.join(col_defs)

    class_def = f"""class {name.replace('.', '_').title()}(CompositeType):
    name: str = {name!r}
    columns: List[Column] = [
        {col_defs_str}
    ]
    type_cls = cast(Type[NamedTuple], namedtuple(name.replace('.', '_'), [str(c.name) for c in columns]))
    """
    custom_types[name] = class_def
    return class_def


def _reflect(schemas: Union[str, list, tuple] = REFLECT_SCHEMAS, rerun=False):
    global _reflected
    if _reflected:
        return

    import asyncio
    loop = asyncio.new_event_loop()

    # async def _reflect(schemas):
    if isinstance(schemas, str):
        schemas = [s.strip() for s in schemas.split(',') if s]

    fut = greenlet_spawn(reflect_load_composite_types, engine.sync_engine, schemas=None)
    loop.run_until_complete(fut)

    for schema in schemas:
        fut = greenlet_spawn(metadata.reflect, schema=schema, views=True, bind=engine.sync_engine)
        loop.run_until_complete(fut)
    loop.run_until_complete(engine.dispose())
    loop.close()

    _reflected = True


# Run the reflection in a separate thread while we lock so we can run another event loop but still block
def reflect(schemas: Union[str, list, tuple] = REFLECT_SCHEMAS, rerun=False):
    return ThreadedRun().setup_run_in_thread(_reflect, schemas, rerun=rerun)


class ThreadedRun:
    def __init__(self):
        self.returned = None

    def _my_target(self, func, *args, **kwargs):
        self.returned = func(*args, **kwargs)

    def setup_run_in_thread(self, func, *args, **kwargs):
        print("running {func!r}", func, file=sys.stderr)
        with _LOCK:
            t = threading.Thread(target=self._my_target, args=(func, *args,), kwargs=kwargs)
            t.start()
            t.join()
            return self.returned


def setup_run_in_event_loop(func, *args, **kwargs):
    print("running {func!r}", func, file=sys.stderr)
    with _LOCK:
        result = execute_in_new_loop(func, *args, **kwargs)
        return result


class DBDescriptor:
    def __init__(self, func):
        self._func = func

    def __get__(self, instance: Optional['DBSA'], owner: 'DBSA'):
        return self.callmethod(instance or owner.conn)

    def callmethod(self, instance_or_owner: 'DBSA'):
        async def call(*args, **kwargs):
            async with instance_or_owner as conn:
                return await conn.execute_wrapped(self._func(*args, **kwargs))

        return call


class SessionDescriptor:
    def __init__(self, name='conn'):
        self.name = name

    def __get__(self, instance, owner) -> session.AsyncSession:
        conn = session.AsyncSession(engine)
        if instance:
            setattr(instance, self.name, conn)
        return conn


def _handle_exception(self, error):
    if self._connection.is_closed():
        self._transaction = None
        self._started = False

    if not isinstance(error, AsyncAdapt_asyncpg_dbapi.Error):
        exception_mapping = self.dbapi._asyncpg_error_translate

        for super_ in type(error).__mro__:
            if super_ in exception_mapping:
                translated_error = exception_mapping[super_](
                    "%s: %s" % (type(error), error),
                    error,
                )
                raise translated_error from error
        else:
            raise error
    else:
        raise error


AsyncAdapt_asyncpg_connection._handle_exception = _handle_exception


class DBSA:
    conn = SessionDescriptor('conn')

    @classorinstancemethod
    async def execute(
            self,
            statement: Executable,
            params: Optional[Mapping] = None,
            execution_options: Mapping = EMPTY_DICT,
            bind_arguments: Optional[Mapping] = None,
            **kw
    ) -> Result:
        async with self.conn as conn:
            result = await conn.execute_wrapped(statement, params, execution_options, bind_arguments, **kw)
            return result

    @classorinstancemethod
    async def one(
            self,
            statement: Executable,
            params: Optional[Mapping] = None,
            execution_options: Mapping = EMPTY_DICT,
            bind_arguments: Optional[Mapping] = None,
            **kw
    ) -> Row:
        async with self.conn as conn:
            result = await conn.execute_wrapped(statement, params, execution_options, bind_arguments, **kw)
            return result.fetchone()

    @classorinstancemethod
    async def all(
            self,
            statement: Executable,
            params: Optional[Mapping] = None,
            execution_options: Mapping = EMPTY_DICT,
            bind_arguments: Optional[Mapping] = None,
            **kw
    ) -> List[Row]:
        async with self.conn as conn:
            result = await conn.execute_wrapped(statement, params, execution_options, bind_arguments, **kw)
            return result.fetchall()

    async def count_estimate(self, query: ClauseElement):
        # Fails, so simply return and done.
        return 0
        compiled = query.compile()
        sql, sql_params, positiontup = compiled.string, compiled.params, compiled.positiontup
        empty_args = tuple('NULL' for _ in positiontup)
        processed_sql = sql % empty_args
        connection = await self.conn.connection()
        est_count_stmt = f"SELECT count_estimate( $func$ {sql} $func$ );"
        count_result = (await connection.exec_driver_sql(est_count_stmt)).fetchone()
        count_estimate = count_result[0]
        print(sql, count_estimate)
        return count_estimate

    def __init__(self, auto_commit=False):
        self.auto_commit = auto_commit

    async def __aenter__(self):
        return await self.conn.__aenter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None and self.auto_commit:
            await self.conn.commit()
        return await self.conn.__aexit__(exc_type, exc_val, exc_tb)


def execute_in_new_loop(func, *args, **kwargs):
    import asyncio
    loop = asyncio.new_event_loop()
    fut = func(*args, **kwargs)
    result = loop.run_until_complete(fut)
    loop.close()
    return result


def register_type_adapter(qualified_name: str, cast_function: callable):
    sql = f"SELECT typrelid, typarray FROM pg_type WHERE typrelid = '{qualified_name}'::regclass::oid"

    async def query():
        async with engine.connect() as conn:
            typrelid, typarray = (await conn.exec_driver_sql(sql)).one()
        await engine.dispose()
        return typrelid, typarray

    typrelid, typarray = ThreadedRun().setup_run_in_thread(execute_in_new_loop, query)
    new_type = psycopg2.extensions.new_type((typrelid,), qualified_name, cast_function)
    new_array_type = psycopg2.extensions.new_array_type((typarray,), f'{qualified_name}[]', new_type)
    psycopg2.extensions.register_type(new_type, None)
    psycopg2.extensions.register_type(new_array_type, None)


def parse_sql_row_to_list(value: str):
    value = value[1:-1]
    items = []
    quoted = False
    current_item = []
    for c in value:
        if c == ',' and not quoted:
            val = ''.join(current_item) if current_item else None
            items.append(val)
            current_item = []
        elif c == '"':
            quoted = not quoted
            if quoted and current_item:
                current_item.append('"')
        else:
            current_item.append(c)
    if items:
        val = ''.join(current_item) if current_item else None
        items.append(val)
    return items


_B = {'f': False, 't': True}


def parse_pg_bool(b: str):
    return _B.get(b.lower())


def adapt_as_str(v):
    return psycopg2.extensions.QuotedString(str(v))


psycopg2.extensions.register_adapter(PhoneNumber, adapt_as_str)
psycopg2.extensions.register_adapter(IPv4Address, adapt_as_str)
# psycopg2.extensions.register_adapter(dict, psycopg2.extras.Json)
