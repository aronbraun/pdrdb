import types
from typing import Union

import sqlalchemy.orm
from sqlalchemy import MetaData, event, create_engine
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.ext.automap import automap_base
import sqlalchemy.dialects.postgresql.asyncpg

from settings import DATABASE
from .custom_types import reflect_load_composite_types

REFLECT_SCHEMAS = DATABASE.REFLECT_SCHEMAS


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

sync_engine = create_engine(
    SQLALCHEMY_DATABASE_URL.replace('postgresql+asyncpg://', 'postgresql://'),
    connect_args={"options": "-csearch_path=public"},
)


metadata = MetaData()


TABLES = {}


@event.listens_for(metadata, 'column_reflect')
def receive_column_reflect(_inspector, table, column_info):
    TABLES[table] = table
    if type(column_info['type']).__name__ == 'UUID':
        column_info['type'].as_uuid = True


Base: sqlalchemy.ext.automap.AutomapBase = automap_base(metadata=metadata)


def reflect(schemas: Union[str, list, tuple] = REFLECT_SCHEMAS, custom_types_module: types.ModuleType = None):

    db_engine = sync_engine

    # async def _reflect(schemas):
    if isinstance(schemas, str):
        schemas = [s.strip() for s in schemas.split(',') if s]
    reflect_load_composite_types(db_engine=db_engine, schemas=None, custom_types_module=custom_types_module)

    for schema in schemas:
        metadata.reflect(schema=schema, views=True, bind=db_engine)
    engine.dispose()
