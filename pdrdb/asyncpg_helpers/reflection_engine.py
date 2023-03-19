import types
from typing import Union

import sqlalchemy.dialects.postgresql.asyncpg
from sqlalchemy import MetaData, event, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.ext.automap import automap_base

from pdrdb.asyncpg_helpers.custom_types import reflect_load_composite_types

metadata = MetaData()

TABLES = {}


@event.listens_for(metadata, 'column_reflect')
def receive_column_reflect(_inspector, table, column_info):
    TABLES[table] = table
    if type(column_info['type']).__name__ == 'UUID':
        column_info['type'].as_uuid = True


Base: sqlalchemy.ext.automap.AutomapBase = automap_base(metadata=metadata)


def reflect(db: Engine | str, schemas: Union[str, list, tuple], custom_types_module: types.ModuleType = None):
    if isinstance(db, str):
        dsn = db
        db = create_engine(
            dsn.replace('postgresql+asyncpg://', 'postgresql://'),
            connect_args={"options": "-csearch_path=public"},
        )

    if isinstance(schemas, str):
        schemas = [s.strip() for s in schemas.split(',') if s]
    reflect_load_composite_types(db_engine=db, schemas=None, custom_types_module=custom_types_module)

    for schema in schemas:
        metadata.reflect(schema=schema, views=True, bind=db)
    db.dispose()
