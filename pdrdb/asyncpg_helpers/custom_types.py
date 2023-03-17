import importlib
import os
import types
from collections import defaultdict
from typing import List, Dict, TYPE_CHECKING, Any, NamedTuple, Type

import sqlalchemy
import sqlalchemy.dialects.postgresql.asyncpg
from sqlalchemy.engine.row import RowProxy


from sqlalchemy import Column
from sqlalchemy import event
from sqlalchemy.types import UserDefinedType

if TYPE_CHECKING:
    # pip install sqlalchemy-stubs
    ColumnType = Column[Any]
else:
    ColumnType = Column


def reflect_load_composite_types(
        db_engine: sqlalchemy.engine.Engine,
        schemas: List[str] = None,
        custom_types_module: types.ModuleType = None,
):
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

    rows: List[RowProxy] = result.fetchall()
    type_defs = defaultdict(list)
    for row in rows:
        type_defs[row['obj_name']].append(dict(row))

    generated_types = []
    for name, columns in type_defs.items():
        typ_def_string = generate_custom_type(name, columns)
        generated_types.append(typ_def_string)

    if custom_types_module:
        try:
            save_path = custom_types_module.__path__[0]
        except AttributeError:
            save_path = custom_types_module.__file__
    else:
        save_path = os.path.join(os.path.dirname(__file__), '_custom_generated_types.py')

    write_custom_types_file(generated_types, save_path)

    if not custom_types_module:
        custom_types_module = importlib.import_module('.._custom_generated_types', __name__)

    importlib.reload(custom_types_module)

    for name, klass in type_defs.items():
        class_name = name.replace('.', '_').title()
        type_class = getattr(custom_types_module, class_name)
        sqlalchemy.dialects.postgresql.asyncpg.dialect.ischema_names[name] = type_class


def write_custom_types_file(generated_types: List[str], save_path: str):

    imports = [
        'from typing import NamedTuple, List, cast, Type',
        'from sqlalchemy import Column',
        'from pdrdb.asyncpg_helpers.custom_types import CompositeType',
        'from collections import namedtuple',
    ]

    new_content = ''.join([
        '\n'.join(imports),
        '\n\n',
        '\n\n'.join(generated_types),
    ])

    with open(save_path, 'w+') as f:
        content = f.read()
        if content != new_content:
            f.seek(0)
            f.write(new_content)


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
