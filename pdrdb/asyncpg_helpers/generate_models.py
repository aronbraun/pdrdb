import datetime
import importlib
import inspect
import os
import re
import shutil
import sys
import textwrap
import types
from collections import defaultdict
from decimal import Decimal
from typing import List, cast, Dict
from uuid import UUID

#  sqlacodegen==3.0.0b2
import sqlacodegen.generators
import sqlacodegen.models
from sqlalchemy import Table, create_engine
from sqlalchemy.sql.base import ColumnCollection

from pdrdb.asyncpg_helpers.reflection_engine import reflect, metadata as _metadata

_args = None

CUSTOM_TYPE_MAP = {}

TYPE_MAP = {
    str: 'str',
    int: 'int',
    Decimal: 'Decimal',
    'TSTZRANGE': 'DateTimeTZRange',
    bool: 'bool',
    list: 'list',
    dict: 'dict',
    float: 'float',
    'UUID': 'UUID',
    UUID: 'UUID',
    datetime.timedelta: 'datetime.timedelta',
    datetime.datetime: 'datetime.datetime',
    datetime.date: 'datetime.date',
    'REGCLASS': 'str',
    'INET': 'IPvAnyInterface',
    'NULL': 'Any',
    tuple: 'dict',
}


def get_column_python_type(column):
    try:
        return column.type.python_type
    except NotImplementedError:
        return str(column.type)


def get_pydantic_schema(tbl):
    columns = {}
    for name, column in tbl.columns.items():
        type_ = get_column_python_type(column)
        columns[name] = {'type': type_, 'column': column}

    return tbl, columns


def get_module_classes(module):
    return [val for key, val in module.__dict__.items() if not key.startswith('_')]


def column_spec(name, column: dict):
    col_type = column['column'].type
    try:
        if col_type in CUSTOM_TYPE_MAP:
            type_string = CUSTOM_TYPE_MAP[col_type]
        elif getattr(col_type, 'name', None) in CUSTOM_TYPE_MAP:
            type_string = CUSTOM_TYPE_MAP[col_type.name]
        else:
            type_string = TYPE_MAP[column['type']]
    except Exception as e:
        print(e)
        raise
    default = None
    optional = False
    if column['column'].server_default:
        default = 'DB_DEFAULT'
    if column['column'].nullable:
        optional = True

    if optional:
        type_string = f"Optional[{type_string}]"
    if default:
        type_string = f"Optional[{type_string}] = {default}"

    spec = f"{name}: {type_string}"
    return spec


def class_name_for_table(table_name: str, schema_name: str = '') -> str:
    table_name = table_name.title()
    schema_name = schema_name.title()
    name = ''.join(n for n in [schema_name, table_name] if n).replace('.', '').replace('_', '')
    return name


def generate_sa_models(base_package: types.ModuleType, dsn: str, schemas: str | list):
    engine = create_engine(
        dsn.replace('postgresql+asyncpg://', 'postgresql://'),
        connect_args={"options": "-csearch_path=public"},
    )

    package_path = base_package.__path__[0]

    custom_types_module = importlib.import_module('.custom_types', f'{base_package.__name__}.auto_generated')

    reflect(engine, schemas, custom_types_module=custom_types_module)

    auto_generated_path = os.path.join(package_path, 'auto_generated')

    models_path = os.path.join(auto_generated_path, 'sa_models')
    create_dir_recursive(models_path)

    tables_types_path = os.path.join(auto_generated_path, 'sqlalchemy_tables.pyi')

    generator = TableStubGenerator(metadata=_metadata, bind=engine, options=set())
    generator.write_to_file(tables_types_path)

    generator = DeclarativeGenerator(base_package=base_package, metadata=_metadata, bind=engine, options={'nojoined'})
    generator.write_to_model_schema_files(models_path)

    generate_sa_tables_from_sa_models(base_package)


def generate_sa_tables_from_sa_models(base_package: types.ModuleType):
    schemas_map = defaultdict(list)
    sa_models = importlib.import_module('.auto_generated.sa_models', base_package.__name__)
    base_module = importlib.import_module('.base', base_package.__name__)
    model_modules = [f[:-3] for f in os.listdir(sa_models.__path__[0]) if f.endswith('.py') and not f.startswith('_')]

    for module_name in model_modules:
        module = importlib.import_module(f'.{module_name}', sa_models.__name__)
        table_members = [
            member for member in inspect.getmembers(module)
            if member[1] is not None and isinstance(member[1], Table)
        ]
        model_members = [
            member for member in inspect.getmembers(module)
            if member[1] is not None and (
                    inspect.isclass(member[1]) and
                    issubclass(member[1], base_module.Base) and member[1] is not base_module.Base
            )
        ]
        for member in table_members:
            table = member[1]
            table_type_var = get_table_type_var_name(table.name, table.schema)
            var_name = class_name_for_table(table.name)
            schemas_map[module_name].append(
                f'{var_name}: {table_type_var!r} = _{table.schema}_models.{member[0]}')

        for member in model_members:
            table = member[1].__table__
            table_type_var = get_table_type_var_name(table.name, table.schema)
            var_name = class_name_for_table(table.name)
            schemas_map[module_name].append(
                f'{var_name}: {table_type_var!r} = _{table.schema}_models.{member[0]}.__table__')

    sa_tables_path = os.path.join(base_package.__path__[0], 'auto_generated', 'sa_tables')
    create_package(sa_tables_path)

    for schema_name in schemas_map:
        lines = [
            'import typing\n',

            (f'if typing.TYPE_CHECKING:\n'
             f'    from {base_package.__name__}.auto_generated import sqlalchemy_tables\n\n'),

            f'from {base_package.__name__}.auto_generated.sa_models import {schema_name} as _{schema_name}_models\n\n',

            '\n'.join(schemas_map[schema_name]),
        ]

        path = os.path.join(sa_tables_path, f'{schema_name}.py')

        with open(path, 'w') as f:
            f.write('\n'.join(lines))


def get_table_type_var_name(table_name: str, schema_name: str = ''):
    table_type_name_prefix = class_name_for_table(table_name, schema_name)
    return f'sqlalchemy_tables.{table_type_name_prefix}TableType'


class DeclarativeGenerator(sqlacodegen.generators.DeclarativeGenerator):

    def __init__(
            self,
            base_package: types.ModuleType,
            metadata: sqlacodegen.generators.MetaData,
            bind: sqlacodegen.generators.Connectable,
            options: sqlacodegen.generators.Set[str]
    ):
        super().__init__(metadata, bind, options)
        self.base_package = base_package

    def render_class_variables(self, model: sqlacodegen.models.ModelClass) -> str:
        variables = [f'__tablename__ = {model.table.name!r}']

        table_type_var = get_table_type_var_name(model.table.name, model.schema)
        variables.append(f"__table__: {table_type_var!r}")

        # Render constraints and indexes as __table_args__
        table_args = self.render_table_args(model.table)
        if table_args:
            variables.append(f'__table_args__ = {table_args}')

        return '\n'.join(variables)

    def render_module_variables(self, models: List[sqlacodegen.models.Model]) -> str:
        declarations = [
            'import typing\n',

            (f'if typing.TYPE_CHECKING:\n'
             f'    from {self.base_package.__name__}.auto_generated import sqlalchemy_tables\n'),

            f'from {self.base_package.__name__}.base import Base, metadata\n',
        ]
        return '\n'.join(declarations)

    def generate_separate_schemas(self) -> Dict[str, str]:
        rendered_schemas: Dict[str, str] = {}

        # Remove unwanted elements from the metadata
        for table in list(self.metadata.tables.values()):
            if self.should_ignore_table(table):
                self.metadata.remove(table)
                continue

            if 'noindexes' in self.options:
                table.indexes.clear()

            if 'noconstraints' in self.options:
                table.constraints.clear()

            if 'nocomments' in self.options:
                table.comment = None

            for column in table.columns:
                if 'nocomments' in self.options:
                    column.comment = None

        # Use information from column constraints to figure out the intended column types
        for table in self.metadata.tables.values():
            self.fix_column_types(table)

        # Generate the models
        all_models: List[sqlacodegen.models.Model] = self.generate_models()

        per_schema_models = defaultdict(list)
        for model in all_models:
            schema = model.schema
            per_schema_models[schema].append(model)

        for schema, schema_models in per_schema_models.items():
            sections: List[str] = []
            # Render collected imports
            groups = self.group_imports()
            imports = '\n\n'.join('\n'.join(line for line in group) for group in groups)
            if imports:
                sections.append(imports)

            # Render module level variables
            variables = self.render_module_variables(schema_models)
            if variables:
                sections.append(variables + '\n')

            # Render models
            rendered_models = self.render_models(schema_models)
            if rendered_models:
                sections.append(rendered_models)

            rendered_schemas[schema] = '\n\n'.join(sections) + '\n'

        return rendered_schemas

    def write_to_model_schema_files(self, dir_path: str):
        models_path = os.path.join(dir_path, '')
        create_package(models_path)

        rendered_schemas = self.generate_separate_schemas()
        for schema, rendered_content in rendered_schemas.items():
            file_path = os.path.join(models_path, f'{schema}.py')
            with open(file_path, 'w') as f:
                # Write the generated model code to the specified file or standard output
                f.write(rendered_content)

        file_path = os.path.join(models_path, f'_all_models.py')
        with open(file_path, 'w') as f:
            # Write the generated model code to the specified file or standard output
            for schema in rendered_schemas:
                f.write(f'from .{schema} import *  # noqa\n')


class TableStubGenerator(sqlacodegen.generators.TablesGenerator):

    def __init__(
            self,
            metadata: sqlacodegen.generators.MetaData,
            bind: sqlacodegen.generators.Connectable,
            options: sqlacodegen.generators.Set[str]
    ):
        super().__init__(metadata, bind, options)
        self.registry = {}

    def write_to_file(self, path: str):
        with open(path, 'w') as f:
            # Write the generated model code to the specified file or standard output
            f.write(self.generate())

    def render_table(self, table: Table) -> str:
        schema_name = table.schema or ""
        table_name = table.name

        table_args: List[str] = [f"__table__ = {table.name!r}", f"__schema__ = {schema_name!r}", ""]
        args = table_args.copy()
        for column in cast(ColumnCollection, table.columns):
            # Cast is required because of a bug in the SQLAlchemy stubs regarding Table.columns
            column_def = self.render_column(column, True)
            args.append(f'{column.name}: Column[Any] = {column_def}')

        defs = []

        def indented(attr_str):
            return (self.indentation if attr_str else '') + attr_str

        klass_name = class_name_for_table(table_name, schema_name=table.schema)
        columns_class_name = f'{klass_name}Columns'
        table_class_name = f'{klass_name}TableType'

        rendered_column_args = f'\n'.join((indented(arg) for arg in args))

        table_args.append(f'columns: {columns_class_name} = {columns_class_name}()')
        table_args.append(f'c: {columns_class_name} = columns')

        rendered_table_args = f'\n'.join((indented(arg) for arg in table_args))
        column_class = f'class {columns_class_name}:\n{rendered_column_args}\n'
        table_class = f'class {table_class_name}(Table):\n{rendered_table_args}\n'

        defs.append(column_class)
        defs.append(table_class)

        registry_key = '.'.join([n for n in (schema_name, table_name) if n])
        self.registry[registry_key] = table_class_name

        return '\n\n'.join(defs)

    def generate(self) -> str:
        sections: List[str] = []

        # Use information from column constraints to figure out the intended column types
        for table in self.metadata.tables.values():
            self.fix_column_types(table)

        # Generate the models
        models: List[sqlacodegen.models.Model] = self.generate_models()

        self.add_literal_import('typing', 'Any')

        # Render collected imports
        groups = self.group_imports()
        imports = '\n\n'.join('\n'.join(line for line in group) for group in groups)
        if imports:
            sections.append(imports)

        # Render module level variables
        variables = self.render_module_variables(models)
        if variables:
            sections.append(variables + '\n')

        # Render models
        rendered_models = self.render_models(models)
        if rendered_models:
            sections.append(rendered_models)

        # table_map_items = [f'{key!r}: {val}' for key, val in self.registry.items()]
        # rendered_table_map_items = f',\n{self.indentation}'.join(table_map_items)
        # sections.append(f'TABLE_MAP = {{\n{self.indentation}{rendered_table_map_items},\n}}')

        return '\n\n'.join(sections) + '\n'

    def render_module_variables(self, models: List[sqlacodegen.models.Model]) -> str:
        return ''

    def render_models(self, models: List[sqlacodegen.models.Model]) -> str:
        rendered = []
        for model in models:
            rendered_table = self.render_table(model.table)
            rendered.append(rendered_table)

        return '\n\n'.join(rendered)


def create_dir_recursive(path: str):
    os.makedirs(path, exist_ok=True)


def create_package(path: str):
    create_dir_recursive(path)
    init_file_path = os.path.join(path, '__init__.py')
    if not os.path.exists(init_file_path):
        with open(init_file_path, 'wb'):
            pass


def copy_files_recursively(src: str, dst: str, override_existing: bool = False):
    """
    Copy files recursively from source directory `a` to destination directory `b`.

    Args:
        src (str): The source directory path.
        dst (str): The destination directory path.
        override_existing (bool): Whether to override existing files at the destination path or not.
                                  Defaults to False.

    Raises:
        ValueError: If `a` or `b` is not a directory or if `a` and `b` form a cyclic loop through symlinks.
    """
    if not os.path.isdir(src):
        raise ValueError(f"{src} is not a directory")

    symlink_paths = set()

    for root, dirs, files in os.walk(src):
        # Check for symlinks in the source directory
        for symlink in dirs + files:
            symlink_path = os.path.join(root, symlink)
            if os.path.islink(symlink_path):
                real_path = os.path.realpath(symlink_path)
                if real_path in symlink_paths:
                    raise ValueError(f"Cyclic loop through symlinks detected between {symlink_path} and {real_path}")
                symlink_paths.add(symlink_path)

        # Create corresponding directories in destination directory
        for directory in dirs:
            source_path = os.path.join(root, directory)
            dest_path = source_path.replace(src, dst, 1)
            if not os.path.exists(dest_path):
                os.makedirs(dest_path)

        # Copy files to destination directory
        for file in files:
            source_path = os.path.join(root, file)
            dest_path = source_path.replace(src, dst, 1)
            if override_existing or not os.path.exists(dest_path):
                shutil.copy2(source_path, dest_path)


def create_base_packages(package_path: str):
    """

    :param package_path: Path to newly created package
    :return: package_object
    """

    global CUSTOM_TYPE_MAP

    copy_from_base_package_path = os.path.join(os.path.dirname(__file__), 'default_generated_package')
    copy_files_recursively(copy_from_base_package_path, package_path, False)

    package_name = os.path.basename(package_path)
    package = importlib.import_module(package_name)

    try:
        custom_types_module = importlib.import_module('.custom_types', package_name)
        CUSTOM_TYPE_MAP = custom_types_module.CUSTOM_TYPE_MAP
    except ModuleNotFoundError:
        pass

    return package


def main(generated_dir: str, dsn: str, schemas: str | list | tuple):
    generated_models_package_name = os.path.basename(generated_dir)
    assert re.match('^[a-zA-Z0-9_]+', generated_models_package_name), 'packagename must be a valid python package name'
    generated_package = create_base_packages(generated_dir)

    generate_sa_models(generated_package,  dsn, schemas)

    # schemas = [key for key in models.__dict__ if not key.startswith('_')]

    imports, pydantic_models = generate_pydantic_models(generated_package, _metadata.tables)
    write_pydantic_models(generated_package, imports=imports, pydantic_models=pydantic_models)


def write_pydantic_models(generated_package: types.ModuleType, imports, pydantic_models):
    path = os.path.join(generated_package.__path__[0], 'auto_generated', 'models', '__init__.py')
    s = '\n\n\n'.join(pydantic_models)
    with open(path, 'w') as f:
        f.write(textwrap.dedent(f"""
        import typing
        from typing import *  # noqa
        import datetime
        from decimal import Decimal  # noqa
        from uuid import UUID  # noqa
        from ipaddress import IPv4Address  # noqa
        from pydantic import IPvAnyInterface  # noqa

        from pdrdb.asyncpg_helpers.pydantic_dbmodel import DBModel
        from pdrdb.pydantic_ext import DateTimeTZRange  # noqa
        from pdrdb.pydantic_ext import optional  # noqa

        try:
            from {generated_package.__name__}.custom_types import *
        except ModuleNotFoundError:
            pass

        DB_DEFAULT = None  # noqa


        """))
        f.write('\n'.join(imports))
        f.write(s)
        f.write('\n')


def generate_pydantic_models(generated_package: types.ModuleType, tables: dict[str, Table]):
    classes = []
    imports = [
        *(f'from {generated_package.__name__}.auto_generated.sa_models import {schema_name} as _{schema_name}_models'
          for schema_name in {table.schema for table in tables.values()}),

        f'from {generated_package.__name__}.base import metadata',

        (f'\n\nif typing.TYPE_CHECKING:\n'
         f'    from {generated_package.__name__}.auto_generated import sqlalchemy_tables\n\n\n'),
    ]

    base_module = importlib.import_module('.base', generated_package.__name__)

    for full_name, table in tables.items():
        tbl, columns = get_pydantic_schema(table)

        class_name = class_name_for_table(table.name)
        schema_name = table.schema

        module = importlib.import_module(f'.auto_generated.sa_models.{schema_name}', generated_package.__name__)

        table_or_model = getattr(module, class_name, None)
        attr_is_model = (
                table_or_model is not None and
                inspect.isclass(table_or_model) and
                issubclass(table_or_model, base_module.Base)
        )

        field_lines = "\n".join(column_spec(col, columns[col]) for col in columns)

        __model__ = f"""_{schema_name}_models.{class_name}""" if attr_is_model else None

        tbl_type = get_table_type_var_name(full_name)
        column_class = f"""{tbl_type[:-len('TableType')]}Columns"""

        wrapped = textwrap.dedent(f"""
            @optional()
            class {table.name.title().replace('_', '')}(DBModel):
                __model__ = {__model__}
                __table__: {tbl_type!r} = metadata.tables['{full_name}']
                m: ClassVar[{__model__}] = {__model__}
                t: ClassVar[{tbl_type!r}]
                c: ClassVar[{column_class!r}]

                {textwrap.indent(field_lines, '                ').strip()}
            """)
        classes.append(wrapped.strip())

    return imports, classes


def run_cli():
    try:
        import generated_models
    except ModuleNotFoundError:
        generated_models = None

    if generated_models:
        default_dir = generated_models.__path__[0]
    else:
        default_dir = 'generated_models'

    import argparse
    parser = argparse.ArgumentParser(__name__)
    parser.add_argument(
        '--dir', '-d',
        dest='generated_path',
        default=default_dir,
        required=not default_dir,
        help='Directory where to generate models, default to "generated_models"',
    )
    parser.add_argument(
        '--dsn', '-db',
        dest='dsn',
        required=True,
        help='the asyncpg database dsn string for the db connection'
    )
    parser.add_argument(
        '--schemas', '-s',
        nargs='+',
        dest='schemas',
        required=False,
        default='public',
        help='the schemas to generate default to "public"'
    )

    args = parser.parse_args(sys.argv[1:])
    main(generated_dir=args.generated_path, dsn=args.dsn, schemas=args.schemas)


if __name__ == '__main__':
    run_cli()
