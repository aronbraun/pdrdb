import re
from typing import Union

from asyncpg.exceptions import PostgresError
from fastapi import HTTPException
from psycopg2 import DatabaseError

from pdrdb.db.sa.constraint_mappings import CONSTRAINT_NAME_MAPPING

key_column_spec = re.compile(r'\(([^)]+)\)=\(([^)]+)\)')
sample = 'Key (ip_inet)=(192.168.4.2/2) conflicts with existing key (ip_inet)=(192.168.187.233/21).'
key_column_spec_excl_val = re.compile(r'\(([^)]+)\)=\(([^)]+)\) conflicts with existing key \(([^)]+)\)=\(([^)]+)\)')


def message_from_db_exception(exception, data):
    sqlstate = exception.diag.sqlstate

    column_name = msg = None

    if sqlstate == '23503':  # foreign key
        match = key_column_spec.findall(exception.diag.message_detail)
        if match:
            column_name, val = match[0]
            msg = f'No {column_name} matching value {val}'

    elif sqlstate == '23505':
        match = key_column_spec.findall(exception.diag.message_detail)
        if match:
            column_name, val = match[0]
            msg = f'A record with {column_name} matching value {val} already exists'

    elif sqlstate == '22P02':  # invalid syntax for type:
        msg = exception.diag.message_primary
        return '__root__', msg

    if column_name:
        return column_loc_message(column_name, data, msg)
    return None, None


def message_from_asyncpg_db_exception(exception: PostgresError, data):
    sqlstate = exception.sqlstate  # noqa
    detail = exception.detail  # noqa
    message = exception.message  # noqa

    column_name = msg = None
    table_name = exception.table_name
    constraint_name = exception.constraint_name

    mapped_column_name = None
    if constraint_name in CONSTRAINT_NAME_MAPPING:
        mapped_column_name = CONSTRAINT_NAME_MAPPING[constraint_name].get('column_name')

    # https://www.postgresql.org/docs/current/errcodes-appendix.html#:~:text=Class%2023%20%E2%80%94%20Integrity%20Constraint%20Violation
    if (
        sqlstate == '23000'  # integrity_constraint_violation
        or sqlstate == '23001'  # restrict_violation
    ):
        # TODO the message should be customized to make sure we dont leak information we dont want to
        column_name = getattr(exception, 'column_name') or '__root__'
        msg = message

    elif sqlstate == '23502': # not_null_violation
        match = key_column_spec.findall(detail)
        if match:
            column_name, val = match[0]
            msg = f'"{column_name}" cannot be null'

    elif sqlstate == '23503':  # foreign key
        match = key_column_spec.findall(detail)
        if match:
            column_name, val = match[0]
            if 'is still referenced' in detail:
                msg = f'"{column_name}" with val "{val}" is in use with "{table_name}" and cannot be deleted'
            else:
                msg = f'No {column_name} matching value {val}'

    elif sqlstate == '23505':
        match = key_column_spec.findall(detail)
        if match:
            column_name, val = match[0]
            msg = f'A record with {mapped_column_name or column_name} matching value {val} already exists'
        else:
            column_name, val = detail.split('=', 1)
            msg = f'A record with {mapped_column_name or column_name} matching value {val}'

    elif sqlstate == '23514':  # Check constraint violation
        msg = f'Check constraint failed to validate ({exception.constraint_name})'
        return '__root__', msg

    elif sqlstate == '23P01':  # exclusion constraint
        match = key_column_spec_excl_val.findall(detail)
        if match:
            column_name, val, existing_col, existing_val = match[0]
            msg = f'A record with {column_name} with a value {existing_val} already includes  {val}'

    elif sqlstate == '22P02':  # invalid syntax for type:
        msg = message
        return '__root__', msg

    column_name = mapped_column_name or column_name
    if column_name:
        return column_loc_message(column_name, data, msg)
    return None, None


def column_loc_message(column_name: str, data: dict, msg: str):
    columns = column_name.split(',')
    fields = []
    for col in (c.strip() for c in columns):
        if col in data:
            fields.append(col)
        elif '.' in col:
            parts = col.split('.')
            cur_d = data
            for i, col_part in enumerate(parts):
                if cur_d and isinstance(cur_d, dict) and col_part in cur_d:
                    cur_d = cur_d[col_part]
                    fields.append(col_part)
            if len(parts) == len(fields):
                return fields, msg

    if len(fields) != 1:
        loc = '__root__'
    else:
        loc = fields[0]
    return loc, msg


def handle_db_exception(exception: Union[DatabaseError, PostgresError], data: dict):
    if isinstance(exception, PostgresError):
        field_name, msg = message_from_asyncpg_db_exception(exception, data)
    elif isinstance(exception, DatabaseError):
        field_name, msg = message_from_db_exception(exception, data)
    else:
        raise exception

    if msg:
        raise HTTPException(422, [{'loc': ('body', field_name), 'msg': msg, 'type': 'constraint_error'}])

    raise exception


class ConstraintError(Exception):
    def __init__(self, msg, field_name=None):
        self.msg = msg
        self.field_name = field_name

    def __str__(self):
        return self.msg
