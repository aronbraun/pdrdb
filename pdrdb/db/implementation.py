import contextvars

from pdrdb.db.db_integrity_helper import DBIntegrityAsyncPG as DBIntegrity  # noqa
from pdrdb.db.sa.asyncpg import MY_DB as DB


DBContext = contextvars.ContextVar('DBContext')


def get_db_context_object() -> DB:
    try:
        return DBContext.get()
    except LookupError:
        val = DB()
        DBContext.set(val)
        return val


get_db = get_db_context_object
