from uuid import UUID

import sqlalchemy.orm
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.dialects.postgresql.asyncpg import (
    AsyncAdapt_asyncpg_dbapi,
    AsyncpgJSONB as _AsyncpgJSONB,
    AsyncpgJSON as _AsyncpgJSON,
)


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
