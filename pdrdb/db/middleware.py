from starlette.types import Receive
from starlette.types import Scope
from starlette.types import Send

from pdrdb.db.implementation import DBContext, DB


class DBContextMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope: 'Scope', receive: 'Receive', send: 'Send') -> None:
        if scope["type"] != "http":  # pragma: no cover
            await self.app(scope, receive, send)
            return

        token = DBContext.set(DB())
        await self.app(scope, receive, send)
        DBContext.reset(token)
        return
