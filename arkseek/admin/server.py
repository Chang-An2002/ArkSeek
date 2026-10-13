import asyncio
import sqlite3
from pathlib import Path

from aiohttp import web
from .repository import Repository, DataUnavailable

STATIC = Path(__file__).resolve().parent / "web"


def create_app(data_dir, demo=False):
    repo = Repository(data_dir)

    @web.middleware
    async def boundary(request, handler):
        # Loopback binding plus Host and Origin checks prevent DNS rebinding and
        # browser-origin access to the local private database. No CORS is enabled.
        host = request.url.host
        if host not in ("127.0.0.1", "localhost"):
            raise web.HTTPForbidden(text="仅允许本机访问")
        origin = request.headers.get("Origin")
        if origin and origin != f"{request.scheme}://{request.host}":
            raise web.HTTPForbidden(text="不允许跨站访问")
        if request.headers.get("Sec-Fetch-Site") == "cross-site":
            raise web.HTTPForbidden(text="不允许跨站访问")
        try:
            response = await handler(request)
        except DataUnavailable as exc:
            response = web.json_response({"error": str(exc)}, status=503)
        except KeyError:
            response = web.json_response({"error": "记录不存在"}, status=404)
        except (ValueError, TypeError):
            response = web.json_response({"error": "查询参数不正确"}, status=400)
        except sqlite3.Error:
            response = web.json_response({"error": "数据库暂不可读，请稍后刷新"}, status=503)
        response.headers.update({"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
                                 "X-Frame-Options": "DENY", "Referrer-Policy": "no-referrer",
                                 "Content-Security-Policy": "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"})
        return response

    app = web.Application(middlewares=[boundary])

    def pagination(request):
        offset = int(request.query.get("offset", 0))
        limit = int(request.query.get("limit", 50))
        if offset < 0 or not 1 <= limit <= 100:
            raise ValueError()
        return {"offset": offset, "limit": limit}

    async def api(request):
        resource = request.match_info["resource"]
        q = request.query.get("q", "")
        if len(q) > 200:
            raise ValueError()
        if resource == "overview":
            data = await asyncio.to_thread(repo.overview)
            data.update(mode="demo" if demo else "live", read_only=True)
        elif resource == "sessions":
            channel = request.query.get("channel", "all")
            if channel not in ("all", "group", "private", "cli"):
                raise ValueError()
            data = await asyncio.to_thread(repo.sessions, query=q, channel=channel, **pagination(request))
        elif resource == "conversation":
            before = request.query.get("before")
            data = await asyncio.to_thread(repo.conversation, request.query["id"], query=q,
                                          before=int(before) if before is not None else None,
                                          limit=pagination(request)["limit"])
        elif resource == "users":
            data = await asyncio.to_thread(repo.users, query=q, **pagination(request))
        elif resource == "user":
            data = await asyncio.to_thread(repo.user, request.query["id"])
        elif resource == "monitor":
            status = request.query.get("status", "all")
            hours = int(request.query.get("hours", 24))
            if status not in ("all", "queued", "processing", "sending", "sent", "failed", "unknown") or hours not in (1, 24, 168):
                raise ValueError()
            data = await asyncio.to_thread(repo.monitor, status=status, hours=hours, **pagination(request))
        elif resource == "logs":
            data = {"items": await asyncio.to_thread(repo.log_files)}
        elif resource == "request":
            data = await asyncio.to_thread(repo.request, request.query["id"])
        elif resource == "log":
            data = await asyncio.to_thread(repo.log_tail, request.query["name"], query=q,
                                          limit=min(pagination(request)["limit"] * 5, 500))
        else:
            raise web.HTTPNotFound()
        return web.json_response(data)

    async def file(request):
        name = request.match_info.get("name", "index.html")
        if name not in ("index.html", "app.css", "app.js"):
            raise web.HTTPNotFound()
        return web.FileResponse(STATIC / name)

    app.router.add_get("/", file)
    app.router.add_get("/api/{resource}", api)
    app.router.add_get("/{name}", file)
    return app
