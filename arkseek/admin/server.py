import asyncio
import sqlite3
import secrets
from pathlib import Path

from aiohttp import web
from .repository import Repository, DataUnavailable
from .management import Management, Conflict
from runtime_settings import RuntimeBusy

STATIC = Path(__file__).resolve().parent / "web"


def create_app(data_dir, demo=False, manage=False):
    repo = Repository(data_dir)
    manager = Management(data_dir, enabled=manage, demo=demo)

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
        except PermissionError as exc:
            response = web.json_response({"error": str(exc)}, status=403)
        except (Conflict, RuntimeBusy) as exc:
            response = web.json_response({"error": str(exc)}, status=409)
        except (ValueError, TypeError) as exc:
            response = web.json_response({"error": str(exc) if request.path.startswith('/api/manage/') else "查询参数不正确"}, status=400)
        except sqlite3.Error:
            response = web.json_response({"error": "数据库暂不可读，请稍后刷新"}, status=503)
        response.headers.update({"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
                                 "X-Frame-Options": "DENY", "Referrer-Policy": "no-referrer",
                                 "Content-Security-Policy": "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"})
        return response

    app = web.Application(middlewares=[boundary], client_max_size=512*1024)

    async def management(request):
        action = request.match_info["action"]
        if request.method == "GET":
            if action == "status":
                return web.json_response(await asyncio.to_thread(manager.status))
            if not manager.enabled:
                raise PermissionError("当前为只读模式")
            if action == "document":
                data = await asyncio.to_thread(manager.document, request.query["id"])
            elif action == "event":
                data = await asyncio.to_thread(manager.event, request.query["id"])
            elif action == "profile-history":
                data = {"items": await asyncio.to_thread(manager.profile_history, request.query["user_id"], request.query["key"])}
            else:
                raise web.HTTPNotFound()
            return web.json_response(data)
        if not manager.enabled:
            raise PermissionError("当前为只读模式")
        if request.headers.get("Origin") != f"{request.scheme}://{request.host}" or not secrets.compare_digest(request.headers.get("X-Admin-Token", ""), manager.csrf):
            raise PermissionError("管理操作来源或令牌无效，请刷新页面")
        if request.content_type != "application/json":
            raise ValueError("管理请求必须使用JSON")
        body = await request.json()
        if action == "preview":
            data = await asyncio.to_thread(manager.preview, body)
        elif action == "commit":
            data = await asyncio.to_thread(manager.commit, str(body.get("token", "")))
        else:
            raise web.HTTPNotFound()
        return web.json_response(data)

    async def cleanup(app):
        await asyncio.to_thread(manager.close)
    app.on_cleanup.append(cleanup)
    app.router.add_get("/api/manage/{action}", management)
    app.router.add_post("/api/manage/{action}", management)

    def pagination(request):
        offset = int(request.query.get("offset", 0))
        limit = int(request.query.get("limit", 50))
        if offset < 0 or not 1 <= limit <= 100:
            raise ValueError()
        return {"offset": offset, "limit": limit}

    def monitor_filters(request):
        range_key = request.query.get("range")
        if range_key is not None and range_key not in ("today", "24h", "7d", "1h", "168h"):
            raise ValueError()
        cid = request.query.get("conversation_id") or None
        if cid and len(cid) > 512:
            raise ValueError()
        return {"range_key": range_key, "day": request.query.get("day") or None,
                "hour": int(request.query["hour"]) if "hour" in request.query else None,
                "conversation_id": cid}

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
            profile_state = request.query.get("profile_state", "all")
            if profile_state not in ("all", "blocked"):
                raise ValueError()
            data = await asyncio.to_thread(repo.users, query=q, profile_state=profile_state, **pagination(request))
        elif resource == "user":
            data = await asyncio.to_thread(repo.user, request.query["id"])
        elif resource == "monitor":
            status = request.query.get("status", "all")
            hours = int(request.query.get("hours", 24))
            if status not in ("all", "queued", "processing", "sending", "sent", "failed", "send_failed", "processing_failed", "interrupted", "unknown") or hours not in (1, 24, 168):
                raise ValueError()
            data = await asyncio.to_thread(repo.monitor, status=status, hours=hours,
                                          sort=request.query.get("sort", "recent"), duration_band=request.query.get("duration_band") or None,
                                          **monitor_filters(request), **pagination(request))
        elif resource == "analytics":
            filters = monitor_filters(request)
            # Analytics charts represent the whole selected range; bucket and
            # duration drill-downs are applied by /monitor, not silently mixed.
            if filters["day"] or filters["hour"] is not None:
                raise ValueError()
            data = await asyncio.to_thread(repo.analytics, range_key=filters["range_key"] or "24h", conversation_id=filters["conversation_id"])
            data.update(mode="demo" if demo else "live", read_only=True)
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
        if name not in ("index.html", "app.css", "app.js", "charts.js", "management.js"):
            raise web.HTTPNotFound()
        return web.FileResponse(STATIC / name)

    app.router.add_get("/", file)
    app.router.add_get("/api/{resource}", api)
    app.router.add_get("/{name}", file)
    return app
