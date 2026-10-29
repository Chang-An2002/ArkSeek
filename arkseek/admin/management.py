"""Opt-in local writes. All operations require an expiring, state-bound preview."""
import json
import secrets
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from profile_rules import SCHEMA as PROFILE_SCHEMA, rules_for
from profile_facts import supported_profile_fact
from runtime_settings import SETTINGS, RunLease, RuntimeBusy, read_settings, validate_settings
from .repository import Repository, DataUnavailable

BASE = Path(__file__).resolve().parent.parent
SCHEMA = """
CREATE TABLE IF NOT EXISTS documents(id TEXT PRIMARY KEY,title TEXT NOT NULL,body TEXT NOT NULL,
 active INTEGER NOT NULL,revision INTEGER NOT NULL,updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS settings(id INTEGER PRIMARY KEY CHECK(id=1),body TEXT NOT NULL,revision INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY,kind TEXT NOT NULL,target TEXT NOT NULL,
 before_json TEXT NOT NULL,after_json TEXT NOT NULL,reason TEXT NOT NULL,created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,kind TEXT NOT NULL,status TEXT NOT NULL,
 created_at TEXT NOT NULL,result_json TEXT,exit_code INTEGER);
"""


class Conflict(RuntimeError):
    pass


def stamp():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def packed(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class Management:
    def __init__(self, directory, enabled=False, demo=False):
        self.root = Path(directory).resolve()
        self.repo = Repository(self.root)
        self.enabled, self.demo = enabled, demo
        self.csrf = secrets.token_urlsafe(32)
        self.previews = {}
        self.lock = threading.RLock()
        self.children = {}  # Popen handles only; never adopt or kill a PID from disk.
        self.bot = None
        if enabled:
            self.root.mkdir(parents=True, exist_ok=True)
            with self.connect() as conn:
                conn.executescript(SCHEMA)
                if demo and not conn.execute("SELECT 1 FROM documents LIMIT 1").fetchone():
                    conn.execute("INSERT INTO documents VALUES (?,?,?,?,?,?)", ("demo-guide", "虚构演示资料", "这是虚构资料，用于演示项目副本管理、版本回退与构建任务，不包含真实科研或聊天内容。", 1, 1, stamp()))

    @contextmanager
    def connect(self):
        if not self.enabled:
            raise PermissionError("当前为只读模式；启用管理需使用 -Live -Manage")
        conn = sqlite3.connect(self.root / "admin_control.db", timeout=2)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def require_idle(self):
        if any(p.poll() is None for p in self.children.values()):
            raise Conflict("本后台管理的进程仍在运行，请先停止或等待任务结束")
        try:
            m = self.repo.monitor(hours=1, limit=1)
        except (DataUnavailable, sqlite3.Error):
            return
        if any(r["heartbeat_live"] for r in m["runtimes"]):
            raise Conflict("检测到近期机器人心跳；请确认进程已停止并等待心跳过期。本后台不会接管外部实例")

    def refresh_jobs(self):
        if not self.enabled:
            return
        with self.connect() as conn:
            for job_id, process in self.children.items():
                code = process.poll()
                if code is None:
                    continue
                path = self.root / "admin_jobs" / (job_id + ".json")
                try:
                    result = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
                except (ValueError, OSError):
                    result = {}
                status = "succeeded" if code == 0 else "failed"
                conn.execute("UPDATE jobs SET status=?,result_json=?,exit_code=? WHERE id=? AND status='running'", (status, packed(result), code, job_id))

    def status(self):
        if not self.enabled:
            return {"enabled": False, "demo": self.demo}
        with self.lock:
            self.refresh_jobs()
            with self.connect() as conn:
                docs = [dict(r) for r in conn.execute("SELECT id,title,active,revision,updated_at,length(body) AS characters FROM documents ORDER BY updated_at DESC")]
                jobs = [dict(r) for r in conn.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT 20")]
                events = [dict(r) for r in conn.execute("SELECT id,kind,target,reason,created_at FROM events ORDER BY rowid DESC LIMIT 30")]
            for j in jobs:
                owned = j["id"] in self.children
                if j["status"] == "running" and not owned:
                    j["status"] = "unknown"  # Service restart lost the handle, not proof of failure.
                j["owned"] = owned
                j["result"] = json.loads(j.pop("result_json") or "{}")
                if j["status"] == "running" and owned:
                    path = self.root / "admin_jobs" / (j["id"]+".json")
                    try:
                        if path.is_file():
                            j["result"] = json.loads(path.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        pass
            process = self.children.get(self.bot)
            local_files = [{"name": p.name, "bytes": p.stat().st_size} for p in sorted((self.root / "knowledge_base").glob("*.txt")) if p.is_file()]
            return {"enabled": True, "demo": self.demo, "csrf": self.csrf,
                    "settings": read_settings(self.root) or {k: v[0] for k, v in SETTINGS.items()},
                    "settings_saved": read_settings(self.root) is not None,
                    "bot": {"owned": process is not None, "alive": process is not None and process.poll() is None,
                            "exit_code": process.poll() if process else None, "job_id": self.bot},
                    "documents": docs, "jobs": jobs, "events": events,
                    "sources": {"local_files": local_files, "catalog_cached": (self.root / "kb_catalog.json").is_file(),
                                "external_configured": (self.root / "kb_sources.json").is_file()}}

    def document(self, doc_id):
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
        if not row:
            raise KeyError(doc_id)
        return dict(row)

    def event(self, event_id):
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
        if not row:
            raise KeyError(event_id)
        data = dict(row)
        data["before"] = json.loads(data.pop("before_json"))
        data["after"] = json.loads(data.pop("after_json"))
        return data

    def profile_state(self, uid, key, conn=None):
        def read(c):
            row = c.execute("SELECT * FROM user_profile WHERE user_id=? AND key=?", (uid, key)).fetchone()
            if row is None:
                raise KeyError(key)
            return {"stored": dict(row), "rule": rules_for(c, uid).get(key)}
        if conn is not None:
            return read(conn)
        with self.repo.connect() as c:
            return read(c)

    def profile_history(self, uid, key):
        with self.repo.connect() as conn:
            if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='admin_profile_history'").fetchone():
                return []
            return [dict(r) for r in conn.execute("SELECT * FROM admin_profile_history WHERE user_id=? AND key=? ORDER BY rowid DESC LIMIT 30", (uid, key))]

    def normalize(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("操作格式不正确")
        kind = payload.get("kind")
        allowed = {"profile", "profile_restore", "settings", "settings_restore", "document", "document_restore", "bot_start", "bot_stop", "index", "job_stop"}
        if kind not in allowed:
            raise ValueError("不支持的操作")
        p = {"kind": kind, "reason": str(payload.get("reason", "")).strip()}
        if not 2 <= len(p["reason"]) <= 500:
            raise ValueError("请填写2–500字的操作原因")
        if kind in ("profile", "profile_restore"):
            p.update(user_id=str(payload.get("user_id", "")), key=str(payload.get("key", "")))
            if not p["user_id"] or len(p["user_id"]) > 512 or not 1 <= len(p["key"]) <= 100:
                raise ValueError("身份或画像键不正确")
            if kind == "profile_restore":
                p["event_id"] = str(payload.get("event_id", ""))
            else:
                p["mode"] = payload.get("mode")
                p["value"] = str(payload.get("value", "")).strip()
                if p["mode"] not in ("override", "disabled", "automatic"):
                    raise ValueError("画像操作不正确")
                if p["mode"] == "override" and (not 1 <= len(p["value"]) <= 2000 or not supported_profile_fact(p["key"], p["value"])):
                    raise ValueError("更正值无效或被画像保护规则屏蔽")
        elif kind == "settings":
            p["values"] = validate_settings(payload.get("values"))
        elif kind == "document":
            p.update(id=str(payload.get("id", "")), title=str(payload.get("title", "")).strip(), body=str(payload.get("body", "")).strip(), active=payload.get("active", True))
            if not 1 <= len(p["title"]) <= 100 or not 1 <= len(p["body"]) <= 100000 or not isinstance(p["active"], bool):
                raise ValueError("标题1–100字、正文1–100000字，且启用状态需为布尔值")
        elif kind in ("document_restore", "settings_restore"):
            p["event_id"] = str(payload.get("event_id", ""))
        elif kind == "job_stop":
            p["id"] = str(payload.get("id", ""))
        return p

    def prepare(self, p):
        kind = p["kind"]
        if kind.startswith("profile"):
            current = self.profile_state(p["user_id"], p["key"])
            after = None if p.get("mode") == "automatic" else {"user_id": p["user_id"], "key": p["key"], "mode": p.get("mode"), "value": p.get("value") if p.get("mode") == "override" else None, "reason": p["reason"], "updated_at": "提交时记录"}
            if kind == "profile_restore":
                history = next((r for r in self.profile_history(p["user_id"], p["key"]) if r["id"] == p["event_id"]), None)
                if not history or current["rule"] != json.loads(history["after_json"]):
                    raise Conflict("画像已经有后续修改，不能覆盖；请重新预览更正")
                after = json.loads(history["before_json"])
            return current, after, "仅改变此身份的L1画像；不删除原话、历史会话或L2向量。停用不等于从历史中遗忘。维护需机器人/CLI停止，新启动后生效。更正或停用期间，自动抽取不能覆盖该键。"
        if kind == "settings":
            return read_settings(self.root), p["values"], "保存非密钥配置覆盖值，下次启动生效；不改.env，不关闭模型思考，不改变同群串行。"
        if kind == "settings_restore":
            event = self.event(p["event_id"])
            before = read_settings(self.root)
            if event["kind"] != "settings" or before != event["after"]:
                raise Conflict("配置已被后续修改，不能覆盖；请重新预览配置")
            return before, event["before"], "回退这一笔后台配置；没有旧覆盖值时恢复使用.env及程序默认值。机器人需停止，下次启动生效。"
        if kind in ("document", "document_restore"):
            if kind == "document":
                before = self.document(p["id"]) if p["id"] else None
                after = {"title": p["title"], "body": p["body"], "active": int(p["active"])}
            else:
                with self.connect() as conn:
                    e = conn.execute("SELECT * FROM events WHERE id=? AND kind='document'", (p["event_id"],)).fetchone()
                if not e:
                    raise KeyError(p["event_id"])
                before = self.document(e["target"])
                recorded = json.loads(e["after_json"])
                if before != recorded:
                    raise Conflict("该资料已有后续修改；不能覆盖新版本")
                old = json.loads(e["before_json"])
                after = {k: old[k] for k in ("title", "body", "active")} if old else {"title": before["title"], "body": before["body"], "active": 0}
            return before, after, "只修改后台导入的纯文本项目副本，保留历史版本。不会修改、移动或删除原始科研文件。下次构建/启动时进入或退出L3索引；正在运行的机器人不热更新。"
        status = self.status()
        before = {"bot": status["bot"], "jobs": [{"id": j["id"], "status": j["status"]} for j in status["jobs"] if j["status"] == "running"], "documents": status["documents"], "settings": status["settings"]}
        if kind == "bot_stop" and not status["bot"]["alive"]:
            raise Conflict("没有可停止的本后台机器人进程")
        if kind == "job_stop" and (p["id"] not in self.children or self.children[p["id"]].poll() is not None or p["id"] == self.bot):
            raise Conflict("仅可停止本后台仍持有句柄的索引任务")
        if kind in ("bot_start", "index"):
            self.require_idle()
        impact = {"bot_start": "启动本后台所属QQ进程，真实模式会连接QQ、调用已配置服务并写会话；演示模式只启动虚构工作进程。进程存活不代表QQ就绪。旧实例必须先关闭。", "bot_stop": "停止本后台持有句柄的机器人；会中断在途请求，未结束请求可能显示状态未确认；不停止任何外部PID。", "index": "独立进程重新加载L3资料并同步现有派生索引；真实模式会读取既有Zotero/JC与本地配置，可能加载/下载模型，耗时较长。不调用LLM、不清除聊天、不删除原始文件。构建失败不宣称完成，必要时重新构建。", "job_stop": "中断本后台索引任务；已写入的派生缓存可能需重新构建，不更改原始资料。"}[kind]
        return before, {"operation": kind}, impact

    def preview(self, payload):
        if not self.enabled:
            raise PermissionError("当前为只读模式")
        p = self.normalize(payload)
        with self.lock:
            before, after, impact = self.prepare(p)
            self.previews = {k: v for k, v in self.previews.items() if v["expires"] > time.monotonic()}
            if len(self.previews) >= 100:
                raise Conflict("预览过多，请稍后重试")
            token = secrets.token_urlsafe(32)
            self.previews[token] = {"payload": p, "before": before, "after": after, "expires": time.monotonic()+120}
            return {"token": token, "expires_in": 120, "kind": p["kind"], "before": before, "after": after, "impact": impact, "reason": p["reason"]}

    def commit_profile(self, preview):
        p, before = preview["payload"], preview["before"]
        path = (self.root / "memory.db")
        # Never create an absent memory database or instantiate MemoryStore.
        conn = sqlite3.connect(path.as_uri()+"?mode=rw", uri=True, timeout=2)
        conn.row_factory = sqlite3.Row
        try:
            conn.executescript(PROFILE_SCHEMA)
            conn.execute("BEGIN IMMEDIATE")
            if self.profile_state(p["user_id"], p["key"], conn) != before:
                raise Conflict("画像已变化，请重新预览")
            after = preview["after"]
            if after:
                after = {**after, "updated_at": stamp(), "reason": p["reason"]}
                conn.execute("INSERT OR REPLACE INTO admin_profile_rules VALUES (?,?,?,?,?,?)", tuple(after[k] for k in ("user_id", "key", "mode", "value", "reason", "updated_at")))
            else:
                conn.execute("DELETE FROM admin_profile_rules WHERE user_id=? AND key=?", (p["user_id"], p["key"]))
            event_id = uuid.uuid4().hex
            conn.execute("INSERT INTO admin_profile_history VALUES (?,?,?,?,?,?,?)", (event_id, p["user_id"], p["key"], packed(before["rule"]), packed(after), p["reason"], stamp()))
            conn.commit()
            return {"event_id": event_id, "message": "画像管理决定已保存，下次启动生效；原始记录未删除"}
        except Exception:
            conn.rollback(); raise
        finally:
            conn.close()

    def launch(self, kind, before, reason):
        job_id = uuid.uuid4().hex
        folder = self.root / "admin_jobs"
        folder.mkdir(exist_ok=True)
        command = [sys.executable, "-u", "-m", "admin.worker", kind, "--data-dir", str(self.root), "--job-id", job_id]
        if self.demo:
            command.append("--demo")
        # A custom live database must never silently start the fixed-path real bot.
        if not self.demo and self.root != (BASE / "data").resolve():
            raise Conflict("真实进程管理仅支持项目默认data目录；自定义目录仅支持浏览及数据维护")
        # Persist ownership intent before spawning; an audit failure cannot leave an unrecorded child.
        with self.connect() as conn:
            conn.execute("INSERT INTO jobs VALUES (?,?,?, ?,NULL,NULL)", (job_id, kind, "running", stamp()))
            conn.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?)", (uuid.uuid4().hex, "bot_start" if kind == "bot" else "index", job_id, packed(before), packed({"operation": kind, "state": "requested"}), reason, stamp()))
        try:
            with (folder / (job_id+".log")).open("wb") as log:
                process = subprocess.Popen(command, cwd=BASE, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError as exc:
            with self.connect() as conn:
                conn.execute("UPDATE jobs SET status='failed',result_json=? WHERE id=?", (packed({"error_kind":type(exc).__name__}),job_id))
            raise Conflict("进程未能启动，已保留失败记录") from exc
        self.children[job_id] = process
        if kind == "bot":
            self.bot = job_id
        return {"job_id": job_id, "message": "演示进程已启动" if self.demo else "任务进程已启动；尚未确认QQ就绪或索引完成"}

    def commit(self, token):
        with self.lock:
            preview = self.previews.pop(token, None)
            if not preview or preview["expires"] < time.monotonic():
                raise Conflict("预览已失效或已执行，请重新预览")
            p = preview["payload"]
            before, after, _ = self.prepare(p)
            if before != preview["before"] or after != preview["after"]:
                raise Conflict("数据或运行状态已变化，请重新预览")
            kind = p["kind"]
            if kind.startswith("profile"):
                self.require_idle()
                with RunLease(self.root):
                    return self.commit_profile(preview)
            if kind in ("bot_start", "index"):
                # Check lease; the child holds its own lock for its whole lifetime.
                with RunLease(self.root):
                    pass
                return self.launch("bot" if kind == "bot_start" else "index", before, p["reason"])
            if kind in ("bot_stop", "job_stop"):
                job_id = self.bot if kind == "bot_stop" else p["id"]
                process = self.children[job_id]
                self.record_event(kind, job_id, before, {"operation":kind,"state":"requested"}, p["reason"])
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    raise Conflict("停止信号已发送，但进程尚未退出；请刷新状态")
                with self.connect() as conn:
                    conn.execute("UPDATE jobs SET status='stopped',exit_code=? WHERE id=?", (process.returncode, job_id))
                return {"message": "所属进程已停止；在途任务可能未完成"}
            self.require_idle()
            with RunLease(self.root), self.connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                event_id = uuid.uuid4().hex
                if kind in ("settings", "settings_restore"):
                    if read_settings(self.root) != before:
                        raise Conflict("配置已被其他后台修改，请重新预览")
                    if after is None:
                        conn.execute("DELETE FROM settings WHERE id=1")
                    else:
                        conn.execute("INSERT INTO settings VALUES (1,?,1) ON CONFLICT(id) DO UPDATE SET body=excluded.body,revision=revision+1", (packed(after),))
                    kind = "settings"
                    target = "runtime"
                else:
                    target = before["id"] if before else uuid.uuid4().hex
                    current = conn.execute("SELECT * FROM documents WHERE id=?", (target,)).fetchone()
                    if (dict(current) if current else None) != before:
                        raise Conflict("资料版本已变化，请重新预览")
                    after = {"id": target, **after, "revision": before["revision"]+1 if before else 1, "updated_at": stamp()}
                    conn.execute("INSERT OR REPLACE INTO documents VALUES (?,?,?,?,?,?)", tuple(after[k] for k in ("id", "title", "body", "active", "revision", "updated_at")))
                    kind = "document"
                conn.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?)", (event_id, kind, target, packed(before), packed(after), p["reason"], stamp()))
                return {"event_id": event_id, "message": "更改已保存；下次启动/构建生效"}

    def record_event(self, kind, target, before, after, reason):
        with self.connect() as conn:
            conn.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?)", (uuid.uuid4().hex, kind, target, packed(before), packed(after), reason, stamp()))

    def close(self):
        # Do not kill a real bot merely because the admin browser/server closes.
        for process in self.children.values():
            if self.demo and process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
