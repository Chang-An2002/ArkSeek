"""Read-only projections of the existing stores; never instantiate MemoryStore."""
import hashlib
import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


class DataUnavailable(Exception):
    pass


def kind(identifier):
    return "group" if identifier.startswith("qq_group:") else ("private" if identifier.startswith("qq_c2c:") else "cli")


def short(identifier):
    return hashlib.sha256(identifier.encode()).hexdigest()[:6]


def age(timestamp):
    try:
        return (datetime.now(timezone.utc) - datetime.fromisoformat(timestamp)).total_seconds()
    except (ValueError, TypeError):
        return float("inf")


class Repository:
    def __init__(self, data_dir, log_dir=None):
        self.data_dir = Path(data_dir).resolve()
        self.log_dir = Path(log_dir).resolve() if log_dir else self.data_dir / "runtime_logs"

    @contextmanager
    def connect(self, name="memory.db"):
        path = self.data_dir / name
        if not path.is_file():
            raise DataUnavailable("尚未建立记忆数据库" if name == "memory.db" else "尚无结构化监测记录；使用新版机器人启动后开始记录")
        conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=0.3)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        try:
            yield conn
        finally:
            conn.close()

    @staticmethod
    def tables(conn):
        return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}

    @staticmethod
    def names(conn):
        if "display_name" not in Repository.tables(conn):
            return {}
        return {r["user_id"]: r["name"] for r in conn.execute("SELECT user_id,name FROM display_name")}

    @staticmethod
    def label(identifier, names):
        if kind(identifier) == "group" and identifier.count(":") == 1:
            return "群聊 · " + short(identifier)
        return names.get(identifier) or ("成员 · " if kind(identifier) == "group" else "用户 · ") + short(identifier)

    @staticmethod
    def speaker_expr(conn):
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(conversation_turns)")}
        return "speaker" if "speaker" in cols else "NULL AS speaker"

    def sessions(self, query="", channel="all", offset=0, limit=50):
        with self.connect() as conn:
            names = self.names(conn)
            where, args = "1=1", []
            if channel == "group":
                where += " AND user_id LIKE 'qq_group:%'"
            elif channel == "private":
                where += " AND user_id LIKE 'qq_c2c:%'"
            elif channel == "cli":
                where += " AND user_id NOT LIKE 'qq_group:%' AND user_id NOT LIKE 'qq_c2c:%'"
            if query:
                matching_names = [i for i, n in names.items() if query.casefold() in n.casefold()]
                # A group member's name should also locate the corresponding group.
                matching_names += [i.rsplit(":", 1)[0] for i in matching_names if kind(i) == "group" and i.count(":") >= 2]
                where += " AND (instr(lower(text),lower(?))>0 OR instr(lower(user_id),lower(?))>0"
                args += [query, query]
                if matching_names:
                    where += " OR user_id IN (" + ",".join("?" for _ in matching_names) + ")"
                    args += matching_names
                where += ")"
            total = conn.execute(f"SELECT COUNT(DISTINCT user_id) FROM conversation_turns WHERE {where}", args).fetchone()[0]
            # Select the matching conversations first, then count all their messages.
            rows = conn.execute(f"""SELECT user_id,COUNT(*) messages,MAX(created_at) last_at,
                  SUM(role='user') user_messages FROM conversation_turns
                  WHERE user_id IN (SELECT DISTINCT user_id FROM conversation_turns WHERE {where})
                  GROUP BY user_id ORDER BY last_at DESC,user_id LIMIT ? OFFSET ?""", args + [limit, offset]).fetchall()
            items = []
            speaker_col = self.speaker_expr(conn)
            for row in rows:
                item = dict(row)
                cid = item.pop("user_id")
                item.update(id=cid, name=self.label(cid, names), channel=kind(cid))
                participants = conn.execute(f"SELECT DISTINCT {speaker_col} FROM conversation_turns WHERE user_id=? AND role='user'", (cid,)).fetchall()
                ids = [r["speaker"] for r in participants if r["speaker"]]
                item["participants"] = [{"id": i, "name": self.label(i, names)} for i in ids]
                item["preview"] = conn.execute("SELECT text FROM conversation_turns WHERE user_id=? ORDER BY turn_index DESC,id DESC LIMIT 1", (cid,)).fetchone()[0][:160]
                items.append(item)
            return {"items": items, "total": total, "offset": offset, "limit": limit}

    def conversation(self, cid, query="", before=None, limit=50):
        with self.connect() as conn:
            names = self.names(conn)
            exists = conn.execute("SELECT 1 FROM conversation_turns WHERE user_id=? LIMIT 1", (cid,)).fetchone()
            if not exists:
                raise KeyError(cid)
            where, args = "user_id=?", [cid]
            if query:
                where += " AND instr(lower(text),lower(?))>0"
                args.append(query)
            if before is not None:
                where += " AND turn_index<?"
                args.append(before)
            total = conn.execute("SELECT COUNT(*) FROM conversation_turns WHERE user_id=?", (cid,)).fetchone()[0]
            speaker_col = self.speaker_expr(conn)
            rows = conn.execute(f"SELECT id,turn_index,role,text,created_at,{speaker_col} FROM conversation_turns WHERE {where} ORDER BY turn_index DESC,id DESC LIMIT ?", args + [limit + 1]).fetchall()
            has_more = len(rows) > limit
            rows = list(reversed(rows[:limit]))
            items = []
            for row in rows:
                item = dict(row)
                # New assistant rows store the reply target; old rows can be NULL.
                sid = item["speaker"] or (cid if kind(cid) != "group" else None)
                item["person_id"] = sid
                item["speaker_name"] = self.label(sid, names) if sid else "身份未记录"
                items.append(item)
            summary = None
            if "l0_summary" in self.tables(conn):
                row = conn.execute("SELECT * FROM l0_summary WHERE user_id=?", (cid,)).fetchone()
                summary = dict(row) if row else None
            return {"id": cid, "name": self.label(cid, names), "channel": kind(cid), "items": items,
                    "total": total, "has_more": has_more, "next_before": rows[0]["turn_index"] if rows else None,
                    "summary": summary}

    def users(self, query="", offset=0, limit=50):
        with self.connect() as conn:
            names = self.names(conn)
            # Preserve the project's identity namespaces; no cross-group person merge.
            sql = "SELECT user_id AS id FROM user_profile UNION SELECT user_id AS id FROM display_name UNION SELECT user_id AS id FROM conversation_turns WHERE user_id NOT LIKE 'qq_group:%'"
            if self.speaker_expr(conn) == "speaker":
                sql += " UNION SELECT speaker AS id FROM conversation_turns WHERE speaker IS NOT NULL"
            ids = [r["id"] for r in conn.execute(sql) if r["id"]]
            items = []
            for uid in ids:
                if query and query.casefold() not in (self.label(uid, names) + " " + uid).casefold():
                    continue
                count = conn.execute("SELECT COUNT(*) FROM user_profile WHERE user_id=?", (uid,)).fetchone()[0]
                items.append({"id": uid, "name": self.label(uid, names), "channel": kind(uid), "profile_count": count,
                              "scope": "本群身份" if kind(uid) == "group" else ("私聊身份" if kind(uid) == "private" else "本机对话身份")})
            items.sort(key=lambda x: (x["name"], x["id"]))
            return {"items": items[offset:offset + limit], "total": len(items), "offset": offset, "limit": limit}

    def user(self, uid):
        from profile_facts import supported_profile_fact
        with self.connect() as conn:
            names = self.names(conn)
            profiles = []
            for row in conn.execute("SELECT key,value,source_text,updated_at FROM user_profile WHERE user_id=? ORDER BY key", (uid,)):
                item = dict(row)
                item["effective"] = supported_profile_fact(item["key"], item["value"], item["source_text"] or "")
                profiles.append(item)
            if self.speaker_expr(conn) == "speaker":
                rows = conn.execute("SELECT DISTINCT user_id FROM conversation_turns WHERE speaker=? OR user_id=?", (uid, uid)).fetchall()
            else:
                rows = conn.execute("SELECT DISTINCT user_id FROM conversation_turns WHERE user_id=?", (uid,)).fetchall()
            if not profiles and not rows and uid not in names:
                raise KeyError(uid)
            return {"id": uid, "name": self.label(uid, names), "channel": kind(uid), "profiles": profiles,
                    "conversations": [{"id": r[0], "name": self.label(r[0], names)} for r in rows]}

    def monitor(self, status="all", offset=0, limit=50, hours=24):
        with self.connect("admin_monitor.db") as conn:
            runtimes = [dict(r) for r in conn.execute("SELECT * FROM runtime ORDER BY started_at DESC LIMIT 20")]
            live = {r["run_id"] for r in runtimes if r["state"] != "stopped" and age(r["heartbeat_at"]) < 10}
            for r in runtimes:
                r["heartbeat_live"] = r["run_id"] in live
            where = "julianday(received_at)>=julianday('now',?)"
            args = [f"-{hours} hours"]
            rows = [dict(r) for r in conn.execute(f"SELECT * FROM requests WHERE {where} ORDER BY received_at DESC LIMIT 10000", args)]
            for r in rows:
                r["recorded_status"] = r["status"]
                if r["status"] in ("queued", "processing", "sending") and r["run_id"] not in live:
                    r["status"] = "unknown"
                try:
                    r["spans"] = json.loads(r.pop("spans_json") or "[]")
                except (TypeError, ValueError):
                    r["spans"] = []
                r["conversation_name"] = self.label(r["conversation_id"], {})
                r["user_name"] = self.label(r["user_id"], {})
            summary = {"total": len(rows), "queued": 0, "processing": 0, "sending": 0, "sent": 0, "failed": 0, "unknown": 0}
            for r in rows:
                category = r["status"] if r["status"] in summary else "failed"
                summary[category] += 1
            durations = sorted(r["total_ms"] for r in rows if r["total_ms"] is not None)
            summary["median_ms"] = (durations[(len(durations) - 1) // 2] + durations[len(durations) // 2]) / 2 if durations else None
            selected = rows if status == "all" else [r for r in rows if r["status"] == status or (status == "failed" and r["status"] in ("failed", "send_failed", "processing_failed", "interrupted"))]
            return {"runtimes": runtimes, "summary": summary, "items": selected[offset:offset + limit],
                    "total": len(selected), "offset": offset, "limit": limit, "hours": hours,
                    "capped": len(rows) == 10000}

    def overview(self):
        result = {"memory_available": False, "monitor_available": False, "notice": []}
        try:
            with self.connect() as conn:
                result["messages"] = conn.execute("SELECT COUNT(*) FROM conversation_turns").fetchone()[0]
                result["sessions"] = conn.execute("SELECT COUNT(DISTINCT user_id) FROM conversation_turns").fetchone()[0]
                result["last_message_at"] = conn.execute("SELECT MAX(created_at) FROM conversation_turns").fetchone()[0]
                result["profile_facts"] = conn.execute("SELECT COUNT(*) FROM user_profile").fetchone()[0]
                result["memory_available"] = True
        except (DataUnavailable, sqlite3.Error):
            result["notice"].append("记忆数据库未建立或暂不可读")
        try:
            result["monitor"] = self.monitor(limit=8)
            result["monitor_available"] = True
        except (DataUnavailable, sqlite3.Error):
            result["notice"].append("结构化监测尚未接入；新版机器人下次启动后开始记录。历史请求的阶段和耗时未补造。")
        return result

    def request(self, request_id):
        with self.connect("admin_monitor.db") as conn:
            row = conn.execute("SELECT * FROM requests WHERE request_id=?", (request_id,)).fetchone()
            if row is None:
                raise KeyError(request_id)
            r = dict(row)
            runtime = conn.execute("SELECT * FROM runtime WHERE run_id=?", (r["run_id"],)).fetchone()
            live = runtime and runtime["state"] != "stopped" and age(runtime["heartbeat_at"]) < 10
            r["recorded_status"] = r["status"]
            if r["status"] in ("queued", "processing", "sending") and not live:
                r["status"] = "unknown"
            try:
                r["spans"] = json.loads(r.pop("spans_json") or "[]")
            except (TypeError, ValueError):
                r["spans"] = []
            r["conversation_name"] = self.label(r["conversation_id"], {})
            r["user_name"] = self.label(r["user_id"], {})
            return r

    def log_files(self):
        if not self.log_dir.is_dir():
            return []
        items = []
        for path in self.log_dir.glob("bot_*.log*"):
            if path.is_file() and path.resolve().parent == self.log_dir.resolve():
                stat = path.stat()
                items.append({"name": path.name, "size": stat.st_size, "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()})
        return sorted(items, key=lambda x: x["modified_at"], reverse=True)

    def log_tail(self, name, query="", limit=200):
        if name not in {r["name"] for r in self.log_files()}:
            raise KeyError(name)
        with (self.log_dir / name).open("rb") as file:
            file.seek(0, 2)
            length = file.tell()
            start = max(0, length - 256 * 1024)
            file.seek(start)
            if start:
                file.readline()  # Drop a partial UTF-8 line at the tail boundary.
            text = file.read().decode("utf-8", errors="replace")
        text = re.sub(r"(?i)((?:api[_-]?key|secret|token|authorization)[\"'\s:=]+)([^\s,;\"']+)", r"\1[已隐藏]", text)
        lines = text.splitlines()
        if query:
            lines = [line for line in lines if query.casefold() in line.casefold()]
        return {"name": name, "lines": lines[-limit:], "tail_bytes": min(length, 256 * 1024),
                "truncated": start > 0 or len(lines) > limit}
