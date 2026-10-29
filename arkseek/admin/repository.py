"""Read-only projections of the existing stores; never instantiate MemoryStore."""
import hashlib
import json
import re
import sqlite3
import math
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path


class DataUnavailable(Exception):
    pass


CHINA = timezone(timedelta(hours=8))
ACTIVE = {"queued", "processing", "sending"}
FAILED = {"failed", "send_failed", "processing_failed", "interrupted"}


def timestamp(value):
    try:
        parsed = datetime.fromisoformat(value)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def percentile(values, fraction):
    values = sorted(v for v in values if isinstance(v, (int, float)) and math.isfinite(v) and v >= 0)
    if not values:
        return None
    position = (len(values) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return round(values[lo] + (values[hi] - values[lo]) * (position - lo), 1)


def time_window(range_key="24h", day=None, hour=None, now=None):
    now = now or datetime.now(timezone.utc)
    local = now.astimezone(CHINA)
    midnight = local.replace(hour=0, minute=0, second=0, microsecond=0)
    if range_key == "today":
        start = midnight
    elif range_key == "7d":
        start = midnight - timedelta(days=6)
    elif range_key in ("1h", "24h", "168h"):
        start = now - timedelta(hours=int(range_key[:-1]))
    else:
        raise ValueError("时间范围不正确")
    end = now
    if hour is not None and (day is None or not 0 <= hour <= 23):
        raise ValueError("小时筛选需要有效日期")
    if day is not None:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
            raise ValueError("日期格式不正确")
        selected = datetime.fromisoformat(day).replace(tzinfo=CHINA)
        if hour is not None:
            selected += timedelta(hours=hour)
        start = max(start, selected)
        end = min(end, selected + (timedelta(hours=1) if hour is not None else timedelta(days=1)))
    return start.astimezone(timezone.utc), end.astimezone(timezone.utc)


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

    def users(self, query="", offset=0, limit=50, profile_state="all"):
        from profile_facts import supported_profile_fact
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
                facts = conn.execute("SELECT key,value,source_text FROM user_profile WHERE user_id=?", (uid,)).fetchall()
                from profile_rules import rules_for, effective_fact
                rules = rules_for(conn, uid)
                blocked = sum(not effective_fact(f, rules.get(f["key"]))["effective"] for f in facts)
                if profile_state == "blocked" and not blocked:
                    continue
                items.append({"id": uid, "name": self.label(uid, names), "channel": kind(uid), "profile_count": len(facts), "blocked_count": blocked,
                              "scope": "本群身份" if kind(uid) == "group" else ("私聊身份" if kind(uid) == "private" else "本机对话身份")})
            items.sort(key=lambda x: (x["name"], x["id"]))
            return {"items": items[offset:offset + limit], "total": len(items), "offset": offset, "limit": limit}

    def user(self, uid):
        from profile_facts import supported_profile_fact
        with self.connect() as conn:
            names = self.names(conn)
            profiles = []
            from profile_rules import rules_for, effective_fact
            rules = rules_for(conn, uid)
            for row in conn.execute("SELECT key,value,source_text,updated_at FROM user_profile WHERE user_id=? ORDER BY key", (uid,)):
                item = effective_fact(row, rules.get(row["key"]))
                profiles.append(item)
            if self.speaker_expr(conn) == "speaker":
                rows = conn.execute("SELECT DISTINCT user_id FROM conversation_turns WHERE speaker=? OR user_id=?", (uid, uid)).fetchall()
            else:
                rows = conn.execute("SELECT DISTINCT user_id FROM conversation_turns WHERE user_id=?", (uid,)).fetchall()
            if not profiles and not rows and uid not in names:
                raise KeyError(uid)
            return {"id": uid, "name": self.label(uid, names), "channel": kind(uid), "profiles": profiles,
                    "conversations": [{"id": r[0], "name": self.label(r[0], names)} for r in rows]}

    @staticmethod
    def decorate_request(row, live, names):
        r = dict(row)
        r["recorded_status"] = r["status"]
        if (r["status"] in ACTIVE and r["run_id"] not in live) or r["status"] not in ACTIVE | FAILED | {"sent"}:
            r["status"] = "unknown"
        try:
            spans = json.loads(r.pop("spans_json") or "[]")
            r["spans"] = spans if isinstance(spans, list) else []
        except (TypeError, ValueError):
            r["spans"] = []
        r["conversation_name"] = Repository.label(r["conversation_id"], names)
        r["user_name"] = Repository.label(r["user_id"], names)
        return r

    def monitor(self, status="all", offset=0, limit=50, hours=24, range_key=None,
                day=None, hour=None, conversation_id=None, sort="recent", duration_band=None):
        start, end = time_window(range_key or f"{hours}h", day, hour)
        names = {}
        try:
            with self.connect() as memory:
                names = self.names(memory)
        except (DataUnavailable, sqlite3.Error):
            pass
        with self.connect("admin_monitor.db") as conn:
            runtimes = [dict(r) for r in conn.execute("SELECT * FROM runtime ORDER BY started_at DESC")]
            live = {r["run_id"] for r in runtimes if r["state"] != "stopped" and age(r["heartbeat_at"]) < 10}
            for r in runtimes:
                r["heartbeat_live"] = r["run_id"] in live
            where = "julianday(received_at)>=julianday(?) AND julianday(received_at)<julianday(?)"
            args = [start.isoformat(), end.isoformat()]
            if conversation_id:
                where += " AND conversation_id=?"
                args.append(conversation_id)
            if duration_band:
                bands = {"under10": (0, 10000), "10to30": (10000, 30000), "30to60": (30000, 60000), "over60": (60000, None)}
                if duration_band not in bands:
                    raise ValueError("耗时范围不正确")
                lower, upper = bands[duration_band]
                where += " AND finished_at IS NOT NULL AND total_ms>=?"
                args.append(lower)
                if upper is not None:
                    where += " AND total_ms<?"
                    args.append(upper)
            matched = conn.execute(f"SELECT COUNT(*) FROM requests WHERE {where}", args).fetchone()[0]
            rows = [self.decorate_request(r, live, names) for r in conn.execute(f"SELECT * FROM requests WHERE {where} ORDER BY julianday(received_at) DESC,request_id DESC LIMIT 10000", args)]
            earliest = conn.execute("SELECT received_at FROM requests WHERE julianday(received_at) IS NOT NULL ORDER BY julianday(received_at) LIMIT 1").fetchone()
            recorded_from = earliest[0] if earliest else None
            summary = {"total": len(rows), "queued": 0, "processing": 0, "sending": 0, "sent": 0, "failed": 0, "unknown": 0}
            for r in rows:
                category = "failed" if r["status"] in FAILED else r["status"]
                summary[category] += 1
            summary["median_ms"] = percentile([r["total_ms"] for r in rows if r["finished_at"]], .5)
            selected = rows if status == "all" else [r for r in rows if r["status"] == status or (status == "failed" and r["status"] in FAILED)]
            if sort not in ("recent", "slowest", "queue"):
                raise ValueError("排序不正确")
            if sort != "recent":
                field = "total_ms" if sort == "slowest" else "queue_ms"
                selected.sort(key=lambda r: r[field] if isinstance(r[field], (int, float)) and r[field] >= 0 else -1, reverse=True)
            return {"runtimes": [r for i, r in enumerate(runtimes) if i < 20 or r["heartbeat_live"]], "summary": summary, "items": selected[offset:offset + limit],
                    "total": len(selected), "offset": offset, "limit": limit, "hours": hours,
                    "capped": matched > 10000, "matched": matched, "recorded_from": recorded_from,
                    "window": {"start": start.isoformat(), "end": end.isoformat(), "range": range_key or f"{hours}h"}}

    def analytics(self, range_key="24h", conversation_id=None):
        data = self.monitor(limit=10000, range_key=range_key, conversation_id=conversation_id)
        rows = data.pop("items")
        start, end = timestamp(data["window"]["start"]), timestamp(data["window"]["end"])
        hourly = range_key != "7d"
        cursor = start.astimezone(CHINA).replace(minute=0, second=0, microsecond=0)
        if not hourly:
            cursor = cursor.replace(hour=0)
        buckets = {}
        step = timedelta(hours=1) if hourly else timedelta(days=1)
        while cursor < end:
            day, hour = cursor.strftime("%Y-%m-%d"), cursor.hour if hourly else None
            buckets[(day, hour)] = {"day": day, "hour": hour, "label": cursor.strftime("%m/%d %H时" if hourly else "%m/%d"),
                                    "count": 0, "sent": 0, "failed": 0, "totals": [], "queues": []}
            cursor += step
        groups = {}
        counts = {key: 0 for key in ("sent", "processing_failed", "send_failed", "failed", "interrupted", "queued", "processing", "sending", "unknown")}
        bands = [{"id": key, "label": label, "count": 0} for key, label in (("under10", "小于10秒"), ("10to30", "10–30秒"), ("30to60", "30–60秒"), ("over60", "60秒及以上"))]
        finished_values, queues, active_users = [], [], set()
        for r in rows:
            counts[r["status"]] += 1
            if r["user_id"]:
                active_users.add(r["user_id"])
            total, queue_ms = r["total_ms"], r["queue_ms"]
            valid_total = isinstance(total, (int, float)) and math.isfinite(total) and total >= 0 and r["finished_at"] is not None
            valid_queue = isinstance(queue_ms, (int, float)) and math.isfinite(queue_ms) and queue_ms >= 0
            if valid_total:
                finished_values.append(total)
                bands[0 if total < 10000 else 1 if total < 30000 else 2 if total < 60000 else 3]["count"] += 1
            if valid_queue:
                queues.append(queue_ms)
            moment = timestamp(r["received_at"]).astimezone(CHINA)
            bucket = buckets.get((moment.strftime("%Y-%m-%d"), moment.hour if hourly else None))
            if bucket is not None:
                bucket["count"] += 1
                bucket["sent"] += r["status"] == "sent"
                bucket["failed"] += r["status"] in FAILED
                if valid_total:
                    bucket["totals"].append(total)
                if valid_queue:
                    bucket["queues"].append(queue_ms)
            group = groups.setdefault(r["conversation_id"], {"id": r["conversation_id"], "name": r["conversation_name"], "channel": kind(r["conversation_id"]), "count": 0, "failed": 0, "queues": [], "totals": []})
            group["count"] += 1
            group["failed"] += r["status"] in FAILED
            if valid_queue:
                group["queues"].append(queue_ms)
            if valid_total:
                group["totals"].append(total)
        for bucket in buckets.values():
            bucket["median_ms"] = percentile(bucket.pop("totals"), .5)
            bucket["queue_median_ms"] = percentile(bucket.pop("queues"), .5)
        for group in groups.values():
            group["queue_median_ms"] = percentile(group.pop("queues"), .5)
            group["median_ms"] = percentile(group.pop("totals"), .5)
        summary = data["summary"]
        summary.update(users=len(active_users), conversations=len(groups), completed_samples=len(finished_values),
                       missing_duration=len(rows) - len(finished_values), queue_samples=len(queues),
                       p90_ms=percentile(finished_values, .9), queue_median_ms=percentile(queues, .5),
                       sent_rate=round(counts["sent"] / len(rows) * 100, 1) if rows else None)
        data.update(series=list(buckets.values()), outcomes=counts, distribution=bands,
                    groups=sorted(groups.values(), key=lambda g: (-g["count"], g["id"])),
                    recent=rows[:6], slowest=sorted([r for r in rows if r["total_ms"] is not None], key=lambda r: r["total_ms"], reverse=True)[:5])
        return data

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
            runtime = conn.execute("SELECT * FROM runtime WHERE run_id=?", (row["run_id"],)).fetchone()
            live = {row["run_id"]} if runtime and runtime["state"] != "stopped" and age(runtime["heartbeat_at"]) < 10 else set()
            return self.decorate_request(row, live, {})

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
