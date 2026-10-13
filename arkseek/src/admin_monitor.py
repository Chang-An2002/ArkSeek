"""Non-blocking, local admin telemetry. No messages, prompts, secrets or nicknames.

Importing this module does not create files. configure() is called by qq_bot.main.
Telemetry is best effort: failures must never prevent a QQ answer.
"""
import atexit
import json
import os
import queue
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


SCHEMA = """
CREATE TABLE IF NOT EXISTS runtime (
 run_id TEXT PRIMARY KEY, pid INTEGER, started_at TEXT, heartbeat_at TEXT,
 state TEXT, dropped INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS requests (
 request_id TEXT PRIMARY KEY, run_id TEXT, conversation_id TEXT, user_id TEXT,
 received_at TEXT, started_at TEXT, finished_at TEXT, status TEXT,
 queue_ms REAL, total_ms REAL, error_kind TEXT, spans_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_requests_time ON requests(received_at);
"""


class Monitor:
    def __init__(self, path):
        self.path = Path(path)
        self.run_id = uuid.uuid4().hex
        self.started_at = utc_now()
        self.state = "starting"
        self.dropped = 0
        self.queue = queue.Queue(maxsize=2048)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._write, name="admin-monitor", daemon=True)
        self.thread.start()

    def emit(self, kind, data):
        try:
            self.queue.put_nowait((kind, data))
        except queue.Full:
            self.dropped += 1

    def begin(self, conversation_id, user_id):
        request_id = uuid.uuid4().hex
        self.emit("begin", (request_id, self.run_id, conversation_id, user_id, utc_now()))
        return request_id

    def phase(self, request_id, status, queue_ms=None):
        self.emit("phase", (status, utc_now(), queue_ms, request_id))

    def finish(self, request_id, status, total_ms, error_kind=None, spans=None):
        # Span detail can contain original query text. Persist timing and name only.
        safe_spans = [{"start_ms": round(s[0] * 1000, 1), "duration_ms": round(s[1] * 1000, 1),
                       "name": s[2], "task": s[4]} for s in (spans or [])]
        self.emit("finish", (utc_now(), status, round(total_ms, 1), error_kind,
                             json.dumps(safe_spans, ensure_ascii=False), request_id))

    def close(self):
        self.state = "stopped"
        self.stop.set()
        self.thread.join(timeout=2)

    def _write(self):
        conn = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.path, timeout=1)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)
            beat_at = 0
            while not self.stop.is_set() or not self.queue.empty():
                try:
                    kind, data = self.queue.get(timeout=0.25)
                    try:
                        if kind == "begin":
                            conn.execute("INSERT INTO requests (request_id,run_id,conversation_id,user_id,received_at,status) VALUES (?,?,?,?,?,'queued')", data)
                        elif kind == "phase":
                            conn.execute("UPDATE requests SET status=?,started_at=COALESCE(started_at,?),queue_ms=COALESCE(queue_ms,?) WHERE request_id=?", data)
                        elif kind == "finish":
                            conn.execute("UPDATE requests SET finished_at=?,status=?,total_ms=?,error_kind=?,spans_json=? WHERE request_id=?", data)
                        conn.commit()
                    finally:
                        self.queue.task_done()
                except queue.Empty:
                    pass
                if time.monotonic() - beat_at >= 2:
                    conn.execute("INSERT INTO runtime VALUES (?,?,?,?,?,?) ON CONFLICT(run_id) DO UPDATE SET heartbeat_at=excluded.heartbeat_at,state=excluded.state,dropped=excluded.dropped",
                                 (self.run_id, os.getpid(), self.started_at, utc_now(), self.state, self.dropped))
                    conn.commit()
                    beat_at = time.monotonic()
            conn.execute("UPDATE runtime SET heartbeat_at=?,state='stopped',dropped=? WHERE run_id=?", (utc_now(), self.dropped, self.run_id))
            conn.commit()
        except Exception:
            # Keep bot operation independent from disk/telemetry failure.
            self.state = "unavailable"
            self.dropped += 1
        finally:
            if conn:
                conn.close()


_monitor = None


def configure(path=None):
    global _monitor
    if _monitor is None:
        _monitor = Monitor(path or Path(__file__).resolve().parent.parent / "data" / "admin_monitor.db")
        atexit.register(_monitor.close)
    return _monitor


def begin(conversation_id, user_id):
    return _monitor.begin(conversation_id, user_id) if _monitor else None


def phase(request_id, status, queue_ms=None):
    if _monitor and request_id:
        _monitor.phase(request_id, status, queue_ms)


def finish(request_id, status, total_ms, error_kind=None, spans=None):
    if _monitor and request_id:
        _monitor.finish(request_id, status, total_ms, error_kind, spans)


def ready():
    if _monitor:
        _monitor.state = "ready_seen"
