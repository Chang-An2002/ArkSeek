"""Targeted admin and QQ transport tests using only temporary synthetic data."""
import ast
import asyncio
import hashlib
import json
import sqlite3
import sys
import tempfile
import time
import unittest
from collections import defaultdict
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))
from aiohttp.test_utils import TestClient, TestServer
from aiohttp import ClientTimeout
import admin_monitor
from timing import Trace
from .demo import create_demo
from .repository import Repository, DataUnavailable
from .server import create_app


@contextmanager
def test_connection(path):
    conn = sqlite3.connect(path)
    try:
        with conn:
            yield conn
    finally:
        conn.close()


class RepositoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        create_demo(self.root)
        self.repo = Repository(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_queries_do_not_modify_original_database(self):
        digest = lambda: hashlib.sha256((self.root / "memory.db").read_bytes()).hexdigest()
        before = digest()
        self.repo.overview(); self.repo.sessions(); self.repo.users()
        self.repo.conversation("qq_group:demo"); self.repo.user("qq_group:demo:a")
        self.assertEqual(before, digest())
        with self.repo.connect() as conn:
            with self.assertRaises(sqlite3.OperationalError):
                conn.execute("DELETE FROM user_profile")

    def test_missing_store_is_not_created(self):
        missing = self.root / "absent"
        repo = Repository(missing)
        with self.assertRaises(DataUnavailable):
            repo.sessions()
        self.assertFalse(missing.exists())

    def test_group_user_boundaries_and_reply_target(self):
        users = self.repo.users()["items"]
        self.assertEqual(len(users), 3)
        self.assertNotIn("qq_group:demo", {u["id"] for u in users})
        c = self.repo.conversation("qq_group:demo")
        self.assertEqual(c["items"][1]["person_id"], "qq_group:demo:a")
        self.assertEqual(c["items"][3]["person_id"], "qq_group:demo:b")

    def test_equal_nicknames_are_not_merged(self):
        with test_connection(self.root / "memory.db") as conn:
            conn.execute("UPDATE display_name SET name='同名' WHERE user_id LIKE 'qq_group:%'")
        users = self.repo.users(query="同名")["items"]
        self.assertEqual(len(users), 2)
        self.assertNotEqual(users[0]["id"], users[1]["id"])

    def test_protected_profile_visible_as_disabled_storage(self):
        fact = self.repo.user("qq_group:demo:b")["profiles"][0]
        self.assertFalse(fact["effective"])
        self.assertEqual(fact["value"], "未提供")

    def test_legacy_rows_have_unknown_group_identity(self):
        with test_connection(self.root / "memory.db") as conn:
            conn.execute("UPDATE conversation_turns SET speaker=NULL WHERE user_id='qq_group:demo'")
        row = self.repo.conversation("qq_group:demo")["items"][0]
        self.assertIsNone(row["person_id"])
        self.assertEqual(row["speaker_name"], "身份未记录")

    def test_backward_pagination_and_query(self):
        first = self.repo.conversation("qq_group:demo", limit=2)
        second = self.repo.conversation("qq_group:demo", limit=2, before=first["next_before"])
        self.assertTrue(first["has_more"])
        self.assertLess(second["items"][-1]["turn_index"], first["items"][0]["turn_index"])
        self.assertEqual(self.repo.sessions(query="成员乙")["total"], 1)
        self.assertEqual(self.repo.sessions(query="' OR 1=1 --")["total"], 0)

    def test_stale_requests_never_claim_processing(self):
        with test_connection(self.root / "admin_monitor.db") as conn:
            conn.execute("UPDATE requests SET status='processing' WHERE request_id='demo-0'")
        d = self.repo.monitor()
        self.assertEqual(d["summary"]["processing"], 0)
        self.assertEqual(d["summary"]["unknown"], 1)
        self.assertEqual(self.repo.request("demo-0")["status"], "unknown")

    def test_selected_request_can_refresh_after_filter_changes(self):
        with test_connection(self.root / "admin_monitor.db") as conn:
            conn.execute("UPDATE requests SET status='sent' WHERE request_id='demo-2'")
        self.assertEqual(self.repo.request("demo-2")["status"], "sent")
        self.assertNotIn("demo-2", {r["request_id"] for r in self.repo.monitor(status="failed")["items"]})

    def test_log_path_traversal_and_bounded_tail(self):
        with self.assertRaises(KeyError):
            self.repo.log_tail("../memory.db")
        path = self.root / "runtime_logs" / "bot_big.log"
        path.write_text("旧行\n" * 100000 + "SECRET=abcdef123\n最近行\n", encoding="utf-8")
        result = self.repo.log_tail(path.name, limit=2)
        self.assertEqual(len(result["lines"]), 2)
        self.assertTrue(result["truncated"])
        self.assertNotIn("abcdef123", "\n".join(result["lines"]))


class HTTPTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        create_demo(self.root)
        self.client = TestClient(TestServer(create_app(self.root, demo=True)), timeout=ClientTimeout(total=5))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()
        self.tmp.cleanup()

    async def test_all_pages_and_read_endpoints(self):
        endpoints = ["/", "/app.js", "/app.css", "/api/overview", "/api/sessions", "/api/conversation?id=qq_group:demo", "/api/users", "/api/user?id=qq_group:demo:a", "/api/monitor", "/api/request?id=demo-2", "/api/logs", "/api/log?name=bot_demo.log"]
        for endpoint in endpoints:
            response = await self.client.get(endpoint)
            self.assertEqual(response.status, 200, endpoint)
            self.assertEqual(response.headers.get("Cache-Control"), "no-store")

    async def test_no_write_routes(self):
        for method in ("post", "put", "delete"):
            response = await getattr(self.client, method)("/api/user?id=qq_group:demo:a", json={"value":"change"})
            self.assertEqual(response.status, 405)

    async def test_host_origin_and_cross_site_boundaries(self):
        cases = [{"Host":"evil.test"}, {"Origin":"https://evil.test"}, {"Sec-Fetch-Site":"cross-site"}]
        for headers in cases:
            response = await self.client.get("/api/overview", headers=headers)
            self.assertEqual(response.status, 403)

    async def test_pagination_validation_and_missing_records(self):
        for endpoint in ("/api/users?limit=1000", "/api/sessions?offset=-1", "/api/monitor?hours=999", "/api/monitor?status=invented"):
            response = await self.client.get(endpoint)
            self.assertEqual(response.status, 400, endpoint)
        response = await self.client.get("/api/log?name=../memory.db")
        self.assertEqual(response.status, 404)

    async def test_unavailable_monitor_has_honest_empty_state(self):
        (self.root / "admin_monitor.db").unlink()
        response = await self.client.get("/api/monitor")
        self.assertEqual(response.status, 503)
        data = await (await self.client.get("/api/overview")).json()
        self.assertFalse(data["monitor_available"])
        self.assertTrue(data["memory_available"])


class TelemetryTests(unittest.TestCase):
    def test_writer_lifecycle_and_no_message_detail(self):
        with tempfile.TemporaryDirectory() as tmp:
            monitor = admin_monitor.Monitor(Path(tmp) / "monitor.db")
            rid = monitor.begin("group:test", "member:test")
            monitor.phase(rid, "processing", 1200)
            monitor.finish(rid, "sent", 5000, spans=[(0, 1, "LLM", "private prompt SHOULD_NOT_PERSIST", "q1")])
            monitor.close()
            self.assertFalse(monitor.thread.is_alive())
            with test_connection(monitor.path) as conn:
                row = conn.execute("SELECT status,queue_ms,total_ms,spans_json FROM requests").fetchone()
                self.assertEqual(row[:3], ("sent", 1200, 5000))
                self.assertNotIn("SHOULD_NOT_PERSIST", row[3])
                self.assertEqual(conn.execute("SELECT state FROM runtime").fetchone()[0], "stopped")

    def test_disk_failure_never_raises_in_bot_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "not-a-directory"
            file.write_text("x")
            monitor = admin_monitor.Monitor(file / "monitor.db")
            monitor.begin("group:test", "member:test")
            monitor.close()
            self.assertEqual(monitor.state, "unavailable")


def transport_class():
    # Extract the transport class without importing pipeline/models/.env or SDK.
    module = ast.parse((BASE / "src" / "qq_bot.py").read_text(encoding="utf-8-sig"))
    node = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "MemoryBotClient")
    node.bases = []
    import re
    namespace = {"asyncio":asyncio,"time":time,"Trace":Trace,"admin_monitor":admin_monitor,"_log":Mock(),
                 "_FORGET_COMMANDS":("/forget",),"intro_message":lambda real:"演示帮助",
                 "SHOW_SOURCE_HEADERS":False,"_HEADER_RE":re.compile(r"^【L\d[^】]*】\s*", re.M),
                 "_URL_RE":re.compile(r"https?://[^\s]+")}
    exec(compile(ast.Module(body=[node],type_ignores=[]),"transport-test","exec"),namespace)
    return namespace["MemoryBotClient"]


class TransportTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        cls = transport_class()
        self.bot = object.__new__(cls)
        self.bot.llm_client = Mock(is_real=True)
        self.bot._conv_locks = defaultdict(asyncio.Lock)
        self.bot._summary_tasks = {}
        self.bot.worker = ThreadPoolExecutor(max_workers=2)
        self.bot._handle = lambda *args: "演示回答"
        self.bot._after_reply = lambda *args: None
        self.tmp = tempfile.TemporaryDirectory()
        self.monitor = admin_monitor.Monitor(Path(self.tmp.name) / "monitor.db")
        self.previous = admin_monitor._monitor
        admin_monitor._monitor = self.monitor
        self.posts = []

    async def asyncTearDown(self):
        await asyncio.gather(*self.bot._summary_tasks.values(), return_exceptions=True)
        self.bot.worker.shutdown(wait=True)
        self.monitor.close()
        admin_monitor._monitor = self.previous
        self.tmp.cleanup()

    async def post(self, content, seq):
        self.posts.append((content, seq))

    def statuses(self):
        self.monitor.queue.join()
        with test_connection(self.monitor.path) as conn:
            return [r[0] for r in conn.execute("SELECT status FROM requests ORDER BY received_at")]

    async def test_success_has_sent_status_and_one_reply(self):
        await self.bot._answer(self.post, "user:test", "问句")
        self.assertEqual(self.statuses(), ["sent"])
        self.assertEqual(self.posts, [("演示回答", 1)])

    async def test_send_failure_not_misreported_as_success(self):
        async def failing(content, seq):
            raise RuntimeError("synthetic failure")
        await self.bot._answer(failing, "user:test", "问句")
        self.assertEqual(self.statuses(), ["send_failed"])

    async def test_url_retry_uses_second_sequence_and_reports_sent(self):
        self.bot._handle = lambda *args: "演示回答 https://example.test/page"
        async def retry(content, seq):
            self.posts.append((content, seq))
            if seq == 1:
                raise RuntimeError("synthetic rejection")
        await self.bot._answer(retry, "user:test", "问句")
        self.assertEqual([seq for _,seq in self.posts], [1,2])
        self.assertNotIn("https://", self.posts[-1][0])
        self.assertEqual(self.statuses(), ["sent"])

    async def test_processing_failure_keeps_error_reply_and_distinct_status(self):
        def fail(*args):
            raise ValueError("synthetic processing failure")
        self.bot._handle = fail
        await self.bot._answer(self.post, "user:test", "问句")
        self.assertEqual(self.statuses(), ["processing_failed"])
        self.assertEqual(len(self.posts), 1)

    async def test_help_does_not_enter_kernel(self):
        self.bot._handle = Mock(side_effect=AssertionError("help entered kernel"))
        await self.bot._answer(self.post, "user:test", "/help")
        self.bot._handle.assert_not_called()
        self.assertEqual(self.statuses(), ["sent"])

    async def test_same_group_order_and_queue_time_preserved(self):
        def handle(uid, text, *args):
            if text == "first":
                time.sleep(0.08)
            return text
        self.bot._handle = handle
        await asyncio.gather(self.bot._answer(self.post,"member:a","first","group:test"),self.bot._answer(self.post,"member:b","second","group:test"))
        self.assertEqual([text for text,_ in self.posts], ["first","second"])
        self.assertEqual(self.statuses(), ["sent","sent"])
        with test_connection(self.monitor.path) as conn:
            self.assertGreater(conn.execute("SELECT MAX(queue_ms) FROM requests").fetchone()[0], 40)


if __name__ == "__main__":
    unittest.main()
