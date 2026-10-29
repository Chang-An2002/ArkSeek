"""Management invariants; every database/process is temporary and fictional."""
import hashlib
import ast
import json
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from .test_admin import BASE, test_connection
from .demo import create_demo
from .management import Management, Conflict
from .server import create_app
from memory_store import MemoryStore
from profile_rules import SCHEMA as PROFILE_SCHEMA
from runtime_settings import RunLease, RuntimeBusy, read_settings
from admin_knowledge import load_admin_chunks
from aiohttp.test_utils import TestClient, TestServer


class ManagementTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        create_demo(self.root)
        self.manager = Management(self.root, enabled=True, demo=True)
        self.uid = "qq_group:demo:a"

    def tearDown(self):
        self.manager.close()
        self.tmp.cleanup()

    def action(self, **payload):
        return self.manager.commit(self.manager.preview({"reason": "虚构管理测试", **payload})["token"])

    def profile(self, mode, **extra):
        return self.action(kind="profile", user_id=self.uid, key="喜欢", mode=mode, **extra)

    def test_preview_expiry_and_replay_never_change_original_rows(self):
        path = self.root / "memory.db"
        before = path.read_bytes()
        p = self.manager.preview(dict(kind="profile",user_id=self.uid,key="喜欢",mode="disabled",reason="测试预览"))
        self.assertEqual(path.read_bytes(), before)
        self.manager.previews[p["token"]]["expires"] = 0
        with self.assertRaises(Conflict):
            self.manager.commit(p["token"])
        self.assertEqual(path.read_bytes(), before)
        self.profile("disabled")
        with self.assertRaises(Conflict):
            self.manager.commit(p["token"])

    def test_override_disabled_and_automatic_are_consumed_by_real_store(self):
        store = MemoryStore(str(self.root / "memory.db"))
        try:
            self.profile("override", value="合作解谜游戏")
            self.assertEqual(store.get_profile(self.uid)["喜欢"], "合作解谜游戏")
            store.upsert_profile_fact(self.uid, "喜欢", "旧值自动覆盖", "测试来源")
            store.delete_profile_fact(self.uid, "喜欢")
            self.assertEqual(store.get_profile(self.uid)["喜欢"], "合作解谜游戏")
            row = store.conn.execute("SELECT value,source_text FROM user_profile WHERE user_id=?", (self.uid,)).fetchone()
            self.assertEqual(tuple(row), ("策略游戏", "我平时喜欢策略游戏。"))
            self.profile("disabled")
            self.assertNotIn("喜欢", store.get_profile(self.uid))
            store.upsert_profile_fact(self.uid, "喜欢", "试图重新抽取")
            self.assertNotIn("喜欢", store.get_profile(self.uid))
            self.profile("automatic")
            self.assertEqual(store.get_profile(self.uid)["喜欢"], "策略游戏")
            store.upsert_profile_fact(self.uid, "喜欢", "新偏好")
            self.assertEqual(store.get_profile(self.uid)["喜欢"], "新偏好")
            self.assertEqual(store.get_profile("qq_c2c:demo")["回答偏好"], "先给结论，再解释")
        finally:
            store.close()

    def test_profile_history_undo_and_stale_conflict(self):
        p1 = self.manager.preview(dict(kind="profile",user_id=self.uid,key="喜欢",mode="override",value="桌游",reason="虚构更正"))
        p2 = self.manager.preview(dict(kind="profile",user_id=self.uid,key="喜欢",mode="disabled",reason="虚构停用"))
        e = self.manager.commit(p1["token"])
        with self.assertRaises(Conflict):
            self.manager.commit(p2["token"])
        self.action(kind="profile_restore", user_id=self.uid, key="喜欢", event_id=e["event_id"])
        self.assertIsNone(self.manager.profile_state(self.uid,"喜欢")["rule"])
        self.assertEqual(len(self.manager.profile_history(self.uid,"喜欢")), 2)
        with self.assertRaises(Conflict):
            self.action(kind="profile_restore",user_id=self.uid,key="喜欢",event_id=e["event_id"])

    def test_changed_raw_quote_invalidates_preview(self):
        p = self.manager.preview(dict(kind="profile",user_id=self.uid,key="喜欢",mode="disabled",reason="虚构停用"))
        with test_connection(self.root / "memory.db") as conn:
            conn.execute("UPDATE user_profile SET source_text='新的原始引用' WHERE user_id=?", (self.uid,))
        with self.assertRaises(Conflict):
            self.manager.commit(p["token"])

    def test_failed_audit_rolls_back_profile_decision(self):
        with test_connection(self.root / "memory.db") as conn:
            conn.executescript(PROFILE_SCHEMA)
            conn.execute("CREATE TRIGGER reject_audit BEFORE INSERT ON admin_profile_history BEGIN SELECT RAISE(ABORT, 'test'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            self.profile("disabled")
        self.assertIsNone(self.manager.profile_state(self.uid,"喜欢")["rule"])

    def test_locked_profile_never_emits_false_memory_updated_note(self):
        source = ast.parse((BASE / 'src' / 'conversation_memory.py').read_text(encoding='utf-8-sig'))
        klass = next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name=='ConversationSession')
        methods = [n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name in ('_l1_write','_l1_step')]
        space = dict(Optional=__import__('typing').Optional, PROFILE_DELETE='（无）', llm_l1_extract=lambda *a,**k:[('喜欢','自动新偏好')], extract_facts=lambda text:[('喜欢','自动新偏好')], looks_like_question=lambda text:False)
        exec(compile(ast.Module(body=methods,type_ignores=[]),'<isolated L1 methods>','exec'),space)
        self.profile('disabled')
        store = MemoryStore(str(self.root / 'memory.db'))
        try:
            fake = SimpleNamespace(store=store,user_id=self.uid,is_group=False,llm_client=None,_try_answer_from_profile=lambda text:None)
            self.assertEqual(space['_l1_write'](fake,'我喜欢自动新偏好'),[])
            self.assertIsNone(space['_l1_step'](fake,'我喜欢自动新偏好'))
        finally:
            store.close()

    def test_maintenance_lease_blocks_writes_and_releases_after_close(self):
        p = self.manager.preview(dict(kind="profile",user_id=self.uid,key="喜欢",mode="disabled",reason="虚构停用"))
        with RunLease(self.root):
            with self.assertRaises(RuntimeBusy):
                self.manager.commit(p["token"])
        self.profile("disabled")

    def test_forget_does_not_resurrect_administrator_overrides(self):
        self.profile("override", value="新偏好")
        store = MemoryStore(str(self.root / "memory.db"))
        try:
            store.forget_member("qq_group:demo", self.uid)
            self.assertEqual(store.get_profile(self.uid), {})
            store.upsert_profile_fact(self.uid,"喜欢","全新的偏好")
            self.assertEqual(store.get_profile(self.uid)["喜欢"], "全新的偏好")
        finally:
            store.close()

    def test_document_copy_versions_disable_and_loader(self):
        original = self.root / "original.txt"
        original.write_text("外部原始文件永远保留", encoding="utf-8")
        before = hashlib.sha256(original.read_bytes()).hexdigest()
        e = self.action(kind="document", title="虚构文本", body="这份文本是管理员导入的虚构知识。", active=True)
        created = self.manager.event(e["event_id"])["after"]
        chunks = load_admin_chunks(self.root, 100)
        self.assertIn(100, [c.chunk_id for c in chunks])
        self.assertTrue(any(c.meta["doc_id"] == "admin:"+created["id"] for c in chunks))
        e2 = self.action(kind="document",id=created["id"],title=created["title"],body=created["body"],active=False)
        self.assertFalse(any(c.meta["doc_id"] == "admin:"+created["id"] for c in load_admin_chunks(self.root)))
        self.action(kind="document_restore",event_id=e2["event_id"])
        self.assertTrue(self.manager.document(created["id"])["active"])
        with self.assertRaises(Conflict):
            self.action(kind="document_restore",event_id=e2["event_id"])
        self.assertEqual(hashlib.sha256(original.read_bytes()).hexdigest(), before)
        self.assertEqual(self.manager.document(created["id"])["revision"], 3)

    def test_settings_validation_undo_and_env_is_untouched(self):
        env = self.root / ".env"
        env.write_text("TEST_VALUE=fictional-private-value", encoding="utf-8")
        for values in ({"SECRET":"bad"}, {"ARKSEEK_LLM_MAX_CONCURRENCY":0}, {"ARKSEEK_LLM_TIMEOUT":300,"ARKSEEK_LLM_MAX_DURATION":120}):
            with self.assertRaises(ValueError):
                self.action(kind="settings", values=values)
        e = self.action(kind="settings", values={"ARKSEEK_LLM_MAX_CONCURRENCY":2})
        self.assertEqual(read_settings(self.root)["ARKSEEK_LLM_MAX_CONCURRENCY"], "2")
        self.action(kind="settings_restore",event_id=e["event_id"])
        self.assertIsNone(read_settings(self.root))
        self.assertEqual(env.read_text(encoding="utf-8"), "TEST_VALUE=fictional-private-value")

    def test_read_only_management_does_not_create_control_store(self):
        other = self.root / "not-created"
        readonly = Management(other)
        self.assertFalse(readonly.status()["enabled"])
        with self.assertRaises(PermissionError):
            readonly.preview({"kind":"bot_start", "reason":"不允许操作"})
        self.assertFalse(other.exists())

    def wait_for(self, predicate):
        deadline = time.monotonic()+8
        while time.monotonic() < deadline:
            result = predicate()
            if result:
                return result
            time.sleep(.03)
        self.fail("虚构工作进程未在期限内达到预期状态")

    def test_demo_process_ownership_idle_checks_and_stop(self):
        with self.assertRaises(Conflict):
            self.action(kind="bot_stop", pid=12345)
        started = self.action(kind="bot_start")
        self.wait_for(lambda: any(j["result"].get("stage")=="demo_running" for j in self.manager.status()["jobs"]))
        self.assertTrue(self.manager.status()["bot"]["alive"])
        with self.assertRaises(Conflict):
            self.profile("disabled")
        self.action(kind="bot_stop")
        self.assertFalse(self.manager.status()["bot"]["alive"])
        self.assertEqual(self.manager.children[started["job_id"]].poll() is not None, True)
        self.assertEqual(self.manager.status()["jobs"][0]["status"], "stopped")
        self.profile("disabled")

    def test_demo_index_reports_real_completion_without_models(self):
        started = self.action(kind="index")
        self.wait_for(lambda: self.manager.children[started["job_id"]].poll() is not None)
        job = next(j for j in self.manager.status()["jobs"] if j["id"]==started["job_id"])
        self.assertEqual(job["status"], "succeeded")
        self.assertEqual(job["result"]["stage"], "done")
        self.assertTrue(job["result"]["demo"])
        self.assertGreater(job["result"]["chunks"], 0)


class ManagementHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        create_demo(self.root)
        self.client = TestClient(TestServer(create_app(self.root,demo=True,manage=True)))
        await self.client.start_server()
        self.origin = str(self.client.make_url('/')).rstrip('/')
        d = await (await self.client.get('/api/manage/status')).json()
        self.headers = {"Origin":self.origin,"X-Admin-Token":d["csrf"]}

    async def asyncTearDown(self):
        await self.client.close()
        self.tmp.cleanup()

    async def test_write_requires_same_origin_token_preview_and_single_commit(self):
        payload = {"kind":"profile","user_id":"qq_group:demo:a","key":"喜欢","mode":"disabled","reason":"虚构HTTP测试"}
        for headers in ({}, {"Origin":self.origin}, {**self.headers,"Origin":"https://evil.test"}):
            r = await self.client.post('/api/manage/preview',json=payload,headers=headers)
            self.assertEqual(r.status,403)
        r = await self.client.post('/api/manage/preview',json=payload,headers=self.headers)
        self.assertEqual(r.status,200)
        p = await r.json()
        self.assertIn("不删除",p["impact"])
        r = await self.client.post('/api/manage/commit',json={"token":p["token"]},headers=self.headers)
        self.assertEqual(r.status,200)
        r = await self.client.post('/api/manage/commit',json={"token":p["token"]},headers=self.headers)
        self.assertEqual(r.status,409)
        user = await (await self.client.get('/api/user?id=qq_group:demo:a')).json()
        self.assertFalse(user["profiles"][0]["effective"])
        self.assertEqual(user["profiles"][0]["management"]["mode"],"disabled")

    async def test_validation_write_routes_and_static_assets(self):
        r = await self.client.post('/api/manage/preview',json={"kind":"shell","reason":"非法命令"},headers=self.headers)
        self.assertEqual(r.status,400)
        r = await self.client.get('/management.js')
        self.assertEqual(r.status,200)
        r = await self.client.delete('/api/manage/document?id=demo-guide',headers=self.headers)
        self.assertEqual(r.status,405)
