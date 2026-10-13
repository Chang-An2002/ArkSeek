"""匿名群聊结构回归：只用内存数据库、假模型和假网络。"""

import io
import json
import logging
import tempfile
import unittest
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import llm_client
from conversation_memory import ConversationSession, GROUP_SUMMARY_PREFIX
from fake_llm_client import ScriptedLLMClient
from length_limit import split_reply, count_units
from memory_store import MemoryStore
from pipeline import Chunk, RetrievedPassage
from profile_facts import supported_profile_fact
from prts_wiki_client import _locate_page
from query_planner import rewrite_for_history
from runtime_logging import ConsoleTee
from wiki_router import _parse_response


class StubPipeline:
    dense = SimpleNamespace(model=None)
    reranker = SimpleNamespace(is_real=False)
    REJECT_MESSAGE = "没有可靠资料"


class GroupQualityTests(unittest.TestCase):
    def setUp(self):
        self.store = MemoryStore(":memory:")
        self.session = ConversationSession(StubPipeline(), self.store, "member-b", conversation_id="test-group")
        self.store.set_display_name("member-a", "测试甲")
        self.store.set_display_name("member-b", "测试乙")
        self.store.append_turn("test-group", "user", "我喜欢烘焙", speaker="member-a")
        self.store.append_turn("test-group", "assistant", "你可以试试烤面包")
        self.store.append_turn("test-group", "user", "睡不好是不是影响血压", speaker="member-b")
        self.store.append_turn("test-group", "assistant", "一般而言可能有关")
        self.store.append_turn("test-group", "user", "我是谁", speaker="member-b")

    def tearDown(self):
        self.store.conn.close()

    def test_placeholder_does_not_override_platform_nickname(self):
        self.store.upsert_profile_fact("member-b", "称呼", "未提供", "最近睡不好")
        names = self.session._speaker_names(self.store.get_all_turns("test-group"))
        self.assertEqual(names["member-b"], "测试乙")

    def test_generic_health_question_is_not_personal_fact(self):
        self.store.upsert_profile_fact("member-b", "健康状况", "血压不稳定", "睡不好是不是影响血压")
        self.assertNotIn("健康状况", self.store.get_profile("member-b"))
        self.assertTrue(supported_profile_fact("健康状况", "高血压", "我有高血压，最近睡不好，怎么办"))

    def test_labels_distinguish_equal_nicknames_and_assistant_recipient(self):
        self.store.set_display_name("member-b", "测试甲")
        turns = self.store.get_all_turns("test-group")
        rows = self.session._labeled(turns[1:], self.session._speaker_names(turns))
        labels = self.session._participant_labels(self.session._speaker_names(turns))
        self.assertIn("助手回复给" + labels["member-a"], rows[0]["text"])
        self.assertIn(labels["member-b"], rows[1]["text"])
        self.assertNotEqual(labels["member-a"], labels["member-b"])

    def test_participant_directory_visible_with_empty_profile(self):
        context = self.session._build_planner_context(self.session.get_context_window())
        self.assertIn("测试乙", context[0]["text"])
        self.assertIn("称呼确实可见", context[0]["text"])

    def test_legacy_summary_is_not_used_as_identity_context(self):
        self.store.set_l0_summary("test-group", "测试甲睡不好，喜欢烘焙", 3)
        self.assertIsNone(self.session.get_session_summary())
        self.store.set_l0_summary("test-group", GROUP_SUMMARY_PREFIX + "群友1聊烘焙", 3)
        self.assertIsNotNone(self.session.get_session_summary())

    def test_old_summary_rebuilds_from_labeled_originals(self):
        for position in range(5):
            self.store.append_turn("test-group", "assistant", f"测试回复{position}")
            self.store.append_turn("test-group", "user", "新话题", speaker="member-a")
        self.store.set_l0_summary("test-group", "旧的混合摘要", 8)
        with patch("conversation_memory.llm_update_summary", return_value="群友1聊烘焙") as generate:
            self.session._update_l0_summary_locked()
        self.assertIsNone(generate.call_args.args[0])
        self.assertIn("助手回复给群友-", generate.call_args.args[1])
        self.assertTrue(self.session.get_session_summary().startswith(GROUP_SUMMARY_PREFIX))

    def test_personal_history_excludes_other_members_questions_and_answers(self):
        passages = self.session._recent_window_passages()
        own = self.session._filter_personal_passages("我的爱好是什么", passages)
        self.assertTrue(own)
        self.assertEqual({passage.chunk.chunk_id for passage in own}, {2, 3})
        self.assertEqual(len(self.session._filter_personal_passages("谁问过烘焙", passages)), len(passages))
        self.assertFalse(self.session._personal_history("我的朋友之前说喜欢什么"))

    def test_new_invalid_profile_writes_are_rejected(self):
        self.session.llm_client = ScriptedLLMClient([{"caller": "llm_l1_extract()", "response": {
            "updates": [{"key": "称呼", "value": "未提供"}, {"key": "健康状况", "value": "睡眠不好"}]}}])
        self.assertEqual(self.session._l1_write("睡不好是不是影响血压"), [])
        self.assertEqual(self.store.get_profile("member-b"), {})

    def test_retrieval_theme_and_answer_style_are_separate(self):
        self.session.llm_client = SimpleNamespace(is_real=True)
        passage = RetrievedPassage(Chunk(0, "匿名课题报告", "用细胞天然标记研究肝脏损伤后细胞来源"), 1, 1, 1)
        queries = []

        def retrieve(query, **kwargs):
            queries.append(query)
            return [passage], True

        self.session.kb_pipeline.retrieve_with_gate = retrieve
        with patch("conversation_memory.llm_generate_answer", return_value="研究肝脏如何修复。[1]") as generate:
            result = self.session._try_answer_from_local_kb("通俗说一下他在干什么", retrieval_query="课题研究内容")
        self.assertTrue(result.ok)
        self.assertEqual(queries, ["课题研究内容"])
        self.assertEqual(generate.call_args.args[0], "通俗说一下他在干什么")

    def test_stable_identity_does_not_depend_on_member_order(self):
        before = self.session._participant_labels({"member-a": "测试甲", "member-b": "测试乙"})
        after = self.session._participant_labels({"member-b": "测试乙"})
        self.assertEqual(before["member-b"], after["member-b"])

    def test_vector_scope_is_applied_before_top_k(self):
        from chroma_index import HistoryVectorIndex
        index = object.__new__(HistoryVectorIndex)
        index.model = None
        index.collection = unittest.mock.Mock()
        index.collection.query.return_value = {"metadatas": [[{"turn_index": 2}]], "distances": [[0.1]]}
        with patch("chroma_index._encode", return_value=[[1.0]]):
            result = index.search("test-group", "我的爱好", {2: 0, 3: 1}, 10)
        self.assertEqual(result[0][0], 0)
        arguments = index.collection.query.call_args.kwargs
        self.assertEqual(arguments["n_results"], 2)
        self.assertEqual(arguments["where"]["$and"][1], {"turn_index": {"$in": [2, 3]}})

    def test_no_answer_can_retry_once_for_original_record(self):
        self.session.llm_client = SimpleNamespace(is_real=True)
        self.session.kb_pipeline.reranker = SimpleNamespace(is_real=True)
        passage = RetrievedPassage(Chunk(2, "旧问题", "之前问过这个话题"), 1, 1, 1, relevance_score=0.9)
        original = RetrievedPassage(Chunk(0, "原始问题", "推荐过的资料"), 1, 1, 1, relevance_score=0.8)
        with patch.object(self.session, "_search_history", return_value=[passage]), \
             patch.object(self.session, "_recent_window_passages", return_value=[]), \
             patch.object(self.session, "_with_following_replies", side_effect=lambda evidence: evidence), \
             patch.object(self.session, "_retry_history_with_rewrites", return_value=[original]) as rewrite, \
             patch.object(self.session, "_answer_history_with", side_effect=[None, "原问答答案"]) as answer:
            result = self.session._try_answer_from_history("之前推荐的资料叫什么")
        self.assertEqual(result, "原问答答案")
        self.assertEqual(rewrite.call_count, 1)
        self.assertEqual(answer.call_count, 2)

    def test_old_direct_evidence_not_overwhelmed_by_recency(self):
        def rerank(query, passages, top_n):
            for passage in passages:
                passage.rerank_score = 0.95 if "原始推荐" in passage.chunk.text else 0.65

        self.session.kb_pipeline.reranker = SimpleNamespace(is_real=True, rerank=rerank)
        self.store.append_turn("test-group", "assistant", "原始推荐：阅读《细胞故事》")
        for position in range(20):
            self.store.append_turn("test-group", "user", "聊过阅读推荐，但没有具体书名", speaker="member-b")
            self.store.append_turn("test-group", "assistant", "阅读推荐是之前的话题")
        self.store.append_turn("test-group", "user", "之前推荐阅读什么", speaker="member-b")
        result = self.session._search_history("之前推荐阅读什么")
        self.assertTrue(any("原始推荐" in passage.chunk.text for passage in result))

    def test_history_rewrite_receives_topic_context(self):
        client = ScriptedLLMClient([{"caller": "rewrite_for_history()", "response": {"rewrites": ["中药可以考虑……"]}}])
        context = [{"role": "user", "text": "刚才讨论了中药"}]
        self.assertEqual(rewrite_for_history("之前推荐的是什么", client, context), ["中药可以考虑……"])

    def test_router_validates_retrieval_field(self):
        route = {"route": "local_kb", "keyword": "", "reply": "", "is_chitchat": False,
                 "retrieval_query": "肝脏课题研究内容"}
        self.assertEqual(_parse_response(json.dumps(route))["retrieval_query"], route["retrieval_query"])
        route["retrieval_query"] = ["invalid"]
        self.assertEqual(_parse_response(json.dumps(route))["retrieval_query"], "")

    def test_reference_links_do_not_count_as_body(self):
        _, body, _ = split_reply("简单介绍。\n\n参考链接：\n[1] https://example.org/very-long-reference")
        self.assertEqual(count_units(body), count_units("简单介绍。"))

    def test_disambiguation_follows_selected_page_once(self):
        pages = {"关卡": {"title": "关卡", "html": '本页面为消歧义页<ul><li><a href="/w/关卡_甲">甲</a></li><li><a href="/w/关卡_乙">乙</a></li></ul>'},
                 "关卡 乙": {"title": "关卡 乙", "html": "目标关卡正文"}}
        with patch("prts_wiki_client.fetch_page_html", side_effect=lambda title: pages.get(title)) as fetch:
            result = _locate_page("关卡", lambda keyword, candidates: "关卡 乙")
        self.assertEqual(result["title"], "关卡 乙")
        self.assertEqual(fetch.call_count, 2)

    def test_disambiguation_rejects_outside_title_and_loops(self):
        page = {"title": "关卡", "html": '本页面为消歧义页<ul><li><a href="/w/关卡_甲">甲</a></li><li><a href="/w/关卡_乙">乙</a></li></ul>'}
        with patch("prts_wiki_client.fetch_page_html", return_value=page):
            self.assertIsNone(_locate_page("关卡", lambda keyword, candidates: "其他页面"))
            self.assertIsNone(_locate_page("关卡", lambda keyword, candidates: "关卡 甲"))


class StreamAndLogTests(unittest.TestCase):
    def test_active_stream_survives_120_seconds(self):
        chunks = [SimpleNamespace(usage=None, choices=[SimpleNamespace(delta=SimpleNamespace(
            content="整理完成", model_extra={"reasoning_content": "仍在思考"}))])]
        class FakeStream:
            closed = False

            def __iter__(self):
                return iter(chunks)

            def close(self):
                self.closed = True

        stream = FakeStream()
        client = object.__new__(llm_client.OpenAIStyleLLMClient)
        client.model = "fake"
        client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: stream)))
        with patch("llm_client.time.perf_counter", side_effect=[0, 130, 131]), patch.object(llm_client, "LLM_MAX_DURATION", 240):
            result = client._stream_once([], {}, {})
        self.assertEqual(result.choices[0].message.content, "整理完成")
        self.assertTrue(stream.closed)

    def test_safety_cap_and_stall_close_stream_without_partial_reply(self):
        class FakeStream:
            closed = False

            def __iter__(self):
                yield SimpleNamespace(usage=None, choices=[SimpleNamespace(delta=SimpleNamespace(
                    content="未完成", model_extra={}))])

            def close(self):
                self.closed = True

        stream = FakeStream()
        client = object.__new__(llm_client.OpenAIStyleLLMClient)
        client.model = "fake"
        client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: stream)))
        with patch("llm_client.time.perf_counter", side_effect=[0, 241, 242]), patch.object(llm_client, "LLM_MAX_DURATION", 240):
            with self.assertRaises(llm_client.LLMOvertime):
                client._stream_once([], {}, {})
        self.assertTrue(stream.closed)

        class ReadTimeout(Exception):
            pass

        class StalledStream(FakeStream):
            def __iter__(self):
                raise ReadTimeout()

        stream = StalledStream()
        with self.assertRaises(llm_client.LLMStalled):
            client._stream_once([], {}, {})
        self.assertTrue(stream.closed)

    def test_complete_console_lines_redacted_and_rotated(self):
        base = Path(__file__).resolve().parent.parent / "data" / "runtime_logs"
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as directory:
            path = Path(directory) / "test.log"
            handler = RotatingFileHandler(path, maxBytes=80, backupCount=3, encoding="utf-8")
            original = io.StringIO()
            tee = ConsoleTee(original, handler, ["private-secret"])
            tee.write("知识库门槛 ")
            tee.write("通过\nprivate-secret\n")
            tee.write("x" * 90 + "\n")
            tee.finish()
            handler.close()
            saved = "\n".join(item.read_text(encoding="utf-8") for item in Path(directory).glob("test.log*"))
            self.assertIn("知识库门槛 通过", saved)
            self.assertIn("[REDACTED]", saved)
            self.assertNotIn("private-secret", saved)
            self.assertIn("private-secret", original.getvalue())
            self.assertTrue((Path(directory) / "test.log.1").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
