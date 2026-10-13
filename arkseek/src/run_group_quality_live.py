"""真实LLM匿名冒烟测试；调用付费接口，但只使用内存库和合成资料，不读群聊。"""

from types import SimpleNamespace

from bot_profile import OWNER_NAME
from conversation_memory import ConversationSession
from llm_client import build_llm_client
from llm_generate import llm_generate_answer, is_no_answer
from llm_l1 import llm_l1_extract
from memory_store import MemoryStore
from pipeline import Chunk, RetrievedPassage
from wiki_router import route_query


def main():
    client = build_llm_client()
    if not client.is_real:
        raise RuntimeError("真实LLM不可用，本次不以占位版冒充验证通过")
    store = MemoryStore(":memory:")
    pipeline = SimpleNamespace(dense=SimpleNamespace(model=None), reranker=SimpleNamespace(is_real=False))
    session = ConversationSession(pipeline, store, "synthetic-b", conversation_id="synthetic-group", llm_client=client)
    store.set_display_name("synthetic-a", "测试甲")
    store.set_display_name("synthetic-b", "测试乙")
    store.upsert_profile_fact("synthetic-b", "爱好", "徒步", "我喜欢徒步")
    store.append_turn("synthetic-group", "user", "我喜欢烘焙", speaker="synthetic-a")
    store.append_turn("synthetic-group", "assistant", "你可以试试烤面包")
    store.append_turn("synthetic-group", "user", "我喜欢徒步", speaker="synthetic-b")
    store.append_turn("synthetic-group", "assistant", "了解你的爱好")
    context = session._build_planner_context(session.get_context_window())
    outcomes = []
    for query in ("我是谁，有什么爱好", "你能看到我的QQ昵称吗"):
        result = route_query(query, client, context)
        passed = bool(result and result["route"] == "general" and "测试乙" in result["reply"])
        if "爱好" in query:
            passed = passed and "徒步" in result["reply"] and "烘焙" not in result["reply"]
        print(f"{'PASS' if passed else 'FAIL'} 身份：{query} -> {result}")
        outcomes.append(passed)

    updates = llm_l1_extract("睡不好是不是也造成血压不稳定", {}, client, context)
    passed = updates == []
    print(f"{'PASS' if passed else 'FAIL'} 泛泛健康问句不写画像：{updates}")
    outcomes.append(passed)

    store.append_turn("synthetic-group", "user", f"请介绍{OWNER_NAME}的课题", speaker="synthetic-a")
    store.append_turn("synthetic-group", "assistant", f"{OWNER_NAME}的课题研究肝脏损伤后不同细胞的来源与变化。")
    context = session._build_planner_context(session.get_context_window())
    for query in ("这些语言太专业了，请简明扼要地说一下在干什么", "用通俗的语言解释一下这个课题"):
        result = route_query(query, client, context)
        topic = result.get("retrieval_query", "") if result else ""
        passed = bool(result and result["route"] == "local_kb" and ("肝" in topic or "课题" in topic))
        print(f"{'PASS' if passed else 'FAIL'} 通俗追问：{result}")
        outcomes.append(passed)

    passages = [RetrievedPassage(Chunk(0, f"{OWNER_NAME}的匿名课题报告（合成测试资料）",
        "通过细胞内天然遗传变异识别细胞克隆，比较正常与肝脏损伤状态下细胞来源和分化状态。"), 1, 1, 1)]
    answer = llm_generate_answer("请用通俗的语言说一下这个课题在干什么", passages, client, context)
    passed = bool(answer and not is_no_answer(answer) and "[1]" in answer)
    print(f"{'PASS' if passed else 'FAIL'} 根据资料通俗整理：{answer}")
    outcomes.append(passed)
    store.conn.close()
    print(f"真实LLM匿名冒烟：{sum(outcomes)}/{len(outcomes)}通过")
    return 0 if all(outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
