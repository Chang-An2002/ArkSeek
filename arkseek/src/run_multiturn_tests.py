# -*- coding: utf-8 -*-
"""
多轮对话批量测试脚本 —— 阶段二 2.1（不接LLM）

跟 run_tests.py 是同一个思路的延续（读配置文件、批量跑、写成一份Markdown
报告，不用再手动复制粘贴终端输出），区别是这里测的是"多轮对话记不记得住人"，
所以配置文件（multiturn_test_cases.json）里每条用例是一段"脚本化的对话"
（多轮 turns），不是阶段一那种单轮问答。

【怎么判断每一轮"答对了"】
跟阶段一 test_cases.json 的 expect_keywords 思路一样——"格式上给出了回复"
不等于"回复内容真的对"，所以每一轮除了检查"回复是从哪一层（L1/L2/L3）
拿到的"（expect_source），还会检查"回复里是不是真的包含该有的关键词"
（expect_keywords）。source对不对是看回复文本的固定前缀判断的（见
classify_source()），这个前缀是 conversation_memory.py 里各层拼回复时
自己加上去的标记，不是猜出来的。

用法：
    cd src
    python run_multiturn_tests.py
"""

import json
import os
from datetime import datetime

from pipeline import MemorySystemPipeline
from memory_store import MemoryStore
from conversation_memory import ConversationSession


def load_test_cases(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def classify_source(reply: str) -> str:
    """根据回复文本的固定前缀，判断这轮回复是从哪一层拿到的——这些前缀
    是 conversation_memory.py 自己拼的标记，不是靠猜测/关键词判断的。

    2.2接入LLM生成之后，L2/L3在gate通过时有两种可能的前缀：2.1原本"展示
    证据"的格式（"【L2情景记忆——翻到了..."/"检索到N条相关片段"），或者
    LLM生成失败时才会退回的这种格式；以及2.2独有的"【L2情景记忆——LLM生成
    回答"/"【L3知识库——LLM生成回答"。这里统一按"L2"/"L3_answer"分类，不
    区分是原始证据展示还是LLM生成——这个测试脚本关心的是"回复到底来自
    哪一层"，不是"这一层具体用什么格式呈现"，格式差异不影响来源判断的
    正确性。
    """
    if reply.startswith("【L1核心记忆"):
        return "L1"
    if reply.startswith("【L2情景记忆"):
        return "L2"
    if reply.startswith("【未找到可靠信息】"):
        return "L3_reject"
    if reply.startswith("检索到") or reply.startswith("【L3知识库"):
        return "L3_answer"
    return "unknown"


def run_case(kb_pipeline: MemorySystemPipeline, store: MemoryStore, case: dict,
             llm_client=None) -> dict:
    """llm_client 不传就是2.1行为（ConversationSession内部自动退回NoLLMClient），
    这个脚本自己的 main() 就是这么用的，不受影响。传入真实的 build_llm_client()
    返回值，就是2.2行为——run_comparison_tests.py 用这条路径跑2.1/2.2对比，
    复用的是同一份"跑一段脚本化对话+判断来源/关键词"的逻辑，不用另外写一份。
    """
    user_id = case["user_id"]
    store.reset_user(user_id)  # 保证每次重跑这份报告，历史都是从空白开始，不受上一次运行残留影响
    session = ConversationSession(kb_pipeline, store, user_id=user_id, llm_client=llm_client)

    turn_results = []
    for turn in case["turns"]:
        say = turn["say"]
        expect_source = turn.get("expect_source")
        expect_keywords = turn.get("expect_keywords", [])
        note = turn.get("note", "")

        reply = session.user_says(say)
        actual_source = classify_source(reply)
        source_passed = (expect_source is None) or (actual_source == expect_source)

        if expect_keywords:
            missing = [kw for kw in expect_keywords if kw not in reply]
            content_passed = (len(missing) == 0)
        else:
            missing = []
            content_passed = None

        passed = source_passed and (content_passed is not False)

        turn_results.append({
            "say": say, "note": note,
            "expect_source": expect_source, "actual_source": actual_source,
            "source_passed": source_passed,
            "expect_keywords": expect_keywords, "missing_keywords": missing,
            "content_passed": content_passed,
            "passed": passed,
            "reply": reply,
        })

    case_passed = all(t["passed"] for t in turn_results)
    return {"name": case["name"], "user_id": user_id, "turns": turn_results, "passed": case_passed}


def render_markdown(cases_results, dense_backend_name: str, reranker_name: str) -> str:
    lines = []
    lines.append("# 多轮对话记忆测试报告（阶段二2.1，不接LLM）\n")
    lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"当前向量检索后端：**{dense_backend_name}**\n")
    lines.append(f"当前精排后端：**{reranker_name}**\n")

    n_pass = sum(1 for c in cases_results if c["passed"])
    lines.append(f"\n## 汇总：{n_pass}/{len(cases_results)} 段对话全部轮次通过预期\n")
    lines.append("（每段对话里只要有一轮没通过，这段对话就算不通过；"
                  "具体是哪一轮出的问题看下面「逐条详情」）\n")
    lines.append("| 对话场景 | 结果 |")
    lines.append("|---|---|")
    for c in cases_results:
        mark = "✅" if c["passed"] else "❌"
        lines.append(f"| {c['name']} | {mark} |")

    lines.append("\n## 逐条详情\n")
    for c in cases_results:
        lines.append(f"### {c['name']}（user_id={c['user_id']}）")
        lines.append(f"- 综合结果：{'✅ 通过' if c['passed'] else '❌ 未通过'}\n")
        for i, t in enumerate(c["turns"], start=1):
            lines.append(f"**第{i}轮 你说：** {t['say']}")
            lines.append(f"- 备注：{t['note']}")
            source_note = ""
            if t["expect_source"] is not None:
                source_note = (f"，预期来源={t['expect_source']}，实际来源={t['actual_source']}"
                                f"（{'匹配' if t['source_passed'] else '不匹配'}）")
            else:
                source_note = f"，实际来源={t['actual_source']}（未设预期，不检查）"
            lines.append(f"- 来源判断{source_note}")
            if t["expect_keywords"]:
                kw_status = "全部命中" if t["content_passed"] else f"缺少 {'、'.join(t['missing_keywords'])}"
                lines.append(f"- 内容关键词检查（应包含：{'、'.join(t['expect_keywords'])}）：{kw_status}")
            lines.append(f"- 这一轮结果：{'✅' if t['passed'] else '❌'}")
            lines.append("- 完整回复：")
            lines.append("  ```")
            for line in t["reply"].splitlines():
                lines.append(f"  {line}")
            lines.append("  ```")
            lines.append("")
        lines.append("")

    return "\n".join(lines)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    kb_dir = os.path.join(here, "..", "data", "knowledge_base")
    cases_path = os.path.join(here, "multiturn_test_cases.json")
    out_path = os.path.join(here, "..", "multiturn_test_results.md")

    print("正在加载知识库和检索模型……")
    kb_pipeline = MemorySystemPipeline(kb_dir)

    # 测试用的L1/L2数据用内存数据库，不落盘到 data/memory.db——测试对话
    # 不是你的真实使用记录，不应该跟真实的个人记忆数据混在一起。
    store = MemoryStore(db_path=":memory:")

    cases = load_test_cases(cases_path)
    print(f"共 {len(cases)} 段多轮对话场景，开始跑……")

    cases_results = [run_case(kb_pipeline, store, case) for case in cases]

    report = render_markdown(cases_results, kb_pipeline.dense.name, kb_pipeline.reranker.name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)

    n_pass = sum(1 for c in cases_results if c["passed"])
    print(f"\n完成：{n_pass}/{len(cases_results)} 段对话全部轮次通过预期。")
    print(f"完整报告已写到：{os.path.abspath(out_path)}")
    print("把这个文件同步到坚果云之后告诉Claude看这个文件就行，跟run_tests.py是同一套习惯。")


if __name__ == "__main__":
    main()
