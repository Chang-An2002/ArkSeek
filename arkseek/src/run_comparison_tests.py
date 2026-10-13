# -*- coding: utf-8 -*-
"""
2.1（不接LLM）vs 2.2（接LLM）正式批量对比测试 —— 当初拆分2.1/2.2的目的

跟 run_tests.py / run_multiturn_tests.py 是同一套"批量跑+写成Markdown报告，
不用再手动复制粘贴终端输出"的习惯，区别是这次同一批`multiturn_test_cases.json`
场景要跑两遍——一遍用2.1（`llm_client=None`，`ConversationSession`内部自动
退回`NoLLMClient`），一遍用2.2（`build_llm_client()`拿到的真实LLM客户端）——
然后把两边的结果并排放进同一份报告，而不是分别生成两份报告让你自己对着看。

【为什么直接复用 multiturn_test_cases.json，而不是给2.2单独写一套用例】
这批用例本来就是照着"L0/L1/L2/L3各自要验证什么"设计的，架构层面2.1/2.2完全
一样（四层、优先级、置信度gate都没变），2.2只是换了"L1怎么判断""L2/L3怎么
组织回答"这两处实现——所以同一批用例对两边都适用，差异会体现在"回复内容
长什么样"上，不需要另起一套测试设计。

【`expect_source`/`expect_keywords` 对2.2还成立吗——成立，但判断方式要放宽】
`expect_source`（来源分类）不受影响：`classify_source()`已经改过，L2/L3的
2.1证据格式和2.2生成格式共用同一个分类结果，只看"这轮回复来自哪一层"，不
关心具体措辞。`expect_keywords`（内容关键词）对2.2也基本成立——2.2的L1合并
结果、L2/L3生成的自然语言，只要真的用到了检索出的信息，该提到的关键词通常
还是会提到；但2.2是概率性的语义理解和生成，不像2.1的正则/证据罗列是确定性
的，个别轮次即使架构上"答对了"，用词也可能没有精确命中`expect_keywords`
里那个词——这种不匹配不能直接当成"2.2变差了"，需要打开报告里的完整回复
文本自己判断，这也是这份报告要把两边的完整回复都摆出来、而不是只给一个
通过/不通过汇总数字的原因。

【2.2那一遍如果LLM不可用会发生什么】
`ConversationSession`内部会自动退回`NoLLMClient`对应的2.1规则版逻辑——也就是
说，云端沙盒（没有真实API key/网络）下跑这个脚本，"2.1"和"2.2"两列会是完全
一样的结果，报告本身仍然能生成、不会报错，只是看不出任何差异，这是预期
行为，不是bug。要看到真实的2.1/2.2差异，必须在配置好真实MINIMAX_API_KEY的
环境（你自己电脑）上跑。

用法：
    cd src
    python run_comparison_tests.py
"""

import os
from datetime import datetime

from pipeline import MemorySystemPipeline
from memory_store import MemoryStore
from llm_client import build_llm_client
from run_multiturn_tests import load_test_cases, run_case


def render_comparison_markdown(cases_21, cases_22, dense_name: str, reranker_name: str,
                                llm_name: str, llm_is_real: bool) -> str:
    lines = []
    lines.append("# 2.1（不接LLM） vs 2.2（接LLM）批量对比测试报告\n")
    lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"当前向量检索后端：**{dense_name}**\n")
    lines.append(f"当前精排后端：**{reranker_name}**\n")
    lines.append(f"当前LLM后端：**{llm_name}**（is_real={llm_is_real}）\n")
    if not llm_is_real:
        lines.append("\n> ⚠️ 当前LLM不可用，下面「2.2」这一列会自动退回2.1的规则版行为，"
                      "跟「2.1」列完全一样——这是预期行为，要看到真实差异需要在配置好"
                      "真实API key的环境下重跑这个脚本。\n")

    n_pass_21 = sum(1 for c in cases_21 if c["passed"])
    n_pass_22 = sum(1 for c in cases_22 if c["passed"])
    lines.append(f"\n## 汇总：2.1 {n_pass_21}/{len(cases_21)} 通过预期，"
                  f"2.2 {n_pass_22}/{len(cases_22)} 通过预期\n")
    lines.append("（`expect_source`/`expect_keywords` 是照着2.1的行为设计的判断标准，"
                  "2.2在这份判断标准之外还可能有质的提升或有意的行为差异——比如L1的"
                  "合并vs覆盖、L2/L3从展示证据变成生成自然语言——这些不一定会体现在"
                  "通过率数字上，请结合下面每一轮的完整回复文本判断，不要只看这两个"
                  "数字。）\n")
    lines.append("| 对话场景 | 2.1 | 2.2 |")
    lines.append("|---|---|---|")
    for c21, c22 in zip(cases_21, cases_22):
        mark21 = "✅" if c21["passed"] else "❌"
        mark22 = "✅" if c22["passed"] else "❌"
        lines.append(f"| {c21['name']} | {mark21} | {mark22} |")

    lines.append("\n## 逐条对比\n")
    for c21, c22 in zip(cases_21, cases_22):
        lines.append(f"### {c21['name']}（user_id={c21['user_id']}）")
        lines.append(f"- 2.1综合结果：{'✅ 通过' if c21['passed'] else '❌ 未通过'}"
                      f" ｜ 2.2综合结果：{'✅ 通过' if c22['passed'] else '❌ 未通过'}\n")

        for i, (t21, t22) in enumerate(zip(c21["turns"], c22["turns"]), start=1):
            lines.append(f"**第{i}轮 你说：** {t21['say']}")
            lines.append(f"- 备注：{t21['note']}")

            for label, t in (("2.1（不接LLM）", t21), ("2.2（接LLM）", t22)):
                source_note = ""
                if t["expect_source"] is not None:
                    source_note = (f"，预期来源={t['expect_source']}，实际来源={t['actual_source']}"
                                    f"（{'匹配' if t['source_passed'] else '不匹配'}）")
                else:
                    source_note = f"，实际来源={t['actual_source']}（未设预期，不检查）"
                kw_note = ""
                if t["expect_keywords"]:
                    kw_status = "全部命中" if t["content_passed"] else f"缺少 {'、'.join(t['missing_keywords'])}"
                    kw_note = f"；内容关键词检查（应包含：{'、'.join(t['expect_keywords'])}）：{kw_status}"
                lines.append(f"- **{label}**：{'✅' if t['passed'] else '❌'}{source_note}{kw_note}")
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
    out_path = os.path.join(here, "..", "comparison_test_results.md")

    print("正在加载知识库和检索模型……")
    kb_pipeline = MemorySystemPipeline(kb_dir)
    llm_client = build_llm_client()
    print(f"当前LLM后端：{llm_client.name}（is_real={llm_client.is_real}）")

    cases = load_test_cases(cases_path)
    print(f"共 {len(cases)} 段多轮对话场景，每段跑两遍（2.1 + 2.2）……")

    # 用内存数据库、且2.1/2.2各自独立的store——避免2.2那一遍的写入（比如L0摘要
    # 表）跟2.1那一遍的数据混在一起相互干扰。两遍分别调用 run_case()（内部会
    # reset_user，保证每段用例从空白开始）。
    store_21 = MemoryStore(db_path=":memory:")
    store_22 = MemoryStore(db_path=":memory:")

    print("  跑2.1（不接LLM）……")
    cases_21 = [run_case(kb_pipeline, store_21, case, llm_client=None) for case in cases]
    print("  跑2.2（接LLM）……")
    cases_22 = [run_case(kb_pipeline, store_22, case, llm_client=llm_client) for case in cases]

    report = render_comparison_markdown(
        cases_21, cases_22,
        kb_pipeline.dense.name, kb_pipeline.reranker.name,
        llm_client.name, llm_client.is_real,
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)

    n_pass_21 = sum(1 for c in cases_21 if c["passed"])
    n_pass_22 = sum(1 for c in cases_22 if c["passed"])
    print(f"\n完成：2.1 {n_pass_21}/{len(cases_21)}，2.2 {n_pass_22}/{len(cases_22)}")
    print(f"完整对比报告已写到：{os.path.abspath(out_path)}")
    print("把这个文件同步到坚果云之后告诉Claude看这个文件就行，跟其它批量测试脚本是同一套习惯。")


if __name__ == "__main__":
    main()
