# -*- coding: utf-8 -*-
"""
"明日方舟万事通"进阶目标的小范围原型验证脚本 —— 跟其它 run_*_demo.py 一样，
人工核对用，不是自动判分的测试套件。

【这个原型验证什么】
验证这条链路能不能走通："问一个明日方舟相关的问题 -> 实时查prts.wiki搜到
相关页面 -> 去掉几个固定的低价值大块小节 -> 把（尽量完整的）正文交给LLM
生成一段带引用的自然语言回答"。

【为什么不做切分+精排——这是从三轮真机验证改回来的】
最早两轮尝试过"切分成候选片段+精排选最相关的几条"（复用阶段一的
chunking.py/reranker.py），但连续两轮真机验证都暴露问题：精排会把"干员
档案"背景故事里顺嘴提到的"天赋"，跟"天赋"技术小节里真正的技能效果说明
搞混；加了规则修正之后，又把"职业"相关的候选片段挤出了结果，而且即使
"天赋"本身答对了，回答也丢失了"潜能提升""模组"这些分散在别的小节、没有
被问题字面提到、但整页阅读时LLM会自己关联起来的加成信息。

三轮真机验证里，唯一一次职业和天赋都答对、且回答组织得有条理的，反而是
最早那次完全不切分、把整页正文直接交给LLM的版本。根本原因：切分+精排是
为"从一大堆文档里挑最相关的几条"设计的（阶段一的181个论文片段），但这里
是"已经确定就是这一个页面"，页面本身没有大到超出LLM一次调用的舒适区间，
用检索去在单一页面内部"挑重点"，反而丢掉了LLM自己阅读连续上下文做综合
判断的能力——继续在这个方向上"发现一个反例就加一条规则"，只会越改越
复杂、覆盖面却上不去。完整的踩坑过程见`prts_wiki_client.py`模块开头的
"架构反思"。

现在的做法：`prts_wiki_client.py`的`fetch_page_for_answer()`只做一件轻量
的事——去掉几个固定的、体量大但几乎不含事实性游戏机制信息的小节（语音
记录/干员密录/悖论模拟/干员模型），控制一下prompt长度，其余部分原样、
连续地交给LLM，不做切分/精排/候选打分。

复用 llm_generate.py 的 llm_generate_answer()，而不是重新写一遍生成逻辑
——它只要求传入的对象有 `.chunk.source_file` / `.chunk.text` 这两个字段，
这里把整段正文当成唯一一条"证据片段"包装一下就能直接复用。

【必须在你自己电脑上跑】
prts_wiki_client.py 需要真实访问 prts.wiki，云端沙盒/远程设备通道都连不上，
这个脚本也一样。

用法：
    cd src
    python run_prts_wiki_demo.py
"""

import os
from dataclasses import dataclass

from prts_wiki_client import fetch_page_for_answer
from llm_generate import llm_generate_answer
from llm_client import build_llm_client

_log_lines = []


def log(msg: str = ""):
    print(msg)
    _log_lines.append(msg)


@dataclass
class _WikiChunk:
    source_file: str
    text: str


@dataclass
class _WikiPassage:
    chunk: _WikiChunk


# 小范围原型先只测一个问题，跑通了再考虑加更多场景。这个问题故意问了两件
# 事（职业+天赋），是三轮真机调试里一直在用的同一个问题，方便直接对比。
_TEST_QUESTION = "能天使是什么职业，天赋是什么？"
_SEARCH_KEYWORD = "能天使"


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(here, "..", "prts_wiki_demo_results.md")

    llm_client = build_llm_client()
    log(f"当前LLM后端：{llm_client.name}（is_real={llm_client.is_real}）")
    if not llm_client.is_real:
        log("警告：LLM不可用，即使prts.wiki查询成功，也拿不到生成结果——"
            "先确认 .env 里的 MINIMAX_API_KEY 配好了。")
    log("")

    log("=" * 70)
    log(f"第一步：实时查询 prts.wiki（关键词：{_SEARCH_KEYWORD}）")
    log("=" * 70)
    result = fetch_page_for_answer(_SEARCH_KEYWORD)

    if result is None:
        log("\n查询失败——如果是网络问题，确认当前设备能直接打开 prts.wiki；"
            "如果是解析报错，看上面打印的报错信息，可能是接口返回结构跟"
            "prts_wiki_client.py里假设的不一样，需要照实际返回结构调整。")
    else:
        log(f"\n搜到页面：{result['title']}")
        log(f"正文长度：{len(result['text'])} 字符（已排除语音记录/干员密录/悖论模拟/干员模型这几个固定小节）")
        log(f"正文前500字预览：\n{result['text'][:500]}")

        log("\n" + "=" * 70)
        log("第二步：把（去掉低价值小节后的）整页正文交给LLM生成回答")
        log("=" * 70)
        passage = _WikiPassage(_WikiChunk(source_file=f"prts.wiki/{result['title']}", text=result["text"]))
        log(f"\n你问：{_TEST_QUESTION}")
        answer = llm_generate_answer(_TEST_QUESTION, [passage], llm_client)
        if answer is None:
            log("生成失败（LLM不可用，或者调用报错）——看上面llm_client打印的报错信息。")
        else:
            log(f"\nLLM回答：\n{answer}")

    log("\n" + "=" * 70)
    log("看完above结果，重点检查：")
    log("1. 第一步搜到的页面标题对不对（关键词有没有搜歪）。")
    log("2. 职业和天赋是不是两个都答对了（这个问题故意问了两件事，之前切分+"
        "精排的版本会顾此失彼），回答组织得是不是比之前分块的版本更有条理"
        "（比如有没有分别提到第一/第二天赋、潜能和模组的加成）。")
    log("3. LLM回答内容有没有带[1]这种引用标注，有没有编造正文里没有的信息。")
    log("=" * 70)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# 明日方舟万事通原型验证结果（小范围原型，非自动化测试，人工核对）\n\n```\n")
        f.write("\n".join(_log_lines))
        f.write("\n```\n")
    print(f"\n完整输出已写到：{os.path.abspath(out_path)}")
