# -*- coding: utf-8 -*-
"""L0滚动摘要接入编排层背景（⑳）的端到端人工验证脚本。

【这次要验证什么】
起因见`conversation_memory.py`模块开头"⑳L0摘要接入编排层背景"一节：
`_maybe_update_l0_summary()`一直在往`memory_store.py`的`l0_summary`表里写
东西，但`decompose()`/`route_query()`这些真正靠"L0窗口+摘要"理解代词/
省略指代的判断，实际上只拿到过`get_context_window()`（最近若干轮原文），
从来没读过这份摘要——`_build_planner_context()`现在改成同一个通道顺手
把摘要也带上，标成"仅用于理解指代、不构成证据"，跟`recent_context`里的
普通历史待遇一致。

【这次要验证的具体行为】
1. 制造一个"信息只留在摘要里、已经滑出L0原始窗口"的场景：先说一句提到
   一个不常见代号的陈述（"烛龙"这条分析流程），再用几句不相关的闲聊把
   窗口挤满，让这句话滑出`window_size`；重点看终端有没有打印
   "[conversation_memory] L0摘要接入编排层背景：..."这行日志、内容里
   有没有覆盖"烛龙"这个关键词——这行日志出现，就说明新的注入机制真的被
   触发了。
2. 之后用一句纯代词提问（"它现在进展到哪一步了？"，故意不重复"烛龙"这
   个词本身），看最终回答有没有正确理解"它"指的是"烛龙"这条流程、进而
   查到第一句话里的内容——如果编排层解不出这个代词，大概率会诚实回复
   "没有找到相关的历史记录"或者答非所问。**如果没查到，不一定是这次改动
   没生效**：也可能是`decompose()`/`route_query()`本身这次判断没有稳定
   命中（这类LLM判断问法敏感有一个消除不到零的残余率，见README"贯穿
   全局的防幻觉设计"一节同类经验），需要结合终端日志一起看，不能只看
   最终回答。
3. 回归检查："能天使是什么职业？"——这条问题走wiki路由的精确标题匹配，
   完全不涉及L0摘要/画像，行为应该跟改动前完全一致，确认这次改动没有
   引入意外的副作用。

【关于"烛龙"这句话本身会不会被L1判成陈述句、写进画像】
这句话说的是"我们实验室"启动了一个新流程，不是"我"的个人属性，预期会
被`_l1_step()`判成`kind="other"`（走编排层，不会被画像短路接住）——但
这是LLM语义分类，不是规则匹配，不能保证100%如此。脚本会在这句话说完
之后打印一次当前画像内容，如果"烛龙"这个关键词意外出现在画像里，说明
这次测试没能把"只能靠摘要才看得到"这个条件隔离干净，需要结合日志重新
判断，而不是直接归为这次改动的bug——`_build_planner_context()`里画像
和摘要是两条独立生效的通路，画像命中不代表摘要通路本身有问题。

【必须在你自己电脑上跑】
需要真实LLM（MiniMax）——`_maybe_update_l0_summary()`本身的滚动摘要
生成、以及`decompose()`/`route_query()`理解注入进去的摘要内容，都需要
真实LLM才会生效。

用法：
    cd src
    python run_l0_summary_context_demo.py
"""

import os

from conversation_memory import ConversationSession
from llm_client import build_llm_client
from memory_store import MemoryStore
from pipeline import MemorySystemPipeline

_log_lines = []


def log(msg: str = ""):
    print(msg)
    _log_lines.append(msg)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    kb_dir = os.path.join(here, "..", "data", "knowledge_base")
    out_path = os.path.join(here, "..", "l0_summary_context_demo_results.md")
    user_id = "demo_user"
    window_size = 4  # 故意调小，只需要少数几轮闲聊就能把测试句挤出窗口

    llm_client = build_llm_client()
    log(f"当前LLM后端：{llm_client.name}（is_real={llm_client.is_real}）")
    if not llm_client.is_real:
        log("警告：LLM不可用——L0滚动摘要的生成、以及编排层理解注入的摘要"
            "内容，都要在真实LLM可用时才会生效，先确认.env里的"
            "MINIMAX_API_KEY配好了。")
    log("")

    kb_pipeline = MemorySystemPipeline(kb_dir)
    store = MemoryStore(db_path=":memory:")  # 用干净的临时数据库，精确控制窗口状态
    session = ConversationSession(
        kb_pipeline, store, user_id=user_id, window_size=window_size, llm_client=llm_client
    )
    log(f"（这次演示把window_size故意调小成{window_size}，只是为了少打几轮"
        f"闲聊就能把测试句挤出L0窗口，不代表生产环境要改这个值）")
    log("")

    # ---- 第一步：说一句提到不常见代号的陈述，之后应该只留在摘要里 ----
    zhulong_statement = (
        "我们实验室这周启动了一个新的分析流程，内部代号叫「烛龙」，"
        "专门处理超长读长测序数据的组装。"
    )
    log("=" * 70)
    log(f"你说：{zhulong_statement}")
    log("-" * 70)
    reply = session.user_says(zhulong_statement)
    log(reply)
    log("")

    profile_after_first = store.get_profile(user_id)
    log(f"【核对】这句话之后的画像内容：{profile_after_first or '（空）'}")
    if profile_after_first and any("烛龙" in str(v) for v in profile_after_first.values()):
        log("【提示】烛龙这个关键词出现在了画像里——见脚本docstring最后一节"
            "的说明，不直接算这次改动的bug，但意味着接下来的代词提问可能是"
            "靠画像通路答对的，不能单独归功于摘要通路，需要结合下面的终端"
            "日志一起判断。")
    else:
        log("【核对通过】画像里没有出现烛龙，这句话确实只会靠L2原始历史"
            "+（滑出窗口之后）L0摘要才能被找回来。")
    log("")

    # ---- 第二步：几句不相关的闲聊，把第一句话挤出L0窗口 ----
    filler_turns = [
        "今天天气有点冷，中午吃了碗泡面。",
        "明天组会得早起，有点犯困。",
        "对了，实验室的咖啡机好像又坏了。",
    ]
    for turn in filler_turns:
        log("=" * 70)
        log(f"你说：{turn}（闲聊填充，用来把上一句挤出L0窗口）")
        log("-" * 70)
        reply = session.user_says(turn)
        log(reply)
        log("")

    recent_window = session.get_context_window()
    window_texts = "；".join(r["text"] for r in recent_window)
    still_in_window = "烛龙" in window_texts
    log(f"【核对】当前L0窗口（最近{window_size}行）内容：{window_texts}")
    if still_in_window:
        log("【核对】烛龙还在窗口里——说明还没挤出去，下面的代词提问不能"
            "验证摘要通路，需要再加几句闲聊。")
    else:
        log("【核对】烛龙已经不在窗口里——已经挤出窗口，符合预期。")
    log("")

    current_summary = session.get_session_summary()
    if current_summary is None:
        log("【核对】当前L0滚动摘要内容：（还没有生成摘要——如果上面窗口"
            "已经把烛龙挤出去了，这里却还是空的，说明_maybe_update_l0_"
            "summary()这次没有成功刷新，见该方法文档字符串"
            "「失败时怎么处理」一节，可能是这次LLM调用恰好失败，"
            "不一定是改动本身有问题）")
    else:
        log(f"【核对】当前L0滚动摘要内容：{current_summary}")
        if "烛龙" in current_summary:
            log("【核对】摘要覆盖了烛龙这个关键词——符合预期。")
        else:
            log("【核对】摘要没有覆盖烛龙这个关键词——不一定是bug，也可能是"
                "LLM滚动摘要时判断这条信息不值得保留，需要结合下一步的最终"
                "回答一起看。")
    log("")

    # ---- 第三步：纯代词提问，验证编排层能不能借助摘要解出指代 ----
    log("=" * 70)
    log("你说：它现在进展到哪一步了？")
    log("【这句话要验证】故意不重复烛龙这个词本身，只用「它」——"
        "如果编排层借助刚才打印的摘要内容正确理解「它」指的是烛龙"
        "这条流程，应该能查到第一句话的内容并给出相关回答；如果解不出这个"
        "代词，大概率会诚实回复没有找到相关记录，或者答非所问。留意终端"
        "有没有打印「[conversation_memory] L0摘要接入编排层背景：..."
        "」这行日志、内容对不对，这行日志本身就是这次改动生效与否的"
        "直接证据，比只看最终回答的措辞更可靠。")
    log("-" * 70)
    reply = session.user_says("它现在进展到哪一步了？")
    log(reply)
    log("")

    # ---- 第四步：回归检查 ----
    log("=" * 70)
    log("你说：能天使是什么职业？")
    log("【这句话要验证】回归检查：这条问题走wiki路由的精确标题匹配，完全"
        "不涉及L0摘要/画像，行为应该跟改动前完全一致。")
    log("-" * 70)
    reply = session.user_says("能天使是什么职业？")
    log(reply)
    log("")

    log("=" * 70)
    log("整体重点检查：")
    log("1. 说完烛龙那句话之后，画像里有没有意外出现这个关键词（见上面第")
    log("   一步的【核对】）——没有的话，后面的代词提问才是干净的摘要通路")
    log("   测试。")
    log("2. 三句闲聊之后，终端有没有打印")
    log("   [conversation_memory] L0摘要接入编排层背景：...  这行日志、")
    log("   内容里有没有正确覆盖烛龙这个关键词。")
    log("3. 它现在进展到哪一步了这句话，最终回答有没有正确关联上烛龙这条")
    log("   流程；如果没有，结合第2点的日志判断是摘要没被注入，还是注入")
    log("   了但decompose()/route_query()这次判断没能正确利用它——后一种")
    log("   情况可能只是问法敏感的正常残余率，不一定是这次改动的bug。")
    log("4. 能天使是什么职业这条回归检查，回答是否跟改动前一致。")
    log("5. 把这几条回答和终端日志一起发回来，一起判断这次要不要往前继续")
    log("   投入，还是可以确认这次改动生效了。")
    log("=" * 70)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# L0滚动摘要接入编排层背景（⑳）——端到端验证结果（人工核对）\n\n```\n")
        f.write("\n".join(_log_lines))
        f.write("\n```\n")
    print(f"\n完整输出已写到：{os.path.abspath(out_path)}")
