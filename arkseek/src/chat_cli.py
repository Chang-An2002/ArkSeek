# -*- coding: utf-8 -*-
"""
多轮对话命令行界面 —— 阶段二入口（默认2.2接LLM，可用--no-llm退回2.1）

跟 cli.py 的关系：cli.py 保留不动，继续代表"阶段一：不接记忆、单轮问答"这个
基线版本，方便随时对比"加了记忆层之后差在哪"；本文件是阶段二的交互式入口，
两个文件都能跑，互不影响，也不共用状态。

跟 cli.py 最大的不同：这里的问答会互相影响——先说"我叫XX"，后面问"我是谁"
能答出来（L1）；提到过的内容，后面能翻回去（L2）；见 conversation_memory.py
的完整设计说明。

【默认接入真实LLM（2.2）】
默认会调用`build_llm_client()`（读取.env里的MINIMAX_API_KEY等配置），把
真实LLM传给`ConversationSession`——这样除了本地知识库，还会多一条"明日
方舟wiki实时查询"的路径、翻旧账问题会先经过统一分流判断再去调用L2、以及
寒暄客套话的直接回复（见`wiki_router.py`/`conversation_memory.py`的
`_dispatch_with_router()`），
生成的回答也会是LLM整理过的自然语言总结，而不是阶段2.1那种"罗列证据来源"
的形式。如果.env没配好，`build_llm_client()`会自动退回`NoLLMClient()`，
这时候效果等同于加了--no-llm——脚本不会报错，只是wiki路径和LLM生成都不会
生效，回答形式退回2.1的样子，同时会打印一行提示。

想显式对比"接LLM前后差在哪"、或者单纯不想消耗LLM额度，可以加--no-llm
强制跑2.1模式（不构造llm_client，跟本文件历史行为完全一致）。

默认把 L1/L2 数据落盘到 data/memory.db（不是内存数据库）——退出命令行再
重新打开，机器人依然"记得你"，这才是"记忆系统"名副其实的地方，不是每次
运行都从空白状态开始。想清空重新开始，加 --reset 参数。

用法：
    cd src
    python chat_cli.py            # 默认：接真实LLM，wiki路径生效（需要.env配好key）
    python chat_cli.py --no-llm   # 强制阶段2.1，不接LLM，方便对比
    python chat_cli.py --reset    # 清空当前用户的L1/L2记忆，从头开始

对话过程中输入 /summary 可以随时查看当前L0滚动摘要的实际内容——2026-09
真机排查"精简重试"那次改动时，用户问过"如果想看看压缩的摘要是什么样子，
应该怎么做"，之前完全没有暴露这个入口，只能自己写脚本查`data/memory.db`
里的`l0_summary`表，加上这个命令之后可以直接在对话里看，不用跳出去。

同理，输入 /profile 可以随时查看当前L1用户画像的完整内容——之前只有启动
时"记得你之前告诉过我：xxx"这一次性提示（还只在启动那一刻有旧画像时才会
打，中途新记住的字段看不到），跟`/summary`一样补齐这个入口，两个调试命令
风格保持一致。

输入 /status 可以随时查看当前实际在用的各个后端（向量检索/对话历史检索/
精排/LLM）以及本次运行发生过的所有降级——接入Chroma时用户明确要求"保留
自动退回，但保证退回时用户能够知晓"：降级提示在启动时打印一次容易被后面
几十行加载日志淹没，所以加载完成后如果发生过降级会再汇总提示一遍，之后
也可以随时用 /status 查看。
"""

import argparse
import os
import sys

from pipeline import MemorySystemPipeline
from memory_store import MemoryStore
from conversation_memory import ConversationSession
from llm_client import build_llm_client
from chroma_index import DEGRADE_NOTICES
from bot_profile import intro_message

# 目前只有一个真实用户（你），先写死一个默认user_id；user_id这个设计本身
# 是为"以后要不要支持多用户"留的口子，不是当前就要用到的功能。
DEFAULT_USER_ID = "default_user"


def _format_status(kb_pipeline, session, llm_client) -> str:
    history_index = session._history_index
    if history_index is not None:
        l2 = f"Chroma（{history_index.location}）"
    elif session._embedding_model is not None:
        l2 = "内存现算BGE向量（降级：Chroma不可用，每次翻旧账都重新计算全部历史的向量）"
    else:
        l2 = "内存现算TF-IDF（降级：没有真实embedding模型）"
    llm_name = llm_client.name if llm_client is not None else "未接入（--no-llm，阶段2.1）"
    ws = session.web_search
    if ws is None:
        web = "未启用（没接LLM时用不上）"
    elif ws.is_real:
        web = f"{ws.name}（本次运行已调用{getattr(ws, 'calls', 0)}次，明细见data/web_search_log.jsonl）"
    else:
        web = ws.name
    lines = [
        "（当前在用的后端）",
        f"  知识库向量检索（L3）：{kb_pipeline.dense.name}",
        f"  对话历史向量检索（L2）：{l2}",
        f"  精排：{kb_pipeline.reranker.name}",
        f"  LLM：{llm_name}",
        f"  联网搜索：{web}",
    ]
    if DEGRADE_NOTICES:
        lines.append(f"（本次运行发生过{len(DEGRADE_NOTICES)}次降级）")
        lines += [f"  - {n}" for n in DEGRADE_NOTICES]
    else:
        lines.append("（本次运行没有发生过降级）")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="记忆系统多轮对话命令行（默认2.2接LLM，--no-llm可退回2.1）")
    parser.add_argument("--reset", action="store_true", help="开始前清空这个用户的L1/L2记忆，从头开始")
    parser.add_argument("--user-id", default=DEFAULT_USER_ID, help="用户ID，默认单用户场景不用改")
    parser.add_argument("--no-llm", action="store_true",
                         help="强制不接LLM，跑阶段2.1效果（本地知识库+证据罗列，wiki路径不生效）")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    kb_dir = os.path.join(here, "..", "data", "knowledge_base")

    print("正在加载记忆系统知识库……")
    kb_pipeline = MemorySystemPipeline(kb_dir)
    store = MemoryStore()  # 默认落盘到 data/memory.db，退出重进依然记得你

    if args.reset:
        store.reset_user(args.user_id)
        print(f"已清空用户「{args.user_id}」的L1/L2记忆，从头开始。")

    if args.no_llm:
        llm_client = None  # ConversationSession内部会用NoLLMClient()
        print("（已用--no-llm强制关闭LLM：阶段2.1效果，wiki路径不会生效）")
    else:
        llm_client = build_llm_client()
        print(f"当前LLM后端：{llm_client.name}（is_real={llm_client.is_real}）")
        if not llm_client.is_real:
            print("（.env里没有配好可用的LLM key，已自动退回阶段2.1效果："
                  "本地知识库正常，但wiki路径和LLM生成都不会生效——如果想测试"
                  "'明日方舟万事通'，检查一下MINIMAX_API_KEY配置）")

    session = ConversationSession(kb_pipeline, store, user_id=args.user_id, llm_client=llm_client,
                                  auto_intro=True)

    print("\n加载完成。输入问题回车即可提问，输入 q 或 exit 退出，输入 /summary "
          "可以随时查看当前L0滚动摘要的内容，输入 /profile 可以随时查看当前L1"
          "用户画像的内容，输入 /status 可以随时查看当前在用的检索/LLM后端，"
          "输入 /help 可以查看机器人的自我介绍（第一次对话时会自动说一次）。")
    if DEGRADE_NOTICES:
        print(f"\n注意：本次启动有{len(DEGRADE_NOTICES)}项功能降级运行（输入 /status 查看详情）：")
        for notice in DEGRADE_NOTICES:
            print(f"  - {notice}")
    if llm_client is not None and llm_client.is_real:
        print("（已接入真实LLM：本地知识库问题正常回答，明日方舟相关问题会自动"
              "路由去查prts.wiki——直接问「能天使是什么职业」之类的问题试试）")
    else:
        print("（阶段二2.1效果：不接生成模型，在阶段一基础上多了L1用户画像+L2历史检索"
              "两层记忆；回答形式还是展示证据来源，不是一句话自然语言总结）")

    existing_profile = store.get_profile(args.user_id)
    if existing_profile:
        print(f"（记得你之前告诉过我：{existing_profile}）")
    print()

    while True:
        try:
            query = input("你说 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见，下次见我还记得你。")
            sys.exit(0)

        if query.lower() in ("q", "exit", "quit"):
            print("再见，下次见我还记得你。")
            break
        if not query:
            continue

        if query == "/summary":
            # 纯调试用命令，不算一轮真正的对话，不会存进历史/触发L1判断。
            all_turns = store.get_all_turns(args.user_id)
            existing = store.get_l0_summary(args.user_id)
            print()
            print(f"（当前累计对话记录数：{len(all_turns)}，L0窗口大小：{session.window_size}）")
            if existing is None:
                print("（还没有生成过L0摘要——要么还没超过窗口大小，要么之前的尝试"
                      "一直没成功过，摘要保持初始的\"没有\"状态。）")
            else:
                print(f"（当前L0摘要，已并入到第{existing['summarized_up_to']}条记录为止）：")
                print(existing["summary"] or "（空字符串——上一次生成时判定新对话里没有值得记住的信息）")
            print()
            continue

        if query == "/help":
            # 开场白（bot_profile.py），跟第一次对话时自动说的是同一段
            print()
            print(intro_message(session.llm_client.is_real))
            print()
            continue

        if query == "/status":
            # 跟/summary、/profile一样是纯调试命令，不算一轮真正的对话。
            print()
            print(_format_status(kb_pipeline, session, llm_client))
            print()
            continue

        if query == "/profile":
            # 跟/summary同样是纯调试命令，不算一轮真正的对话。
            profile = store.get_profile(args.user_id)
            print()
            if not profile:
                print("（当前L1用户画像是空的——还没有记住任何关于你的信息。）")
            else:
                print(f"（当前L1用户画像，共{len(profile)}个字段）：")
                for key, value in profile.items():
                    print(f"  {key}：{value}")
            print()
            continue

        print()
        print(session.user_says(query))
        print()


if __name__ == "__main__":
    main()
