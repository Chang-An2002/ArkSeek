# -*- coding: utf-8 -*-
"""
命令行问答界面

对应方案文档 6.1 节"阶段一（核心逻辑验证，不接 IM）"里说的"命令行/网页问答界面"。
先做命令行版本，跑通逻辑；网页版可以后面用 Streamlit/Gradio 快速套一层，
核心逻辑（pipeline.py）不用改。
"""

import os
import sys

from pipeline import MemorySystemPipeline


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    kb_dir = os.path.join(here, "..", "data", "knowledge_base")

    print("正在加载记忆系统知识库……")
    pipeline = MemorySystemPipeline(kb_dir)

    print("\n加载完成。输入问题回车即可提问，输入 q 或 exit 退出。\n"
          "（当前是阶段一：只做检索+可靠性判断，不接生成模型，看到的是原始召回片段）\n")

    while True:
        try:
            query = input("你问 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            sys.exit(0)

        if query.lower() in ("q", "exit", "quit"):
            print("再见。")
            break
        if not query:
            continue

        print()
        print(pipeline.answer(query))
        print()


if __name__ == "__main__":
    main()
