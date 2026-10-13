# -*- coding: utf-8 -*-
"""
离线生成课题报告片段的"这段能回答的问题"（见kb_questions.py）。

    cd src && python build_kb_questions.py

只给缓存里还没有的片段生成（报告没改就什么都不做）；结果写进 data/kb_question_cache.json。
下次启动机器人/跑测试时会自动读缓存、写进片段，Chroma索引发现片段内容变了会自动重算这些片段的向量。
不用加载检索模型，只读文档、切片、调LLM。
"""

import os

import kb_sources
from bot_profile import OWNER_NAME
from kb_questions import CACHE_PATH, generate_missing
from llm_client import build_llm_client

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    client = build_llm_client()
    if not client.is_real:
        print("需要真实LLM才能生成。")
        return
    kb_dir = os.path.join(HERE, "..", "data", "knowledge_base")
    chunks = kb_sources.load_knowledge_sources(kb_dir, kb_sources.build_catalog(), attach_questions=False)
    n = generate_missing(chunks, client, OWNER_NAME)
    print(f"生成了{n}个片段的问题，缓存：{os.path.abspath(CACHE_PATH)}")


if __name__ == "__main__":
    main()
