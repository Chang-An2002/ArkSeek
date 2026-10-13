# -*- coding: utf-8 -*-
"""
稠密检索模块（Dense / Embedding Retrieval）

对应方案文档 2.2 节"稠密检索 / Embedding"。真正的生产做法是用 BGE-M3 之类的
预训练模型把文本转成语义向量。但当前云端环境访问不了 HuggingFace/PyPI，
装不上 sentence-transformers，没法下载预训练模型权重。

所以这里做了一个"可插拔后端"的设计（这本身也是一个值得在项目里讲的工程判断）：
- EmbeddingDenseRetriever：真正的语义向量版本，用 sentence-transformers +
  BAAI/bge-small-zh-v1.5（轻量中文向量模型）。在你自己电脑上装好依赖之后可以用这个。
- TfidfDenseRetriever：退化版本，用字符 n-gram 的 TF-IDF + 余弦相似度，
  不需要任何预训练模型、不需要联网，现在就能跑。它不是真正的"语义"检索
  （比如换个说法但意思一样的问法，效果会比真正的embedding差很多），
  但作为验证"两路召回+RRF融合"这套架构逻辑是否正确，已经够用。

pipeline.py 会自动尝试加载 EmbeddingDenseRetriever，失败就退回 TfidfDenseRetriever，
并打印一条提示，不需要手动改代码切换。

【接入Chroma之后：三级后端】
build_dense_retriever() 现在按这个顺序尝试：
  1. ChromaDenseRetriever（chroma_index.py）：BGE向量存进持久化的Chroma集合，
     启动时只重新嵌入新增/变化的知识库文件——正常情况走这一级
  2. EmbeddingDenseRetriever：chromadb装不上/初始化失败时退回，每次启动全量
     重新嵌入，检索结果跟第1级一致，只是慢
  3. TfidfDenseRetriever：连BGE模型都加载不了时退回，不是真正的语义检索
第2、3级都是"降级"，会通过 chroma_index.announce_degrade() 打印醒目提示并
记进 /status，.name 里也会写明"降级"，批量测试报告的"当前向量检索后端"
一行直接能看出来——不会悄悄降级。
"""

from typing import List, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from chunking import Chunk


# 知识库扩充加入英文文献后换成中英通用的bge-m3（原来的bge-small-zh-v1.5只懂中文，
# 英文文献和"lineage tracing"这类英文问法对不上）。换模型后Chroma里的旧向量会被
# 自动识别出来并重建（chroma_index.open_collection()按embed_model比对）。
DEFAULT_EMBED_MODEL = "BAAI/bge-m3"


class TfidfDenseRetriever:
    """用字符n-gram TF-IDF 模拟"稠密检索"，零依赖、零下载，随时能跑。"""

    name = "TF-IDF（占位版，不是真正的语义embedding）"

    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        # 中文没有天然的词边界，用字符2~4-gram的TF-IDF是常见的轻量替代方案
        self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4))
        texts = [c.text for c in chunks]
        self.doc_matrix = self.vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.doc_matrix)[0]
        ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


class EmbeddingDenseRetriever:
    """真正的语义向量检索，需要本地能装 sentence-transformers 并下载模型。"""

    name = "真实语义embedding"

    def __init__(self, chunks: List[Chunk], model_name: str = DEFAULT_EMBED_MODEL, model=None):
        """model：已经加载好的 SentenceTransformer 实例（build_dense_retriever()
        先加载一次、传进来，避免"Chroma初始化失败退回这一级"时再加载第二遍）；
        不传就自己加载。"""
        import numpy as np

        self.chunks = chunks
        self.model = model if model is not None else load_embedding_model(model_name)
        self.model_name = model_name
        self.name = f"{model_name}（真实语义embedding）"
        texts = [c.text for c in chunks]
        self.doc_embeddings = self.model.encode(texts, normalize_embeddings=True)
        self._np = np

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        query_emb = self.model.encode([query], normalize_embeddings=True)[0]
        sims = self.doc_embeddings @ query_emb  # 已归一化，点积即余弦相似度
        ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


def load_embedding_model(model_name: str = DEFAULT_EMBED_MODEL):
    import logging

    from sentence_transformers import SentenceTransformer  # 延迟导入，装不上时不影响其他模块

    # 真机测试时用户发现终端会打印一条HF Hub的"未认证请求"提示（"You are
    # sending unauthenticated requests to the HF Hub..."）——不影响功能
    # （只是不认证时下载速率限制更低），但每次加载模型都刷一遍，观感不好。
    # 这条是huggingface_hub库自己用logging模块打的日志（不是Python内置
    # warnings模块的警告），所以直接调高这个logger的级别，只让真正的
    # ERROR级别问题（比如模型下载彻底失败）继续冒出来——那些是靠抛异常
    # 被上层build_dense_retriever()的try/except接住的，不受这里影响，
    # 跟reranker.py那条activation_fct过期警告是同一类"库自身噪音"问题，
    # 处理思路也一致：只压噪音，不掩盖真正的错误。
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    return SentenceTransformer(model_name)


def build_dense_retriever(chunks: List[Chunk], chroma_dir: Optional[str] = None):
    """按 Chroma -> 内存版BGE -> TF-IDF 的顺序尝试，见模块开头"三级后端"。
    chroma_dir 为 None 时直接跳过Chroma这一级（不算降级，是调用方的选择）。"""
    from chroma_index import announce_degrade  # 纯Python模块，不依赖chromadb

    try:
        model = load_embedding_model(DEFAULT_EMBED_MODEL)
    except Exception as e:
        notice = announce_degrade(
            "向量检索", f"无法加载embedding模型 {DEFAULT_EMBED_MODEL}（{type(e).__name__}: {e}）",
            "退回TF-IDF占位版，不是真正的语义检索，检索质量明显变差",
        )
        retriever = TfidfDenseRetriever(chunks)
        retriever.name = f"{retriever.name}【降级：{notice}】"
        return retriever

    if chroma_dir is not None:
        try:
            from chroma_index import ChromaDenseRetriever
            retriever = ChromaDenseRetriever(chunks, model, DEFAULT_EMBED_MODEL, chroma_dir)
            print(f"[dense_retriever] 已加载：{retriever.name}")
            return retriever
        except Exception as e:
            notice = announce_degrade(
                "向量检索", f"Chroma不可用（{type(e).__name__}: {e}）",
                "退回内存版向量检索：结果不变，但每次启动都要重新计算全部知识库向量",
            )
            retriever = EmbeddingDenseRetriever(chunks, DEFAULT_EMBED_MODEL, model=model)
            retriever.name = f"{retriever.name}【降级：{notice}】"
            return retriever

    retriever = EmbeddingDenseRetriever(chunks, DEFAULT_EMBED_MODEL, model=model)
    print(f"[dense_retriever] 已加载真实语义模型：{retriever.name}")
    return retriever


if __name__ == "__main__":
    import os
    from chunking import load_knowledge_base

    here = os.path.dirname(os.path.abspath(__file__))
    kb_dir = os.path.join(here, "..", "data", "knowledge_base")
    chunks = load_knowledge_base(kb_dir)

    retriever = build_dense_retriever(chunks)
    query = "多种分子层面的数据怎么放到一起分析"
    results = retriever.search(query, top_k=3)

    print(f"\n查询：{query}\n")
    for idx, s in results:
        print(f"score={s:.3f}  ({chunks[idx].source_file})  {chunks[idx].text[:50]}...")
