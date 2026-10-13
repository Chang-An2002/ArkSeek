# -*- coding: utf-8 -*-
"""
稀疏检索模块（BM25）

对应方案文档 2.2 节"稀疏检索 / BM25"。这里手写了一个标准 BM25（Okapi BM25）的实现，
没有依赖第三方的 rank_bm25 库（当前环境访问不了 PyPI，所以干脆自己写一遍——
好处是能在文档/面试里把公式和每一步都讲清楚，而不是"调了个库"）。

中文分词说明：正常做法是用 jieba 做中文分词，但同样因为装不上第三方库，
这里退而求其次，用了一个轻量替代方案——对中文字符做"字符二元组（bigram）"切分，
对英文/数字按单词切分。这不是最优的分词方式，但足够验证 BM25 的检索逻辑；
等你本地环境能装 jieba 之后，把 tokenize() 换成 jieba.lcut() 效果会更好。

【踩坑记录】最早的版本里，除了二元组还额外保留了单字token（比如"什么"会拆出"什"
"么"两个单字），本意是防止查询词太短匹配不上，结果在真实语料（181个片段的论文全文）
上测试时，一个完全不相关的问题（"米哈游的游戏运营岗位需要什么能力？"）居然也召回了
结果——原因是"什""么"这两个单字、以及"什么"这个二元组，在这篇几万字的论文里恰好
只出现了一两次，BM25的IDF公式会把"在语料里很少见"直接当成"很重要"，于是这几个
其实毫无信息量的常见字/词被当成了强匹配信号。这是IDF"语料里稀有"和"语言里稀有"
不是一回事的一个典型例子。修复方式：1) 去掉单字token，中文最少以二元组为单位；
2) 加一份停用词表，把"的""是""什么"这类不管在哪篇中文文档里都高频出现的功能词
过滤掉，不参与匹配。
"""

import re
import math
from collections import Counter
from typing import List, Tuple

from chunking import Chunk


_CJK_PATTERN = re.compile(r"[一-鿿]")
_WORD_PATTERN = re.compile(r"[A-Za-z0-9]+")

# 常见中文功能词/停用词（不是完整列表，够覆盖踩过的坑和最高频的场景即可；
# 真要严谨可以换成 jieba 自带的停用词表）
_STOPWORDS = {
    "的", "了", "是", "在", "和", "与", "及", "或", "也", "都", "就",
    "而", "之", "其", "对于", "通过", "我们", "本文", "本课题", "这个",
    "那个", "什么", "怎么", "如何", "可以", "以及", "以", "为", "被",
    "把", "让", "使", "有", "无", "不", "很", "非常", "以及", "但",
    "但是", "因为", "所以", "如果", "这些", "那些", "一个", "一种",
    "进行", "具有", "存在", "属于", "由于", "由", "对", "从", "到",
    "上", "下", "中", "内", "外", "等", "着", "过", "呢", "吗", "啊",
}


# 知识库加入英文文献后补的英文停用词：虚词在英文论文里极高频，但对"问的是什么"
# 没有信息量（IDF会把它们压低，但压不到零，长文献里仍然会稀释真正的关键词）
_EN_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "in", "on", "at", "to", "for", "with", "by", "from", "as",
    "is", "are", "was", "were", "be", "been", "being", "this", "that", "these", "those", "it", "its",
    "we", "our", "they", "their", "which", "what", "how", "who", "whom", "does", "do", "did", "can",
    "could", "would", "should", "may", "might", "not", "no", "but", "if", "than", "then", "into",
    "such", "also", "using", "used", "use", "based", "between", "both", "each", "other", "all",
}


def tokenize(text: str) -> List[str]:
    """分词：中文按字符二元组切（过滤停用词），英文/数字按单词切。"""
    tokens = []

    # 英文/数字单词
    tokens.extend(w for w in (w.lower() for w in _WORD_PATTERN.findall(text)) if w not in _EN_STOPWORDS)

    # 中文字符二元组（bigram），比如"基因组"-> ["基因", "因组"]
    # 注意：不再额外保留单字token——中文单字几乎从不构成有效检索单元，
    # 反而容易被IDF公式误判成"稀有=重要"（见上面的踩坑记录）
    cjk_chars = [ch for ch in text if _CJK_PATTERN.match(ch)]
    for i in range(len(cjk_chars) - 1):
        bigram = cjk_chars[i] + cjk_chars[i + 1]
        if bigram in _STOPWORDS:
            continue
        tokens.append(bigram)

    return tokens


class BM25Index:
    """标准 BM25（Okapi）实现，k1 和 b 用行业常见默认值。"""

    def __init__(self, chunks: List[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b

        self.doc_tokens: List[List[str]] = [tokenize(c.text) for c in chunks]
        self.doc_len = [len(toks) for toks in self.doc_tokens]
        self.avg_doc_len = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0.0

        self.term_freqs: List[Counter] = [Counter(toks) for toks in self.doc_tokens]

        # 文档频率：每个词出现在多少个文档里
        df = Counter()
        for tf in self.term_freqs:
            for term in tf.keys():
                df[term] += 1
        self.doc_freq = df
        self.n_docs = len(chunks)

    def _idf(self, term: str) -> float:
        n_t = self.doc_freq.get(term, 0)
        # BM25+ 平滑写法，避免 idf 出现负值
        return math.log((self.n_docs - n_t + 0.5) / (n_t + 0.5) + 1)

    def score(self, query: str, doc_idx: int) -> float:
        query_tokens = tokenize(query)
        tf = self.term_freqs[doc_idx]
        dl = self.doc_len[doc_idx]

        score = 0.0
        for term in query_tokens:
            if term not in tf:
                continue
            f = tf[term]
            idf = self._idf(term)
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / (self.avg_doc_len or 1))
            score += idf * (numerator / denominator)
        return score

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """返回 [(chunk下标, BM25分数), ...]，按分数降序。"""
        scores = [(i, self.score(query, i)) for i in range(self.n_docs)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


if __name__ == "__main__":
    import os
    from chunking import load_knowledge_base

    here = os.path.dirname(os.path.abspath(__file__))
    kb_dir = os.path.join(here, "..", "data", "knowledge_base")
    chunks = load_knowledge_base(kb_dir)

    index = BM25Index(chunks)
    query = "变异检测用什么工具"
    results = index.search(query, top_k=3)

    print(f"查询：{query}\n")
    for idx, s in results:
        print(f"score={s:.3f}  ({chunks[idx].source_file})  {chunks[idx].text[:50]}...")
