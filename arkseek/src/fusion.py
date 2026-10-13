# -*- coding: utf-8 -*-
"""
融合排序模块（Reciprocal Rank Fusion, RRF）

对应方案文档 2.2 / 4.2 节。核心特点：只看"排第几"，不看"具体打了多少分"，
所以能把量纲完全不同的 BM25 分数和余弦相似度分数放在一起融合，不需要额外调参。

公式： score(d) = sum_i 1 / (k + rank_i(d))
其中 rank_i(d) 是文档 d 在第 i 路召回结果里的排名（从1开始），k 常取 60（Elasticsearch的默认值）。
"""

from typing import List, Tuple, Dict


def reciprocal_rank_fusion(
    rankings: List[List[Tuple[int, float]]],
    k: int = 60,
) -> List[Tuple[int, float]]:
    """
    参数：
        rankings: 多路召回结果列表，每一路是 [(chunk_idx, 原始分数), ...]（已按分数降序）
        k: RRF 公式里的平滑常数，默认60

    返回：
        融合后的 [(chunk_idx, RRF分数), ...]，按RRF分数降序
    """
    fused_scores: Dict[int, float] = {}

    for ranking in rankings:
        for rank, (chunk_idx, _original_score) in enumerate(ranking, start=1):
            fused_scores[chunk_idx] = fused_scores.get(chunk_idx, 0.0) + 1.0 / (k + rank)

    fused = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return fused


if __name__ == "__main__":
    # 简单自测：两路召回，验证融合逻辑符合预期
    bm25_ranking = [(0, 11.7), (1, 8.9), (2, 8.8)]       # 片段0在BM25里排第1
    dense_ranking = [(2, 0.4), (0, 0.3), (3, 0.2)]        # 片段2在向量检索里排第1

    fused = reciprocal_rank_fusion([bm25_ranking, dense_ranking], k=60)
    print("融合结果（chunk_idx, RRF分数）：")
    for idx, score in fused:
        print(f"  chunk {idx}: {score:.5f}")
    # 片段0和片段2都在各自那一路排第1，融合后应该并列最高，且都高于只出现在一路里的片段
