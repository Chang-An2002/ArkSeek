# -*- coding: utf-8 -*-
"""
精排模块（Cross-Encoder Rerank）

对应方案文档 2.2 / 4.2 节流水线的第三步：
    并行召回（BM25 + 向量） -> RRF融合排序 -> 精排（本文件） -> 最终展示

之前的版本里这一步一直是空的（见 pipeline.py 旧注释），因为环境装不上reranker
模型。现在补上，用的是和 dense_retriever.py 一样的"可插拔后端"思路：

- CrossEncoderReranker：真正的精排模型，用 sentence-transformers 自带的
  CrossEncoder 接口加载 BGE-reranker 系列模型。跟 dense_retriever.py 里的
  EmbeddingDenseRetriever 用的是同一个 sentence-transformers 库，不需要
  额外装新的pip包，只需要能联网下载这个模型（跟下载BGE embedding模型是同一类
  限制：云端环境下载不了，本地环境装好sentence-transformers之后可以下载）。

【模型选择：从 base 换成 v2-m3】一开始选的是轻量版 BAAI/bge-reranker-base
（用户当时的理由是"轻量、同一个库就能跑"）。踩坑5两处修复（sigmoid、候选池
大小）都用真实数据验证生效后，六条测试用例还剩最后一条没解决：知识库里
明确写着"可以特异性地在小鼠体内对肝脏组织中的细胞进行谱系标记"这句完整
连贯的话（片段79），BM25能找到它（5.79分）、候选池也没漏掉它（排第9/19），
但 bge-reranker-base 给它打的相关性分数只有0.008——几乎判定为完全不相关。
这说明瓶颈已经从"架构/代码问题"变成了"模型本身的语义理解能力不够"：轻量版
可能更适应"问题直接对应一句陈述句答案"的简单结构，对这种把关键信息包裹在
技术性长句里的表达方式，语义关联判断得不够准。用户决定换用参数量更大、
效果通常更好的 BAAI/bge-reranker-v2-m3 实测对比，直接检验"模型能力不够"
这个诊断是否成立——如果v2-m3能正确识别这条，就证明换模型有效；如果还是
不行，说明问题比预期更深（比如切分策略本身），需要再往下查。
- NoOpReranker：精排模型不可用时的退化版本，直接原样返回RRF融合的排序结果，
  不做任何调整——这样即使装不上reranker，整条流水线也能照常跑，行为跟"接入
  精排之前"完全一样。

关于"为什么精排比原始分数更靠谱"：BM25/向量的原始分数只在同一次查询内部的
排序上有意义（"这条比那条更像"），不是一个跨查询可比的绝对量尺——这也是之前
四次踩坑的根源。精排模型不一样，它是专门训练来"读懂"一对(查询, 候选文本)、
直接输出一个校准过的相关性分数的，理论上可以直接拿这个分数做阈值判断，
不用再靠"BM25>=5"这种手工拼出来的经验规则。

BGE-reranker 系列输出的是一个未经归一化的原始logit，习惯做法是过一遍
sigmoid把它压缩到(0, 1)区间再解读成"相关概率"，本文件里也是这么做的。

【踩坑5——sigmoid被套了两遍，所有分数都糊在0.5附近】
第一版代码里直接对 `model.predict(pairs)` 的输出手动做了一次sigmoid。用户在
自己电脑上实测后发现问题很明显：所有"应该拒答"的问题（比如"今天晚饭吃什么"），
不管候选片段是什么，精排分数全都精确地是0.500；哪怕是真正相关的问题，最高分也
从来没超过0.73。这不是巧合——sentence-transformers 的 CrossEncoder 对这一类
reranker 模型，很可能已经在 `predict()` 内部按模型自带的配置自动做过一次sigmoid，
输出的就已经是 (0,1) 区间的"相关概率"；代码又手动做了一遍sigmoid，相当于
sigmoid(sigmoid(raw))——两次压缩后，原本该趋近于0的"完全无关"分数被sigmoid(0)
精确映射成0.5，原本该趋近于1的"高度相关"分数最多也只能到sigmoid(1)≈0.731（跟实测
观察到的最高分0.729几乎完全对上，这是最直接的证据）。
更麻烦的是：0.5这个双重sigmoid后的"无关"基准值，刚好卡在原来定的置信度阈值
RERANK_FLOOR=0.5 上，用">="判断时"完全无关"反而被判定成了"可靠匹配"——这也是
为什么这次实测里，两条本该拒答的问题全都被错误地放行了。
修复：调用 `predict()` 时显式传入 `activation_fct=torch.nn.Identity()`，强制拿到
未经任何内部激活函数处理的原始logit，不依赖库或模型"默认要不要自动sigmoid"这个
不透明的行为，自己只做唯一一次sigmoid。
这也说明一件事：精排模型本身给出的是校准过的相关性信号，但"怎么正确地从库的
API里把这个信号取出来"，还是需要认真读文档、拿真实数据验证输出的数值范围
合不合理——不能想当然地假设"raw score就一定是没处理过的logit"。

【小插曲——activation_fct改名成activation_fn，装的sentence-transformers版本
不同，参数名不一样】用户真机测试时发现终端里反复打印一条`The activation_fct
argument was renamed and is now deprecated. Please use activation_fn instead.`
——不影响功能（库自己做了向后兼容），但每次提问都刷一遍，观感很差。这是
sentence-transformers某个版本把`CrossEncoder.predict()`的这个参数名从
`activation_fct`改成了`activation_fn`，旧名字还能用但会打印这条deprecation
warning。没有直接硬编码成新名字`activation_fn`，是因为不确定用户以后会不会
在更老版本的环境里跑这份代码（老版本根本没有`activation_fn`这个参数，硬编码
新名字会直接报错，等于把"警告"换成了"崩溃"，更糟）。改成在`__init__`加载
模型的时候用`inspect.signature()`看一眼当前装的这个版本的`predict()`实际
支持哪个参数名，装哪个用哪个——不用猜版本号，直接问库本身"你认哪个名字"，
两种版本的sentence-transformers都能拿到干净的输出、不打印这条警告。
"""

import os
from typing import List, Sequence


class NoOpReranker:
    """精排模型不可用时的占位版：不改变传入的候选顺序，只是原样透传。"""

    name = "无精排（占位版，直接用RRF融合排序）"
    is_real = False

    def rerank(self, query: str, passages: Sequence, top_n: int) -> List:
        return list(passages[:top_n])


class CrossEncoderReranker:
    """真正的精排模型：sentence-transformers 的 CrossEncoder + BGE-reranker。"""

    is_real = True

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", backend: str = "torch",
                 device: str = None, label: str = None):
        """backend="openvino"时model_name是build_openvino_reranker.py导出的本地目录，
        device是"GPU"/"CPU"（见模块末尾"精排加速"一节）。"""
        import inspect
        import logging

        from sentence_transformers import CrossEncoder  # 延迟导入，装不上时不影响其他模块

        # 跟 dense_retriever.py 的 EmbeddingDenseRetriever 一样，压掉HF Hub的
        # "未认证请求"提示，只压噪音，不影响真正的错误（异常照常抛出，由
        # build_reranker()的try/except接住）。
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

        self.model_name = model_name
        self.name = f"{label or model_name}（真实精排模型）"
        if backend == "torch":
            self.model = CrossEncoder(model_name)
        else:
            self.model = CrossEncoder(model_name, backend=backend, model_kwargs={"device": device or "CPU"})

        # sentence-transformers某个版本把 predict() 的这个参数从 activation_fct
        # 改名成了 activation_fn（旧名字还能用，但会打印deprecation warning，
        # 每次提问都刷一遍很影响观感）——用inspect在加载时直接问一下当前装的
        # 这个版本认哪个名字，而不是硬编码某一个，这样新旧两种版本的
        # sentence-transformers都能拿到干净输出，见上方"小插曲"一节。
        predict_params = inspect.signature(self.model.predict).parameters
        self._activation_kwarg = "activation_fn" if "activation_fn" in predict_params else "activation_fct"

    def rerank(self, query: str, passages: Sequence, top_n: int) -> List:
        import math
        import torch

        if not passages:
            return []

        pairs = [(query, p.chunk.text) for p in passages]
        # 强制拿到未经任何激活函数处理的原始logit，不依赖 sentence-transformers
        # 库或模型配置里"默认要不要自动过一遍sigmoid"这个不透明的行为——见上方
        # "踩坑5"，这正是之前一版代码翻车的地方。用哪个参数名由__init__里探测
        # 好的 self._activation_kwarg 决定，兼容新旧两种版本。
        raw_scores = self.model.predict(pairs, **{self._activation_kwarg: torch.nn.Identity()})

        for p, raw in zip(passages, raw_scores):
            # 自己动手做唯一一次sigmoid，把原始logit压缩到(0,1)，数值上更接近
            # "相关概率"，也方便和别的0~1分数放在一起看。
            p.rerank_score = 1.0 / (1.0 + math.exp(-float(raw)))

        ranked = sorted(passages, key=lambda p: p.rerank_score, reverse=True)
        return ranked[:top_n]


# 【精排加速（2026-09-30，回复慢）】精排在CPU上每条候选约0.5秒，本地检索最慢的一步。
# 同一个模型换成Intel的OpenVINO推理引擎、跑在本机的Arc核显上：60题对比
# （run_openvino_rerank_eval.py）过门槛判定60/60一样、找对31/32一样、前3条58/60一样、
# 最高分最大偏差0.004，精排总耗时803→166秒（约快4.8倍）。int8量化在CPU上快3.3倍但
# 前3条有13题变了、丢了1道找对的，没用。
# 需要的openvino等依赖会把transformers降级，装在单独的虚拟环境里；模型用
# build_openvino_reranker.py导出。环境变量ARKSEEK_RERANKER：
#   auto（默认）：导出过OpenVINO模型、且当前环境装了openvino，就用核显；否则用原来的PyTorch
#   openvino-gpu / openvino-cpu / torch：指定
# OpenVINO加载失败时自动退回PyTorch，再失败退回NoOp，跟以前一样不会让系统起不来。
def _openvino_reranker(device: str):
    import importlib.util
    from build_openvino_reranker import ov_dir
    path = ov_dir()[0]
    if importlib.util.find_spec("openvino") is None:
        raise RuntimeError("当前Python环境没有装openvino")
    if not os.path.exists(os.path.join(path, "openvino", "openvino_model.xml")):
        raise RuntimeError(f"没有找到导出的OpenVINO模型（{path}），先运行build_openvino_reranker.py")
    return CrossEncoderReranker(path, backend="openvino", device=device,
                                label=f"BAAI/bge-reranker-v2-m3 · OpenVINO {device}")


def build_reranker():
    """优先尝试真实精排模型，装不上/下载不了就自动退回 NoOp 占位版。"""
    mode = os.environ.get("ARKSEEK_RERANKER", "auto").lower()
    if mode in ("auto", "openvino-gpu", "openvino-cpu"):
        device = "CPU" if mode == "openvino-cpu" else "GPU"
        try:
            reranker = _openvino_reranker(device)
            print(f"[reranker] 已加载真实精排模型：{reranker.name}")
            return reranker
        except Exception as e:
            print(f"[reranker] 没有用OpenVINO加速（{type(e).__name__}: {e}），用PyTorch版")
    try:
        reranker = CrossEncoderReranker()
        print(f"[reranker] 已加载真实精排模型：{reranker.name}")
        return reranker
    except Exception as e:
        print(f"[reranker] 无法加载真实精排模型（{type(e).__name__}: {e}），"
              f"退回无精排占位版，置信度判断会继续用BM25兜底。"
              f"等本地环境能下载 BAAI/bge-reranker-v2-m3 后会自动切换。")
        return NoOpReranker()


if __name__ == "__main__":
    import os
    from chunking import load_knowledge_base
    from sparse_retriever import BM25Index
    from dense_retriever import build_dense_retriever
    from fusion import reciprocal_rank_fusion
    from pipeline import RetrievedPassage

    here = os.path.dirname(os.path.abspath(__file__))
    kb_dir = os.path.join(here, "..", "data", "knowledge_base")
    chunks = load_knowledge_base(kb_dir)

    bm25 = BM25Index(chunks)
    dense = build_dense_retriever(chunks)
    reranker = build_reranker()

    query = "这个课题是和哪个课题组合作的？"
    bm25_ranking = bm25.search(query, top_k=10)
    dense_ranking = dense.search(query, top_k=10)
    bm25_raw = dict(bm25_ranking)
    dense_raw = dict(dense_ranking)
    fused = reciprocal_rank_fusion([bm25_ranking, dense_ranking], k=60)

    pool = [
        RetrievedPassage(
            chunk=chunks[idx], rrf_score=score,
            bm25_score=bm25_raw.get(idx, 0.0), dense_score=dense_raw.get(idx, 0.0),
        )
        for idx, score in fused[:10]
    ]

    reranked = reranker.rerank(query, pool, top_n=3)
    print(f"\n查询：{query}\n")
    for p in reranked:
        rerank_display = f"{p.rerank_score:.3f}" if p.rerank_score is not None else "(无精排分数)"
        print(f"精排={rerank_display}  RRF={p.rrf_score:.4f}  BM25={p.bm25_score:.2f}  向量={p.dense_score:.3f}")
        print(f"  {p.chunk.text[:60]}...")
