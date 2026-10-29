# -*- coding: utf-8 -*-
"""
检索流水线（把切分/BM25/向量/RRF融合/精排串起来）

对应方案文档 4.2 节的三段式流水线：
  并行召回（BM25 + 向量） -> RRF融合排序 -> 精排（Cross-Encoder Rerank）

以及 2.3 / 4.2 节的"可靠性机制"：相关性分数低于阈值时，不硬答，明确说没找到
可靠信息，并且每条结果都带着来源文件名，方便"引用约束"。

关于精排：这一步之前一直是空的，因为环境下载不了reranker模型。现在补上了
（见 reranker.py），用的是和向量检索一样的"可插拔后端"设计：本地环境能下载
BGE-reranker 系列模型（当前默认 BAAI/bge-reranker-v2-m3，见 reranker.py 里
"模型选择：从base换成v2-m3"的说明）就用真实精排，装不上就自动退回"不做精排、
直接用RRF排序"的占位版，不影响其余代码。精排接入后，置信度判断也跟着换了
信号来源，接入过程中也确实测出了新问题（sigmoid被套两遍、候选池太窄）——见
answer() 里"第五次踩坑"的完整说明。

关于查询改写（原"查询拆解"，已更正）：阶段一排查"lineage tracing课题使用
的是什么动物的什么组织"这条已知局限时，曾归因为"精排对一次问两件事的复合
问法系统性打低分"，并写了一个正则拆分模块（query_decompose.py）做缓解。
2026-09-24的变量分离实验推翻了这个归因：同一个片段79，复合/单一问法分数
几乎一样，真正决定分数的是措辞——问题用英文"lineage tracing"，原文写的是
中文"谱系标记""CRISPR条形码系统"。当年的对照实验同时改了"拆分"和"措辞"
两个变量。正则拆分实测对结果零贡献（还略微抬高了干扰项分数），已删除。
现在的做法是接LLM时由 query_planner.rewrite_for_kb() 把问题改写成用词更
贴近知识库原文的等价问法（multi-query retrieval），只在gate第一次没过时
触发，见 retrieve_with_gate()；完整经过见README"关于精排与已知局限"一节。
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence

from chunking import Chunk, load_knowledge_base
import kb_sources
from sparse_retriever import BM25Index
from dense_retriever import build_dense_retriever
from fusion import reciprocal_rank_fusion
from reranker import build_reranker
from kb_questions import strip_questions
from timing import span


@dataclass
class RetrievedPassage:
    chunk: Chunk
    rrf_score: float
    bm25_score: float
    dense_score: float
    rerank_score: Optional[float] = None  # 没接真实精排模型时为 None
    matched_query: Optional[str] = None  # 这个分数是靠原始问题还是改写问法打出来的
    # L2专用：时间衰减/闲聊降权之前的原始相关性分数。置信度gate看这个（"相不
    # 相关"），排序看衰减后的 rerank_score/bm25_score（"相关的里面谁优先"），
    # 见 conversation_memory._search_history()。L3不用这个字段。
    relevance_score: Optional[float] = None


class MemorySystemPipeline:
    def __init__(self, kb_dir: str, top_k_per_channel: int = 10, rrf_k: int = 60,
                 rerank_pool_size: int = None, chroma_dir: Optional[str] = "default",
                 use_external_sources: bool = True):
        """
        参数：
            kb_dir: 知识库目录
            chroma_dir: 知识库向量索引的持久化目录（接入Chroma之后新增，见
                chroma_index.py）。默认放在本机目录（chroma_index.default_chroma_dir()，
                不在坚果云同步目录里，原因见该函数）；传 None 表示不用Chroma、每次启动
                在内存里全量计算向量（改动前的行为）。
            top_k_per_channel: 每一路召回各取多少候选
            rrf_k: RRF融合公式里的平滑常数
            rerank_pool_size: 融合排序后，交给精排模型重新打分的候选池大小。
                不传的话默认是 `2 * top_k_per_channel`（两路各10条、互不重叠时
                最多20条唯一候选）——也就是"两路召回到的候选，一条不漏地全部
                交给精排去判断"，而不是先用 RRF 融合排名砍掉一部分。这是实测
                踩坑5之后才调整的：一开始图省事只让精排看融合排序前10名，结果
                发现某条BM25排名不错、但只被BM25一路召回到（向量那一路没覆盖
                到）的候选，会因为RRF融合公式本身的机制被压到融合排名10名之后，
                根本轮不到精排看它一眼——精排能力再强，也没法给它没见过的候选
                打分。所以候选池不该被RRF这种粗筛提前砍得太窄，精细排序应该
                交给更懂语义的精排去做，RRF在这里只负责"去重+给个初始参考顺序"。

        置信度阈值没有做成初始化参数，是因为这部分判断逻辑几次实测下来一直在变
        （见 answer() 里的详细说明），不适合现在就定死一个数字参数。
        """
        # 知识库扩充之后：本地txt + Zotero文献目录 + JC材料，见kb_sources.py。
        # use_external_sources=False 退回只读本地txt（改动前的行为，旧测试/对照用）。
        self.catalog = kb_sources.build_catalog() if use_external_sources else None
        if use_external_sources:
            self.chunks: List[Chunk] = kb_sources.load_knowledge_sources(kb_dir, self.catalog)
        else:
            self.chunks = load_knowledge_base(kb_dir)
        from admin_knowledge import load_admin_chunks
        from pathlib import Path
        self.chunks.extend(load_admin_chunks(Path(kb_dir).resolve().parent, start_chunk_id=len(self.chunks)))
        if not self.chunks:
            raise ValueError(f"知识库目录 {kb_dir} 下没有找到任何文档")

        n_docs = len(set(c.meta.get("doc_id", c.source_file) for c in self.chunks))
        if self.catalog:
            import bot_profile
            bot_profile.set_local_kb_description(kb_sources.local_kb_description(self.catalog))
        print(f"[pipeline] 已加载知识库：{len(self.chunks)} 个片段，来自 {n_docs} 份文档")

        self.bm25 = BM25Index(self.chunks)
        if chroma_dir == "default":
            from chroma_index import default_chroma_dir
            chroma_dir = default_chroma_dir()
        self.dense = build_dense_retriever(self.chunks, chroma_dir=chroma_dir)
        self.reranker = build_reranker()
        # 模型实例被多个线程共用（qq_bot不同对话并行、一条消息内子问题并行，见
        # concurrency.py）：HuggingFace的fast tokenizer被两个线程同时调用会报
        # "Already borrowed"，所以encode/predict各套一把锁。单次只要毫秒到几百毫秒，
        # 排队代价很小。ConversationSession和Chroma索引拿到的都是这里的同一个实例
        from concurrency import LockedCalls
        if getattr(self.dense, "model", None) is not None:
            self.dense.model = LockedCalls(self.dense.model, {"encode"})
        if getattr(self.reranker, "model", None) is not None:
            self.reranker.model = LockedCalls(self.reranker.model, {"predict"})

        self.top_k_per_channel = top_k_per_channel
        self.rrf_k = rrf_k
        self.rerank_pool_size = rerank_pool_size or (2 * top_k_per_channel)

    # 同义说法表的扩展问法：召回时每个都查，精排时合成一个问法、代替原问题只打一遍分
    # （2026-09-30，回复慢）。False = 改动前的做法（原问题和每个扩展问法各精排一遍）
    MERGE_EXPANSIONS_FOR_RERANK = True

    def retrieve(self, query: str, top_n: int = 3,
                 rewrites: Optional[List[str]] = None,
                 doc_types: Optional[Sequence[str]] = None,
                 expansions: Optional[List[str]] = None,
                 paper_keys: Optional[Sequence[str]] = None) -> List[RetrievedPassage]:
        """rewrites：LLM改写出的、用词更贴近知识库原文的等价问法（见
        query_planner.rewrite_for_kb()），只由 retrieve_with_gate() 在gate
        第一次没过时传入。传入时每个问法都各自跑一遍BM25+向量召回，所有
        排名一起做RRF融合（multi-query retrieval的标准做法），候选池按问法
        数量等比放大，精排时每条候选取所有问法里的最高分。不传时行为跟
        改动前完全一致。

        expansions：同义说法表给出的扩展问法（query_synonyms.expand_for_project()）。
        召回时跟rewrites一样每个都查；精排时合成一个问法（"原问题（栏目名…）"，
        已经包含原问题），代替原问题只打一遍分，候选池也不再按扩展问法的个数放大——
        命中同义表时原来要精排两三遍，是最慢的一种情况（见run_latency_profile.py）。

        doc_types：只在这几类文档里检索（元数据过滤，LangChain的self-query
        retriever是同一个思路），比如路由判断问的是作者自己的课题时只查
        ["课题报告"]。知识库扩充后，"这个课题的scRNA-seq用什么软件比对"
        检索到的是一篇文献里的Bowtie1——文献原文比两份报告多得多，问"这个
        课题"的问题被冲散了。实现上是两路各自全量排序后按类型筛，再取前
        top_k_per_channel条（知识库只有一千多个片段，全量排序代价可以忽略）。

        paper_keys：只在这几篇文献相关的片段里检索（片段元数据paper_key：文献原文、
        目录条目、讲这篇的JC讲稿/PPT/笔记都带）。2026-10-02加，给"最近一次JC讲的
        什么"这类按时间定位的问题用——是哪一次由代码按目录日期算（见
        conversation_memory._jc_session_filter()），检索不懂"最近""第一次"。"""
        expansions = list(expansions or [])
        all_queries = [query] + list(rewrites or []) + expansions  # 召回用
        rerank_queries = [query] + list(rewrites or [])            # 精排用
        if expansions and self.MERGE_EXPANSIONS_FOR_RERANK:
            from query_synonyms import merge_variants
            rerank_queries[0] = merge_variants(query, expansions)
        else:
            rerank_queries += expansions
        allowed = None
        if doc_types or paper_keys:
            allowed = {i for i, c in enumerate(self.chunks)
                       if (not doc_types or c.meta.get("doc_type") in doc_types)
                       and (not paper_keys or c.meta.get("paper_key") in paper_keys)}

        def channel(retriever, q):
            if allowed is None:
                return retriever.search(q, top_k=self.top_k_per_channel)
            ranked = retriever.search(q, top_k=len(self.chunks))
            return [(i, s) for i, s in ranked if i in allowed][:self.top_k_per_channel]

        rankings = []
        bm25_raw, dense_raw = {}, {}
        for q in all_queries:
            scope = "只查" + "/".join(doc_types) if doc_types else ("只查这几篇文献" if paper_keys else "全库")
            with span("召回", f"BM25+向量，{scope}"):
                bm25_ranking = channel(self.bm25, q)
                dense_ranking = channel(self.dense, q)
            rankings += [bm25_ranking, dense_ranking]
            # 记录原始分数，供置信度判断/展示用（RRF分数只反映排名，见 answer()
            # 里的说明）；多个问法时取各问法里的最高分
            for idx, sc in bm25_ranking:
                bm25_raw[idx] = max(bm25_raw.get(idx, 0.0), sc)
            for idx, sc in dense_ranking:
                dense_raw[idx] = max(dense_raw.get(idx, 0.0), float(sc))

        fused = reciprocal_rank_fusion(rankings, k=self.rrf_k)

        # 先按融合排序取一个候选池（可能比最终要返回的 top_n 大），再交给精排
        # 模型重新打分排序——这就是"RRF融合排序 -> 精排"这两步的分工：RRF负责
        # 从全部片段里粗筛出一批看起来还行的候选，精排负责在这批候选里精细排序。
        # 多个问法时池子等比放大，保持"各路召回到的候选一条不漏交给精排"
        # （踩坑5的教训，见 __init__ 里 rerank_pool_size 的说明）。
        pool_size = max(top_n, self.rerank_pool_size * len(rerank_queries))
        pool = [
            RetrievedPassage(
                chunk=self.chunks[chunk_idx],
                rrf_score=score,
                bm25_score=bm25_raw.get(chunk_idx, 0.0),
                dense_score=dense_raw.get(chunk_idx, 0.0),
            )
            for chunk_idx, score in fused[:pool_size]
        ]

        # 精排：用每个问法（原问题+改写）各给整个候选池打一遍分，每条候选取
        # 所有问法里的最高分作为最终精排分数，matched_query 记下是哪个问法
        # 打出来的。没有改写时就是原问题打一遍，跟改动前完全一样。
        queries_to_try = rerank_queries

        best_score = {}
        best_query = {}
        for q in queries_to_try:
            with span("精排", f"{len(pool)}条候选"):
                self.reranker.rerank(q, pool, top_n=len(pool))  # 就地把每条候选的 rerank_score 改成这个问法下的分数
            for p in pool:
                cid = p.chunk.chunk_id
                if p.rerank_score is not None and (
                    cid not in best_score or p.rerank_score > best_score[cid]
                ):
                    best_score[cid] = p.rerank_score
                    best_query[cid] = q

        for p in pool:
            cid = p.chunk.chunk_id
            if cid in best_score:
                p.rerank_score = best_score[cid]
                p.matched_query = best_query[cid]
            else:
                p.matched_query = query  # NoOp占位版：没有精排分数，记原始问题占位

        ranked = sorted(
            pool,
            key=lambda p: p.rerank_score if p.rerank_score is not None else -1,
            reverse=True,
        )
        return ranked[:top_n]

    # 置信度阈值：跟 answer() 里五次踩坑记录的结论保持完全一致，只是从函数体
    # 内的局部变量提升成类常量，好让 retrieve_with_gate() 和 answer() 共用
    # 同一份判断逻辑，不用维护两份容易慢慢跑偏的重复代码。
    BM25_FLOOR = 5.0
    RERANK_FLOOR = 0.5
    # 靠LLM改写才过线时用更高的门槛：改写等于多给了几次机会、每条候选取所有
    # 问法里的最高分，偶然出现高分的概率随之变大（跟多重检验要收紧显著性
    # 阈值是一个道理）。0.75来自 run_kb_rewrite_demo.py 的真机观察：被改写
    # 救回的真阳性 0.864~0.985，"领域相近但知识库没有答案"的探针被改写推高到
    # 0.404~0.651（其中一例是把hit译成"命中靶点"后跟原文"CRISPR靶点参考序列"
    # 字面撞上，过了0.5）。样本很小（真阳性3道题、假阳性探针2道题），是初始
    # 经验值，以后有更多数据再校准。
    REWRITE_RERANK_FLOOR = 0.75
    REJECT_MESSAGE = "【未找到可靠信息】没有检索到置信度足够的相关内容，不做猜测性回答。"

    def retrieve_with_gate(self, query: str, top_n: int = 3, query_rewriter=None,
                           doc_types: Optional[Sequence[str]] = None,
                           paper_keys: Optional[Sequence[str]] = None):
        """跟 answer() 是同一套检索+置信度判断，只是不格式化成字符串，直接把
        (passages, passed) 原样返回——2.2要在置信度通过之后再调用LLM生成一段
        自然语言回答，需要拿到结构化的passages去拼引用来源，而不是一段已经
        格式化好的字符串；这个方法就是给2.2用的，2.1的 answer() 不受影响，
        只是内部改成调用这个方法，判断逻辑完全没变，行为不变。

        passed=False 时 passages 可能是空列表也可能有内容（比如召回到了、但
        分数都没过阈值）——调用方判断"要不要往下走"只应该看 passed，不要看
        passages 是否为空。passed=False 时不应该再调用LLM生成——置信度gate
        是防幻觉的硬性防线，这条防线不能被生成模型绕过去，见模块开头的说明。

        query_rewriter：可选，`query -> List[str]`，接LLM时由
        conversation_memory.py 传入 query_planner.rewrite_for_kb()。只在第一次
        gate没过、且精排可用时才调用：LLM把问题改写成用词更贴近知识库原文的
        等价问法（比如英文术语换成中文），跟原问题一起重新召回+精排，再判一次
        gate——第二次用更高的 REWRITE_RERANK_FLOOR（见该常量的说明）。改写
        本身被约束为不许把猜测的答案写进问题（见 rewrite_for_kb()），否则
        就是让LLM用自己的猜测帮自己通过这道防线。

        doc_types/paper_keys：见 retrieve()，改写重试时沿用同一个过滤条件。

        query里"100字以内"这类对回答形式的要求先去掉再检索（会拉低相关性，见
        length_limit.strip_length_request()）；生成回答时调用方用的仍是完整原句。
        """
        from length_limit import strip_length_request
        from query_synonyms import expand_for_project
        query = strip_length_request(query)
        # 问课题时，按同义说法表多带一两个写着报告栏目名的问法一起检索（查询扩展，
        # 人工维护、结果确定，跟原问题同一道门槛），见query_synonyms.py
        variants = expand_for_project(query, doc_types)
        passages = self.retrieve(query, top_n=top_n, doc_types=doc_types, expansions=variants,
                                 paper_keys=paper_keys)
        passed = self._gate(passages)
        if passed or query_rewriter is None or not self.reranker.is_real:
            return passages, passed

        llm_rewrites = query_rewriter(query) or []
        rewrites = llm_rewrites + variants  # 只用于下面的日志
        if not llm_rewrites:
            return passages, passed
        retried = self.retrieve(query, top_n=top_n, rewrites=llm_rewrites, doc_types=doc_types,
                                expansions=variants, paper_keys=paper_keys)
        retried_passed = self._gate(retried, rerank_floor=self.REWRITE_RERANK_FLOOR)
        best = max((p.rerank_score for p in retried), default=0.0)
        print(f"[pipeline] 置信度没过，用LLM改写的{len(rewrites)}个问法重新检索："
              f"{rewrites} -> 最高精排分{best:.3f}（改写路径门槛{self.REWRITE_RERANK_FLOOR}），"
              f"{'通过' if retried_passed else '仍未通过'}")
        return retried, retried_passed

    def _gate(self, passages: List[RetrievedPassage], rerank_floor: Optional[float] = None) -> bool:
        if not passages:
            return False
        if self.reranker.is_real:
            floor = self.RERANK_FLOOR if rerank_floor is None else rerank_floor
            return max(p.rerank_score for p in passages) >= floor
        return max(p.bm25_score for p in passages) >= self.BM25_FLOOR

    def format_evidence(self, passages: List[RetrievedPassage]) -> str:
        """2.1原本 answer() 里"展示证据"那段格式化逻辑，抽出来给2.2复用——
        LLM生成失败时（网络问题/调用失败，chat()返回None），2.2应该退回展示
        原始证据，而不是编一段假回复或者直接报错；这正是2.2各层"LLM能用就用，
        不能用就退回2.1对应实现"这个统一降级设计在L3这里的落地。
        """
        lines = [f"检索到 {len(passages)} 条相关片段"
                 f"（{'已精排' if self.reranker.is_real else '按融合排序，未精排'}，"
                 f"附来源，可用于后续生成回答时的引用）：\n"]
        for i, p in enumerate(passages, start=1):
            rerank_display = f"，精排={p.rerank_score:.3f}" if p.rerank_score is not None else ""
            lines.append(
                f"[{i}] 来源：{p.chunk.source_file}"
                f"（RRF={p.rrf_score:.4f}，BM25={p.bm25_score:.2f}，向量={p.dense_score:.3f}{rerank_display}）\n"
                f"    {strip_questions(p.chunk.text)}\n"  # 补进去的问题只给检索用，不展示
            )
        return "\n".join(lines)

    def answer(self, query: str, top_n: int = 3) -> str:
        """
        这一版本不接生成模型，只做"检索+可靠性判断+引用展示"，
        对应方案文档里说的"先验证核心逻辑，生成环节等检索质量稳定了再接"。

        置信度判断的三次实测踩坑（写在这里是因为都是测出来的，不是拍脑袋定的）：

        第一次：一开始直接拿 RRF 分数做阈值判断，测试后发现不行——RRF分数只反映
        "排第几"，哪怕两路召回其实都跟问题毫不相关，排名第一的候选依然会拿到一个
        正的RRF分数。改成看BM25/向量的原始分数。

        第二次：换成用3篇占位小样本测的时候，"原始分数是否精确为0"这个规则够用；
        但换上真实的181个片段的论文全文后，问一个完全无关的问题（"米哈游的游戏
        运营岗位需要什么能力？"）BM25分数不再是精确的0，而是3.72——原因是"需要"
        这种常见词恰好只出现在181个片段中的4个里，被IDF公式当成了"稀有词"。
        真正相关的问题（哪怕是召回质量比较弱的），BM25分数都在6以上。所以把判断
        标准从"精确为0"放宽成一个凭这几次测试定的经验下限：BM25>=5 或 向量分数
        >=0.1，两个都够不上才判定"没有可靠信息"。

        第三次：换上真实的 BGE embedding（BAAI/bge-small-zh-v1.5）后，用户自己在
        本地测了4个问题，暴露了"向量分数>=0.1 也能单独放行"这条OR规则的问题：
          - "今天晚饭吃什么"——正确拒答（BM25/向量都很低）
          - "这个课题是和哪个课题组合作的"——正确召回，BM25=11.69~14.74
          - "lineage tracing课题使用的是什么动物的什么组织"——召回较弱但相关
          - "米哈游的游戏运营岗位需要什么能力？"——本该拒答，却被放行了：
            BM25 只有 2.64~2.88（远低于阈值5），但向量分数是 0.366~0.396，
            比 DENSE_FLOOR=0.1 高出不少，靠OR条件混过了置信度检查
        原因：真实 sentence embedding（尤其是较小的模型，没做白化/校准）普遍存在
        "各向异性"问题——不同句子的向量在空间里挤在一个比较窄的锥形区域内，导致
        哪怕两句话完全不相关，原始余弦相似度也会被"垫高"到一个中等偏上、看起来
        "有点像"的数值，不能像 TF-IDF 那样直接拿一个固定阈值去卡。而 BM25 在这
        几次测试里表现稳定：只要问题跟知识库真的无关，分数都明显低于5，跟真正
        相关的问题（6以上，通常两位数）拉得开差距。
        修复：向量分数不再拥有"独立放行"的权力，改成必须通过 BM25 这一关
        （BM25 >= BM25_FLOOR）才算"有真实匹配"；向量分数继续算出来、展示出来，
        供参考和后续调试，但不再单独决定"能不能通过置信度检查"。

        第四次：用批量测试脚本（run_tests.py）在真实embedding下正式跑了一遍6条
        用例，5条通过，但"lineage tracing课题使用的是什么动物的什么组织"这条本该
        召回却被误判成"没有可靠信息"。排查发现：融合排序第1名的候选片段BM25只有
        4.88（差一点点没过5.0的门槛），但排第2名的候选片段BM25其实高达10.05、
        明显相关——只是它没有进入向量检索自己的Top-K候选（记录成向量分数0.000），
        所以RRF融合时排名被压到了第2位。而当时的判断逻辑只看
        `passages[0].bm25_score`，也就是只看融合排序里的第1名，压根没看第2名，
        于是明明"有一条很相关的片段"，却因为它没排到融合结果的第1位而被忽略。
        这本质上和第一次踩坑是同一类问题的变种：融合排名只反映"综合两路信号的
        排序"，不等于"BM25最强的那条一定排第一"——BM25和向量各自的候选池不同、
        排序方式也不同，两路一融合，某条单项分数很高的候选完全可能被挤到第2、
        第3名。只检查排名第1的原始分数，等于把"融合重排"的副作用当成了"这条
        信息不可靠"的证据。
        修复：置信度判断不再只看融合排序第1名的BM25，而是看返回的这几条候选里
        BM25最高的那个值（`max(p.bm25_score for p in passages)`），只要其中有一条
        真正匹配，就不因为它被排到第2、第3名而漏判。

        这四次踩坑说明同一件事：简单阈值规则天生不稳定，换一种检索后端、换一批
        语料、甚至换一条新的测试问题，都可能暴露新的边界情况；阈值和判断逻辑要
        跟着实测持续修正。更严谨的做法是接一个精排（reranker）模型给出校准过的
        相关性分数做判断依据——现在真的接上了（见 reranker.py），置信度判断的
        信号来源也跟着换了，见下面"第五次"的说明。

        第五次：接入精排（BAAI/bge-reranker-base）之后，用户在自己电脑上第一次
        真实跑 run_tests.py，结果从接入前的4/6反而退步到3/6——两条本该拒答的
        问题（"今天晚饭吃什么"、"米哈游..."）全都被误判成"可靠匹配"放行了。
        排查发现两个独立问题叠在一起：
          (a) sigmoid被套了两遍。代码里手动对 `model.predict(pairs)` 的输出又
          做了一次sigmoid，但 sentence-transformers 的 CrossEncoder 对这类
          reranker模型很可能已经在 predict() 内部按模型自带配置自动做过一次
          sigmoid了。两次sigmoid叠加后，"完全无关"的分数被精确压成了
          sigmoid(0)=0.5，"高度相关"的分数最多也只能到sigmoid(1)≈0.731（跟
          实测最高分0.729几乎完全对上，这是最直接的证据）。更糟的是0.5正好
          卡在当时定的RERANK_FLOOR=0.5上，用">="判断时"完全无关"反而被判成
          "可靠"——这是两条应拒答用例被误放行的直接原因。
          修复：调用 predict() 时显式传 `activation_fct=torch.nn.Identity()`，
          拿到未经内部sigmoid处理的原始logit，自己只做唯一一次sigmoid（详见
          reranker.py 里的完整记录）。
          (b) 候选池被RRF融合提前砍得太窄。即使分数算对了，"林盛达实验室/
          小鼠肝脏"那条已知局限用例（chunk 79）依然答不对，因为它一开始就没
          进入精排能看到的候选池——旧代码只把RRF融合排序前10名交给精排，而
          chunk 79 因为向量检索没覆盖到它、被RRF挤到10名之外，精排能力再强
          也没法给一个它没见过的候选打分。
          修复：候选池大小从固定10改成 `2 * top_k_per_channel`（默认20，见
          __init__ 里 rerank_pool_size 参数的说明），让RRF只做"去重+给个初始
          参考顺序"，不再替精排提前筛掉候选。
        这次教训和前四次是同一个道理：换一个新组件（精排模型），哪怕理论上
        更靠谱，也照样可能在真实数据上暴露没想到的问题（库的隐藏行为、上游
        步骤的副作用）——所以两处修复都还需要你在自己电脑上重新跑一遍
        run_tests.py 验证，RERANK_FLOOR=0.5 这个阈值本身也是在"分数被污染"
        的情况下定的，修复后分数分布会完全不同，同样需要用真实数据重新
        确认这个阈值是否合适。

        【2.2接入后的说明】这个方法内部现在调用的是 retrieve_with_gate() +
        format_evidence()（见上方两个方法），判断逻辑和格式化文本跟这段
        docstring记录的历史完全一致，只是抽成了可以被2.2复用的两个独立步骤，
        本方法的行为（包括返回值的文本格式）没有任何变化。
        """
        passages, passed = self.retrieve_with_gate(query, top_n=top_n)
        if not passed:
            return self.REJECT_MESSAGE
        return self.format_evidence(passages)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    kb_dir = os.path.join(here, "..", "data", "knowledge_base")

    pipeline = MemorySystemPipeline(kb_dir)

    test_queries = [
        "变异检测常用什么工具？",
        "多组学数据整合有哪些方法？",
        "今天晚饭吃什么？",  # 故意问一个知识库里完全没有的问题，验证"可靠性兜底"
    ]

    for q in test_queries:
        print("=" * 60)
        print(f"问：{q}\n")
        print(pipeline.answer(q))
