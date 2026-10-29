# -*- coding: utf-8 -*-
"""
多轮对话记忆入口（L0会话窗口 + L1用户画像 + L2情景记忆 + L3知识库）
—— 阶段二 2.1（不接LLM）

对应方案文档 4.1 节："L0/L1 决定机器人记不记得你，L2 决定翻旧账的能力，
L3 决定回答准不准、可不可靠"——这三句话对应本文件里的三个方法：
    _l1_step()                  ->  L1，记不记得"你是谁"
    _try_answer_from_history()  ->  L2，翻不翻得到"之前聊过的相关内容"
    _try_answer_from_kb()       ->  L3，阶段一已经验证过的知识库检索（2.2加了LLM生成这一层）

【2.1的核心设计取舍：不接生成模型意味着什么】
没有LLM，"回答"就没法把L0/L1/L2/L3揉合成一句自然语言总结，只能老实展示
"命中了哪一层、证据是什么"——这跟阶段一"给结构化片段而不是一句话答案"是
同一种诚实的做法，只是现在多了L1/L2两种"证据来源"。四层的优先级是：
先看L1（这是不是在问我认不认识你），再看L2（是不是在问之前聊过的事），
都没命中才落到L3（在知识库里找）——这个优先级本身就是一个规则化的判断，
不是语义理解，局限性在各自方法的注释里说明。

【L2检索为什么直接复用L3的BM25/RRF/精排代码，而不是另起一套】
方案文档4.2节写得很明确："混合召回流水线，应用于 L2 情景记忆 与 L3 知识库"
——检索这件事的本质（关键词匹配+语义相似度+精排）不因为语料是"论文片段"
还是"历史对话"而改变，唯一的区别是L2需要多叠加一个"时间衰减"权重（越新的
对话权重越高）。所以这里没有重新发明检索逻辑，而是把每一轮历史对话包成
一个 chunking.Chunk，交给已经在阶段一验证过的 BM25Index / dense_retriever /
reciprocal_rank_fusion 重新跑一遍——语料从"知识库文档"换成"对话历史"，
代码完全不用改。精排模型也直接复用 kb_pipeline.reranker 这同一个已加载的
实例，不重新加载模型。

【一个必须避开的性能坑：不能每轮对话都重新加载embedding模型】
sentence-transformers 加载模型本身有实打实的耗时（哪怕文件已经在本地缓存），
如果L2检索每轮都重新 `EmbeddingDenseRetriever(chunks)`，多轮对话场景下会
一轮比一轮卡——这是查询拆解那次"3倍精排开销"踩坑给的教训：新功能带来的
性能代价，能在设计阶段想清楚就不要等实测才发现。这里的做法是直接复用
kb_pipeline.dense 已经加载好的模型实例（`self._embedding_model`）来给
对话历史临时编码，只有"语料变了"（每轮历史都在变长），没有"重新加载模型"
这一步；BM25/TF-IDF 本身零加载成本，现建现用没有这个问题。

【接入Chroma之后：L2稠密检索不再每次现算全部历史】
上面这个做法避开了"重新加载模型"，但还剩一个随历史增长越来越慢的问题：
每次翻旧账都要把这个用户的全部历史重新编码一遍。现在稠密这一路改查
chroma_index.HistoryVectorIndex（持久化在data/chroma_db/，每条发言只嵌入
一次），BM25这一路不变（历史语料小，现建现用没有性能问题）。SQLite仍然是
原件，Chroma只是索引：`_reconcile_history_index()`是唯一的同步入口，会话
启动时、每轮对话结束时、每次检索前各调一次。查询范围刻意保持跟改动前
一致（role=user、排除is_general_chat=1），完整设计见README"接入Chroma"一节。

【这个文件同时是2.1和2.2的入口——靠 llm_client 参数区分，不是两份代码】
构造 ConversationSession 时不传 llm_client（或传None），就是纯2.1行为：
llm_client 会自动变成 llm_client.py 里的 NoLLMClient，L1的读写全走本文件
下面 _l1_step() 里的规则版分支（extract_facts/_try_answer_from_profile/
looks_like_question），跟2.1原本的行为完全一致，可以直接拿去跑2.1的回归
测试。传入一个真实的LLM客户端（build_llm_client()的返回值），_l1_step()
会优先尝试 llm_l1.py 的语义分类，只有LLM调用失败/返回不合法JSON时才会
退回同一套规则版逻辑——这正是 llm_client.py/llm_l1.py 模块开头反复强调的
"2.1不是被淘汰的废弃品，是2.2天然的降级路径"这句话在这个文件里的具体
落地方式。

【2.2在L1之外新增的两块：L0滚动摘要 + L2/L3生成式回答】
L1的LLM化验证通过之后，2.2补上剩下两块——都遵循同一套"llm_client.is_real
才做、不可用就退回2.1对应实现"的降级哲学：
- `_maybe_update_l0_summary()`：L0窗口每滑出去一段内容，就用LLM把它滚动
  合并进一份持续更新的摘要（memory_store.py的l0_summary表），不再是滑出
  窗口就"只能靠L2精确检索翻回来"这一种去处。完整设计见 llm_l0_summary.py
  模块开头的说明。2.1（NoLLMClient）下这个方法整体是空操作。
- `_try_answer_from_kb()`（L3）和 `_try_answer_from_history()`（L2）：置信度
  gate判定"可信"之后，不再直接展示编号证据列表，而是先尝试用LLM把这些
  证据揉合成一段自然语言回答（带[编号]引用），生成失败/LLM不可用时退回
  2.1原本的"展示证据"格式。gate判定"不可信"时完全不会调用LLM——防幻觉的
  硬性防线不能被生成模型绕过，完整设计见 llm_generate.py 模块开头的说明。

【L0上下文窗口接入L1判断】
`get_context_window()`（L0窗口）之前只用来给`cli.py`展示，没有喂给过任何
LLM调用——包括`_l1_step()`调用的`llm_l1_turn()`，导致L1判断"这句话是不是
关于用户自己的新信息"时看不到最近几轮说了什么，追问式的短答案容易被误判。
完整问题和修复设计见`llm_l1.py`模块开头"L0上下文窗口未接入L1判断"一节，
这里只记落地方式：`user_says()`在把当前发言存进数据库**之前**先取一次
`get_context_window()`，往下传给`_l1_step()` -> `llm_l1_turn()`；一定要在
存库之前取，不然取到的窗口会把这句话自己也算进"最近对话"里。

【L3的第二个知识来源：明日方舟wiki实时查询】
"进阶目标"原型（`prts_wiki_client.py`/`fetch_page_for_answer()`）真机验证
通过、精排适用性也认真实验验证过之后（完整过程见README"进阶目标"一节），
接入这里成为L3的第二个知识来源——之前只是独立脚本，现在能在真实多轮对话
里被自然调用。新增`_try_answer_from_router()`，内部先用`wiki_router.py`的
`route_query()`判断这句话该怎么处理（这一步本身也是一次LLM调用，只在2.2下
生效，见该文件模块开头说明）：路由到wiki就调用`fetch_page_for_answer()`
查页面、复用`llm_generate_answer()`生成回答——生成函数本来就是按"传入的
对象只要有.chunk.source_file/.chunk.text就能用"设计的（见llm_generate.py
说明），所以这里直接把整页正文包成一个`RetrievedPassage`复用，不用另写
一份生成逻辑。`_try_answer_from_kb()`（L3主入口）改成先试这条路由、没有
结果再落到本地知识库——"没有结果"覆盖三种情况（路由判断不是wiki、页面
没查到、生成失败），任何一种都不直接拒答，而是把机会让给本地知识库再试
一次：路由判断本身也可能出错，不能让它一步错就全盘拒答。

【防幻觉的适用范围：只管本地知识库和明日方舟wiki这两个领域，不该管全部对话】
这一节的结论演变了两轮，`wiki_router.py`模块开头有完整的过程记录，这里只
记落地后的结论。第一轮真机测试发现"你好"被当成检索性问题拒答，加了
route="chitchat"直接生成寒暄回复，同时保留route="none"专门处理"问了具体
问题但两个知识库都答不上来"的情况（比如"今天晚饭吃什么"），依然让它落到
本地知识库gate拒答。第二轮真机测试，用户指出这个区分本身站错了立场：这
个系统首先应该是个正常的聊天助手，能自然接住闲聊、请求建议这类根本不需要
"证据支撑"的内容；"检索不到就不瞎编"这条防幻觉原则该管的是"本地知识库/
明日方舟wiki这两个专门做了检索的领域"，不该被泛化成"任何答不出确切依据的
话都不能说"——"今天晚饭推荐吃什么"根本不存在什么需要验证的事实，拒答反而
是把系统能力边界收得比实际应该的更窄。

所以`route="chitchat"`和`route="none"`合并成了一个更宽的`route="general"`：
不属于这两个知识库领域的所有内容（寒暄、闲聊、请求建议、观点性问题、其他
通用问题）统一处理成"不检索、直接像普通聊天助手一样自然回应"，回复内容在
路由那一次LLM调用里顺带生成好。`_try_answer_from_router()`收到
`route="general"`直接返回`route["reply"]`——**这个改动完全没有触碰
`route="wiki"`和`route="local_kb"`这两条路径原有的置信度gate/拒答逻辑**，
一句明确落在wiki领域但查不到证据的问题（比如"明日方舟最新的干员是谁"）
依然会正常拒答，不会因为这次改动就被放宽成随便瞎猜——防幻觉收紧到它真正
该管的地方，不是被削弱。唯一需要如实记录的权衡：路由判断错误的后果变重了
——以前误判成`none`最坏结果是"多余的拒答"（安全但保守），现在误判成
`general`最坏结果是"该走防幻觉流程的问题被自由回答"，只能靠prompt里的
分类标准和示例把边界画清楚，没有更好的兜底手段，完整讨论见
`wiki_router.py`模块开头。

【L2语料该不该收录route="general"命中的发言：不该，已经排除】
真机测试暴露过一个具体案例：用户反复说过几次"吃了吗"这类寒暄，某一轮
L2的历史检索把之前那几句"吃了吗"翻了出来（BM25/语义相似度确实高——同样
的寒暄文字重复出现，检索分数天然就高），套上"【L2情景记忆——...引用来源
见下】"这套面向"翻旧账、给证据"设计的正式格式去回一句寒暄，跟内容本身
的轻松语气很不搭，是"回复风格前后不一致"这个体验问题的根源。

分析过两种不работ的修法：①把路由判断挪到L2检索之前，先过滤掉"general"
类内容——但这样会连累"我们组打算用什么做向量数据库来着"这类真正需要L2
翻旧账的问题，也被路由误判成"跟两个知识库都无关"直接丢给自由聊天，L2这个
已经反复验证过的真实能力反而被架空；②在L2生成回答之后，判断"这次生成的
内容是不是本质上在婉拒/其实就是聊天而不是引用证据"——但这个判断没有可靠
信号来源，容易做出一个同样脆弱的新规则。

真正的问题不在"L1->L2->L3这个优先级顺序"，也不在"L2的检索/生成逻辑本身"，
而在于L2检索语料的构成：`_search_history()`目前不加区分地把用户说过的
每一句话都当成"可能值得被翻出来的历史"去检索匹配，但寒暄/闲聊/请求建议
这类内容（也就是被route="general"接住的那些）根本不是"事实"，反复出现
纯粹是语言习惯，不代表"用户在问回去、想让我翻旧账"——这类内容不该出现在
L2的检索候选池里，就像`fetch_page_for_answer()`一开始就固定排除语音记录
等低价值小节一样，是"从语料源头把不该被检索到的内容筛掉"，而不是"检索到
之后再判断要不要用"。

具体做法：`conversation_turns`表新增`is_general_chat`列（默认0），
`user_says()`每轮结束时，如果这轮最终算general闲聊，就请`memory_store.py`
的`mark_turn_general_chat()`回头把这句用户发言标记上，`_search_history()`
在原有"只看role='user'"这条过滤之外，再加一条"排除is_general_chat=1的
发言"——这一版上线时是靠一个`self._last_turn_was_general_chat`跨方法临时
标记来实现的（`_try_answer_from_router()`确认route="general"时设成True，
`user_says()`处理完这轮后读一次），但这个实现方式后来被证明有个真实漏洞，
见下一节。

老实说这个修复一开始不是回溯性的：修复之前已经存在的历史轮次默认还是
`is_general_chat=0`，不会被追溯排除，只能靠L2本来就有的时间衰减
（DECAY_RATE=0.15）自然淡出候选池，或者用`chat_cli.py --reset`清空重来，
这个限制到目前为止依然成立、没有改变。

【上面这版标记机制的真实漏洞：L2抢答会让新发言也永远打不上标记，闲聊会
自我延续】
真机测试验证时发现：跨方法标志位这个实现，只有在route="general"真正被
调用到的时候才会触发标记——但`user_says()`原来的优先级是L1->L2->L3
（路由判断在L3内部，排在L2后面）。这意味着只要L2（`_search_history()`+
置信度gate）能从历史里翻到点什么、gate又通过了，它就会直接生成回答，
根本不会走到路由这一步，也就永远不会被标记。真机复现到的具体现象：
"吃了吗您"命中L2、引用了修复之前遗留的旧数据，生成了一句诚实但仍然套着
正式引用格式的回复；这句回复本身作为新的一轮用户发言存进去之后同样没被
标记（因为它是被L2接住的，不是被general接住的），于是紧接着再问一次
"吃了吗"，L2又把这句刚生成不久的发言当成"历史证据"翻了出来——闲聊不断
把自己新产生的内容喂给自己，标记机制形同虚设，不会随时间真正断掉。

分析过一个"不用改标记逻辑，直接在L2答完之后顺手再调一次route_query()"的
补救方案，也否决了：路由是四选一分类器（当时还是三选一：wiki/local_kb/
general），根本不知道"这是在问历史对话"这回事，像"我们组打算用什么做
向量数据库来着？"这种真正该被L2翻旧账答对的问题，字面上跟两个知识库都不
沾边，路由多半也会把它误判成general——这样反而会把本该被记住的真实内容
也标记成"闲聊"，将来自己都翻不到自己，制造一个更隐蔽的新回归。

真正区分"吃了吗"和"我们组打算用什么做向量数据库来着"的，不是问题长什么
样，而是"检索到的历史片段里到底有没有真实信息"——这需要先看到实际检索
结果才能判断，光靠一个通用分类器盲猜问题文本本身做不到。这就跟"L2该不该
排在路由前面"这个更根本的顺序问题绑在了一起，最终解法见下一节。

【最终解法：把"L2无条件先跑"改成"路由先四选一，选中history才跑L2"】
既然问题根源是"L2排在路由前面、抢答之后路由再也没机会判断"，那就不再
让L2无条件排在最前面，而是让`wiki_router.py`的`route_query()`加一个第4个
分类`route="history"`——专门识别"这句话看起来是在追问之前对话里的具体
内容"（"来着""我刚才说的""你还记得（某件事）"这类回指线索），一次性在
最前面判断清楚"这句话该谁答"：`route="history"`才去调用L2的
`_try_answer_from_history()`，`route="general"`直接用路由那次调用里顺带
生成好的回复，`route="wiki"`/`route="local_kb"`分别对接明日方舟wiki和本地
论文知识库。新增`_dispatch_with_router()`把这四条分支统一收在一起，返回
`(回复文本, 这一轮算不算general闲聊)`——general与否直接由这一次路由结果
决定，不再需要一个跨方法的临时标志位去事后猜测，`is_general_chat`标记
因此变得完全准确、不会再漏标。`_try_answer_from_kb()`拆成了
`_try_answer_from_local_kb()`（本地知识库，总是有确定回复）和
`_try_answer_from_wiki_page()`（wiki查询，失败时返回None、退回本地知识库
再试）两个更小的方法，各自职责更清楚。

这个重新设计也是主动接受了一个新的诚实权衡：以前L2永远排在最前面，"这是
不是在问历史"这件事有真正的检索结果撑腰，理论上不会平白无故猜错；现在
改成先猜分类、猜中"history"才去检索，如果一句真正该翻旧账的话在语言上
缺少"来着"这类明显的回指线索，有可能被误判成general，被自由聊天直接
回答、而不是老实去翻历史——这跟当初"route='general'扩大化"那次讨论
的风险性质是同一类（分类错了不再是"多余拒答"这种安全的失败模式，而是
"该走某条更严谨路径的问题被自由回答"），目前同样只能靠prompt里的分类
标准和示例尽量把边界画清楚，完整讨论见 wiki_router.py 模块开头。
route="history"判断对了但L2实际没能翻出可信内容时，回复固定的
`_HISTORY_MISS_REPLY`文案（"这个我这边没翻到相关的记录…"），不调用LLM
去凭空猜"之前说过什么"——避免在"看起来像是在翻旧账"这种更容易让人误以为
是真实记忆的场景下产生编造内容，风险比单纯的general闲聊更需要谨慎对待。

【后续追加：route="general"命中不再意味着"一定可以排除出L2候选池"】
上面的设计定下"general命中就标is_general_chat=True、排除出L2"这条规则时，
隐含假设是"能落进general的都是寒暄/闲聊/请求建议这类不值得被翻旧账的内容"。
真机测试又暴露一个反例：陈述句（比如"我们组决定用Chroma做向量数据库"）
以前因为general的prompt/示例只覆盖问句/寒暄，容易被误判进wiki/local_kb、
触发置信度gate拒答；修法是把陈述句也纳入general的识别范围，但陈述句和
真正的闲聊不是一回事——它是以后可能被翻旧账追问的具体事实，标成
is_general_chat=True会把它永久排除出L2候选池，是比"分类错了"更隐蔽也更
难补救的问题（排除掉就真的翻不回来了，不像误判general只是答错这一轮）。
最终解法：`wiki_router.py`的`route_query()`给route="general"额外加一个
`is_chitchat`布尔字段，由LLM顺带判断这一命中到底是真闲聊还是陈述新信息，
`_dispatch_with_router()`不再对general无条件返回True，改成读这个字段——
完整设计取舍（为什么默认给True、两种误判方向的后果谁更可接受）见
`wiki_router.py`模块开头"真机测试暴露的第5个问题"一节，不在这里重复。

【检索架构升级：`_dispatch_with_router()`之上新加了一层`_dispatch_with_
planner()`】
完整设计动机、"agentic能不能吃掉静态分解"这个问题的结论、为什么不做真正
的DAG调度、依赖怎么解析这几件事，全部记在`query_planner.py`模块开头，这里
只记本文件这边落地的部分。

`user_says()`调用的入口从`_dispatch_with_router()`换成了`_dispatch_with_
planner()`：先用`query_planner.decompose()`判断这句话要不要拆成多个先后
执行的子查询，绝大多数情况（不需要拆）直接原样委托给`_dispatch_with_
router()`，行为、返回值、prompt都跟没有这层编排之前逐字一致——这是新
功能"完全不影响现状"的退化路径。需要拆分时，新增的`_answer_multi_hop()`
按依赖顺序执行每个子查询（`_execute_sub_query()`），有依赖的子查询会拿到
它依赖的前面子查询的结果当背景（复用`route_query()`解析代词/指代的同一套
机制，不是另外发明占位符替换逻辑），独立的子查询互不影响；某一跳失败了，
如果后面有子查询依赖它，诚实终止整个流程（不让后续步骤带着缺失的前提
瞎猜），如果没有任何子查询依赖它，只影响这一跳自己，其它独立的子查询
继续跑，最后老实说明"这部分没查到"。

【`_try_answer_from_history()`/`_try_answer_from_local_kb()`/`_try_answer_
from_wiki_page()`返回值从裸字符串改成了`SubAnswer`】
这三个方法原来直接返回格式化好的字符串（或者None表示没查到/失败），现在
统一返回`query_planner.SubAnswer`（display/core/ok三个字段，定义和取舍见
该文件"single query plan"附近的说明）——单轮对话场景下调用方只用`.display`
字段，语义完全不变；多跳编排器额外用到`.core`/`.ok`，判断这一跳有没有查到
确定的东西、要不要把结果当背景喂给依赖它的下一步。三个方法同时新增了
`extra_context`参数，透传给`llm_generate.py`的生成步骤（默认None，不传时
行为不变）——这是HANDOFF"L0上下文窗口未接入路由判断"一节末尾记录的遗留
缺口里，"生成措辞"这一半的修复；"检索匹配质量"那一半（BM25/向量检索本身
用到的query文本要不要先做指代消解）由`query_planner.resolve_for_retrieval()`
负责，只在多跳编排器执行"有依赖"的子查询时才会被调用，不影响普通单轮
对话的检索行为，完整理由见`query_planner.py`模块开头。

`_dispatch_with_router()`顺带也把`recent_context`传给了这三个方法的
`extra_context`——这是在原有单轮对话路径上就能拿到的免费改进（不需要
额外的LLM调用，`recent_context`参数本来就已经传进来了），需要在真机测试
里跟其它改动一起验证没有引入回归。

【F（多候选页面消歧）：`_answer_multi_hop()`新增fan-out分支】
完整设计动机、策略选型（"都查一遍、合并回答" vs 反问用户 vs 只说明不消歧，
用户确认选了第一种）、范围限制全部记在`query_planner.py`模块开头"F正式
实现"一节，这里只记本文件这边落地的部分。

`_execute_sub_query()`新增`detect_candidates`参数（默认False，不传时跟
改动前逐字一致），只有`_answer_multi_hop()`判断"这个子查询的结果会被
后面的子查询依赖"时才会传True；为True且路由到wiki时，`_try_answer_from_
wiki_page()`会多做一次`query_planner.detect_answer_candidates()`调用，
核对wiki原文里是不是真的存在不止一个候选。本轮只在route="wiki"这条路径
接了检测——local_kb/history理论上也可能出现类似歧义，但目前唯一的真实
案例（"玛恩纳的侄女"）是wiki场景，没有必要现在就给另外两条路径也接上
同样的机制，等真出现了再处理。

`_answer_multi_hop()`执行完所有子查询后，检查有没有子查询的结果带着
超过一个候选（`SubAnswer.candidates`）——没有就跟改动前完全一样直接走
`synthesize()`；有的话交给新增的`_answer_multi_hop_with_fanout()`：对
直接依赖这个歧义子查询的每个下游子查询，把依赖注入的背景替换成"锁定为
这个候选"，各执行一次，再拼成一份"虚拟展开后的plan"直接复用`synthesize()`
合并成最终回答（不需要为这个场景另写一个合并prompt）。同一个plan里出现
不止一个子查询同时产生歧义时，为避免组合爆炸，退化成每个都只取第一个
候选、按原来的单分支流程处理，不崩溃、只是回到"不做消歧"的旧行为。

【F真机测试第一轮：三个真实发现，两个已修复，一个是范围认知的澄清】
用户自己设计了5组真机测试。①"玛恩纳的侄女的模组是什么"完整走通：正确
fan-out到瑕光/临光两个候选，各自的模组信息都答对，F这条主链路确认真的
work。②"能天使的同事的种族包含哪些"暴露了`_execute_sub_query()`那个
route_query()选错候选的真实bug（已在上面"F真机测试暴露的真实bug"一节
记录并修复），以及`MAX_CANDIDATES`截断掉了第4个候选"可颂"——这一点不是
bug，是设计取舍：F的fan-out机制本来是为"一个关系应该只有一个正确答案，
但生成时被非确定地挑中一个"这种场景设计的（比如"侄女"），"同事有哪些"
这类问题的候选数量在设计上就是开放的，跟固定实体数量的关系型问题不是
同一类，不该用同一个机制无限兜底——`MAX_CANDIDATES`从3调到6只是缓解
这次的具体案例，不是要把这个机制改造成能处理任意大小的枚举查询；真正
需要枚举/聚合能力的问题（"某类干员有哪些"这种），留给排在F后面的
"索引/聚合/动态更新"那一项处理，不在F的范围内继续加大这个数字硬撑。

③"陈的异格是什么职业"分两轮问（先问"陈有哪几个形态"，后问"异格是什么
职业"）时，`decompose()`能借助上一轮的L0历史直接把两个异格的具体名字
代入，生成两个独立子查询——这条路径本来就有效，不需要改动。④"W的异格是
什么职业"作为单独一轮问出时没有被拆分、编排层完全没有介入，`route_query()`
只能在"W"这一个页面的正文里连蒙带猜，能答对纯粹是运气（页面底部的干员
导航栏恰好也列出了异格干员的职业分类）——这是`decompose()`该识别"链式
依赖"却没识别出来的真实缺口，修复见`_DECOMPOSE_SYSTEM_PROMPT`新增的
"不要被自己的训练数据记忆抢跑"一节。

【画像可见性：⑥收窄`kind="question"`范围之后暴露的新缺口，及修复】
⑥那次修复本身是对的——但用户真机测试它自己举的负例"我导师推荐我读的是
哪篇文献，我读完了吗"时发现：`_l1_step()`正确判成`kind="other"`放行之后，
画像在整条`_dispatch_with_planner()`编排链路里彻底消失。`decompose()`/
`route_query()`/`_try_answer_from_history()`/`synthesize()`这些函数签名里
都没有`profile`参数，只有`recent_context`会被到处传递——这句话被拆成两跳
后，第一跳"我导师推荐我读的是哪篇文献？"被路由到`route="history"`翻L2
逐轮对话原文，但"导师推荐的文献是ABC、已读完"这个事实只存在于画像里
（很久以前一次陈述句被抽取写进去的，从未以完整发言形式单独出现过），L2
自然翻不到，两跳查询在第一跳就诚实终止，报告"没能找到可信的答案"——用户
真机第二次测试同一句话时，这次`decompose()`判定不拆分、走单查询路径
`route="history"`，`_search_history()`翻到了两条候选（用户自己反复问这
同一个问题留下的历史提问本身），置信度gate通过，`llm_generate_answer()`
诚实指出"这两条只是你自己重复问的同一个问题，没有答案信息"——同一句话
两次真机运行，暴露的是同一个根因在两条不同分支下的两种表现：画像里明明
有答案，却因为"L1判定不是简单问题"就整条链路都看不见它了。

**根因不是⑥这次收窄错了，是画像的出场方式本身有问题**：现在对画像的
处理更像是"猜中了才给你看一眼"（只在`_l1_step()`这一个前置短路口出场，
一旦被判定不是简单问题就完全不可见），而不是像L0的`recent_context`那样
"不管这句话问什么，先无条件带一份最近对话在身上"。参照业界core memory
的设计（Letta/MemGPT把"关于用户是谁"的画像当成始终在场的背景，不需要
检索/判断就能用，区别于需要显式检索的历史/知识库），这里补两条通路，
不是把`_l1_step()`改成到处调用：

1）**顺手注入`recent_context`**：`user_says()`新增`_build_planner_context()`，
在`_l1_step()`判定放行之后，给要传给`_dispatch_with_planner()`的背景里
追加一行打了"（用户画像，代表当前已确认的最新状态）"标记的画像文本，
沿着已有的`recent_context`/`extra_context`通道免费扩散到`decompose()`/
`route_query()`/生成步骤，不需要新开任何函数签名参数。只在这一条链路
使用，不传给`_l1_step()`（它已经直接拿到`profile`本身，不需要重复喂）。
`llm_generate.py`的生成prompt为这一行内容单独开了个例外——允许被当成
证据引用、且跟历史冲突时以画像为准，见该文件模块开头"画像可见性修复
新增"一节；`route_query()`/`decompose()`不需要改prompt，它们本来就只是
用背景帮助理解这句话在问什么，多一行画像描述不影响这条既有逻辑。

2）**编排链路的最后一道兜底**：`llm_l1.py`新增独立函数
`find_profile_answer()`（不是复用`llm_l1_turn()`，两者职责分开，见该文件
模块开头说明），只在`_try_answer_from_history()`判定"没有候选"或者"gate
没通过"、准备诚实放弃之前调用一次——这是run1那次真实失败（L2检索阶段就
返回空，生成措辞层面的改进完全救不了）需要的那道防线，"顺手注入
recent_context"只能覆盖"L2检索到点什么、但内容不够"这种run2式的失败
（靠生成步骤看到画像背景来补救），两种失败都需要各自的应对，缺一个都
补不全。正常情况下（L2/L3已经查到东西了）不会多花一次LLM调用，只在最
坏结果发生前多一道防线，跟`detect_answer_candidates()`"只在真正需要时
才加成本"是同一种克制。这道兜底只接在`_try_answer_from_history()`
——这个方法的职责本来就是"翻旧账"，画像是"用户说过的话"的一份已经去重/
整理过的摘要，语义上是这个方法最自然的兜底来源；`_try_answer_from_
local_kb()`（本地论文知识库领域）跟用户自身事实无关，不需要接这道兜底。

【这项修复已完成真机验证】沙盒结构性验证通过之后，用户在自己电脑上用
真实LLM（MiniMax）重新跑了`run_profile_visibility_demo.py`，两次原本失败
的问法（"我导师推荐我读的是哪篇文献，我读完了吗"/换一种问法）都正确用上
了画像信息，回归检查（本地知识库路径）也未受影响。`_try_answer_from_
profile_fallback()`本身在这次真机运行里没被直接触发（L2检索本身已经查到
了东西），仍只有沙盒验证覆盖，如实记录在这里，不夸大验证范围。

【⑳L0摘要接入编排层背景】起因是给这个系统画架构示意图时发现的一处死代码
疑点：`_maybe_update_l0_summary()`确实在写`memory_store.py`的`l0_summary`
表，但`get_session_summary()`只在一个demo脚本里被调用、`chat_cli.py`的
`/summary`也只是直接读表展示——`decompose()`/`route_query()`这些真正靠
"L0窗口+摘要"理解指代的判断，实际上只拿到了`recent_context`（原始窗口），
从来没读过这份摘要，即"写了但没有被任何真实回答路径读取"。

这不是刻意的取舍，是当初实现"顺手注入`recent_context`"（见上面"画像可见性"
一节）时只想着补画像这一块，没有把同样早就存在的L0摘要一起接上——找到
这个疑点，办法就是"把它接上"而不是"把它删掉"：摘要机制本来就是为了让
`decompose()`/`route_query()`这类判断能看到超出窗口大小的更早对话（否则
用户问起一件已经滑出`recent_context`、又还没被L2检索命中关键词的旧事，
这些判断会因为缺这段上下文而解不出指代），删掉只是回避问题，不是解决
问题。

具体做法：`_build_planner_context()`在原有"顺手注入画像"的基础上，同一
通道再顺手带上`get_session_summary()`（如果有），标成"仅用于理解指代、
不构成证据"，跟`recent_context`里的普通历史待遇一致——不需要碰
`llm_generate.py`的生成prompt（那条"唯一可引用例外"规则只认"用户画像"
这个标记，摘要行不会被误当成证据），也不需要改`_l1_step()`（`kind`
三分类判断的是这句话本身是不是画像陈述/画像问题，跟"这句话提到的旧事
是什么"关系不大，接摘要对它价值有限）。

真机验证结果：摘要生成→注入编排层背景→`route_query()`借助摘要正确做出
路由判断，三个环节都已确认生效（间接证据：纯代词提问"它现在进展到哪一
步了？"被正确分类成`route="history"`，回复文本是`_HISTORY_MISS_REPLY`，
只有分类判对了才会走到这条固定回复）。但最终检索仍然没有查到相关内容，
往下追查是一个新的、独立的缺口，见下面"㉑单查询代词解析缺口"一节。

【㉑单查询代词解析缺口：真机验证⑳时暴露】`route_query()`能借助⑳新接入
的摘要正确分类，但`_dispatch_with_planner()`的单查询分支（`decompose()`
判定不需要拆分时）以前直接把原始文本转交给`_dispatch_with_router()`，
从未调用过`resolve_for_retrieval()`把代词换成具体名字——所以"它"最终还
是以"它"的样子被拿去检索，跟"烛龙"毫无字面重合，自然查不到。

根因：`resolve_for_retrieval()`原有的调用前提是"子查询有`depends_on`"，
隐含假设"没有依赖＝这句话本身独立完整，不需要改写"。这个假设对"多跳
编排内部的独立子查询"成立，但对"用户这一整句话本身就是跨轮代词指代"
不成立——`decompose()`判断"不需要拆分"是对的，但不代表这句话检索前
不需要改写，这是两个不同的判断，之前的实现把它们当成了一回事。

修复：单查询分支转交`_dispatch_with_router()`之前，先调用一次
`resolve_for_retrieval(query, recent_context, self.llm_client)`，用改写
后的文本而不是原始`query`。复用的是跟多跳编排器完全相同的函数和安全网
（没有上下文/client不可用时原样返回，LLM失败/返回空同样原样返回），
`_RESOLVE_SYSTEM_PROMPT`本身要求"已完整就原样输出"，对regression场景
预期无影响。代价如实记录：这条分支多了一次LLM调用（`decompose()`+
`resolve_for_retrieval()`+`route_query()`），没有另外设计规则版触发
条件去省这次调用——理由跟`fact_extractor.py`一贯的态度一致，规则会
漏判，漏判的代价比多一次调用的延迟成本更值得避免。

已知残留缺口（这次没有一并处理）：多跳编排器内部（`_execute_sub_query()`）
"没有`depends_on`的独立子查询"仍然沿用"独立＝不用改写"的旧假设，同样
可能漏解代词——同一个根因、不同发生位置，目前没有真机场景触发过，留待
以后真遇到再处理。


【㉒/㉓：真机验证㉑时顺带发现的两个独立新问题（不是㉑那次改动引入的，
是`decompose()`/`_search_history()`两处各自独立存在的问题）】
真机验证㉑（重跑`它现在进展到哪一步了？`→`能天使是什么职业？`这组场景）
时，用户反馈"回答看着有点错位""怎么还在烛龙"，完整终端日志揪出两个各自
独立、都不是㉑那次改动引入的问题：

㉒（根因、修复、验证状态的完整记录在`query_planner.py`模块开头，因为
问题出在`_DECOMPOSE_SYSTEM_PROMPT`）：`decompose()`把"能天使是什么职业？"
拆成了两个子查询，其中q1是自己凭空生成的、追问上一轮那个没查到答案的
"烛龙进展"问题——用户这一轮根本没问这个。这次运气好没有污染最终回答，
但纯属僥幸，不是设计上的保证。这次改动只碰了`_DECOMPOSE_SYSTEM_PROMPT`
的文字，不涉及本文件。

㉓：同一份日志里再次确认了`_search_history(query, top_n=2)`固定凑够
`top_n`条返回、不看每条候选自己是否过线的问题——低分陪跑候选被当成
"引用来源"列进了回复。修复：排序之后、截断到`top_n`之前，先用跟
`_try_answer_from_history()`同一套及格线（`L2_RERANK_FLOOR`/`L2_BM25_
FLOOR`）逐条过滤掉没有单独过线的候选，见`_search_history()`方法本身
的注释。过滤后不足`top_n`条、甚至一条都不剩，不做特殊处理，直接如实
返回——一条都不剩时调用方会走"压根没检索到候选"那条已有的画像兜底
路径，效果等价。这是纯过滤逻辑改动，云端沙盒可以完整结构性验证。

【㉔：修复㉒/㉓之后重新真机验证，㉑/㉒都已确认生效，但顺带暴露出㉑本身
的修复方式选错了位置，已修正】用户按上面的说明重新跑了一遍
`run_l0_summary_context_demo.py`。好消息：㉑确认生效——"它现在进展到
哪一步了？"被正确改写关联上了"烛龙"；㉒确认生效——"能天使是什么职业？"
这一轮的诊断打印不再出现凭空拆出的额外子查询；㉓确认生效——"它现在
进展到哪一步了？"这一轮的"引用来源"只剩`[1]`，不再混入不相关的陪跑
候选。但完整终端日志里又出现了一个新问题：紧接在"我们实验室这周启动了
...烛龙..."这句话后面的纯闲聊"今天天气有点冷，中午吃了碗泡面。"，被
`resolve_for_retrieval()`整句改写成了上一句的内容（"我们实验室这周启动
了一个新的分析流程，内部代号叫「烛龙」，专门处理超长读长测序数据的
组装。"）——这句话本身完整、没有任何代词/省略指代，根本不需要改写，
`_RESOLVE_SYSTEM_PROMPT`的设计保证在这里失效了。

根因不在`resolve_for_retrieval()`的prompt本身，而在㉑第一版实现选错了
调用位置：`_dispatch_with_planner()`把这次调用放在路由判断**之前**，
对每一句单查询（不管闲聊、陈述句还是真正需要检索的问题）无差别生效。
但`resolve_for_retrieval()`唯一真正的消费者只有`_try_answer_from_
history()`/`_try_answer_from_local_kb()`这两个不经过LLM、只能做字面
检索匹配的方法——general路由的闲聊回复由`route_query()`自己生成，
根本用不上一份"改写过的检索文本"，之前的调用位置让它对general路由也
无差别生效。扩大了不该扩大的范围之后，又叠加了一个真实存在的prompt
弱点——模型在"一句朴素、跟上下文无关的当前输入"和"上下文里一句更突出、
更具体的旧话"之间，容易把两者搞混，最终生成`route_query()`的闲聊回复
时用的输入已经不是用户真正说的那句话（这次真机测试里"好的，这个我记下
了～"这个回复够通用、侥幸没有露出破绽，但底层输入已经错了）。

修复：把`resolve_for_retrieval()`的调用点从`_dispatch_with_planner()`
搬到`_dispatch_with_router()`里，改成路由判断**之后**，只在确定要走
route="history"/route="local_kb"这两条字面检索路径时才调用，
route="general"/route="wiki"两条分支完全不再触碰它。完整设计、代价
分析、已知未处理的边界见`_dispatch_with_router()`方法本身的文档字符串。
这也顺带把㉑第一版"每句单查询都多一次LLM调用"的成本收窄成"只有真的
路由到history/local_kb时才多一次"，比第一版更省。

【真机验证结果】用户重新跑了一遍`run_l0_summary_context_demo.py`，㉔
确认生效：紧跟在"烛龙"那句话后面的纯闲聊"今天天气有点冷，中午吃了碗
泡面。"这次没有再打印任何"检索文本已改写"的日志，回复也正常回应了这句
话本身（"哈哈，大冷天来碗热泡面确实舒服……"），不再绕着"烛龙"打转。
同一轮里"它现在进展到哪一步了？"被正确判成route="history"、`resolve_
for_retrieval()`把"它"完整改写成了"实验室这周新启动的那个超长读长测序
数据分析流程（代号「烛龙」）"，检索到`[1]`（只有一条，无陪跑候选，㉓
同时确认）并诚实回复"没有涉及具体进展"；"能天使是什么职业？"这一轮
`decompose()`也没有再凭空拆出额外子查询（㉒同时确认）。至此⑳-㉔这一整
条链路（含期间顺带修的㉒/㉓）全部真机验证通过。
"""

import os
import re
import threading
from datetime import datetime, timedelta, timezone
from concurrent.futures import FIRST_COMPLETED, wait
from typing import Dict, List, Optional, Tuple

from concurrency import parallel_enabled, submit
from timing import Trace, current_trace, span, timed
from length_limit import (count_units, exceeds_limit, parse_length_request, split_reply,
                          strip_length_request)

from chunking import Chunk
from sparse_retriever import BM25Index
from dense_retriever import TfidfDenseRetriever
from fusion import reciprocal_rank_fusion
from pipeline import MemorySystemPipeline, RetrievedPassage
from memory_store import MemoryStore
from fact_extractor import extract_facts, looks_like_question
from llm_l1 import PROFILE_DELETE, llm_l1_extract, find_profile_answer
from llm_l0_summary import llm_update_summary
from llm_generate import (
    LLM_KNOWLEDGE_LABEL,
    WEB_SEARCH_LABEL,
    compact_citations,
    is_no_answer,
    llm_answer_from_own_knowledge,
    llm_answer_from_web,
    llm_generate_answer,
)
from web_search_client import build_web_search
from llm_client import NoLLMClient
from bot_profile import OWNER_NAME, intro_message, is_owner
from wiki_router import route_query, disambiguate_wiki_candidates
from prts_wiki_client import fetch_page_for_answer
from prts_cargo_client import query_operators
from query_planner import (
    DecomposePlan,
    SubAnswer,
    decompose,
    detect_answer_candidates,
    resolve_for_retrieval,
    rewrite_for_history,
    rewrite_for_kb,
    synthesize,
)


# L1：识别"在问自己的画像"这类问题的规则，命中了就优先查 user_profile，
# 不命中就往下交给L2/L3——跟fact_extractor.py一样，只覆盖几种直白的问法，
# 换个问法（比如"你了解我吗"这种开放式问题）规则就识别不出来，这是规则化
# 方案本身的局限，如实记录，不夸大。
_PROFILE_QUESTION_PATTERNS = [
    (re.compile(r"我(?:叫|是)(?:什么|谁)"), ["称呼", "身份"]),
    (re.compile(r"研究方向.{0,6}(?:是什么|呢|吗)"), ["研究方向"]),
    (re.compile(r"我(?:的)?偏好.{0,4}(?:是什么|吗)|我喜欢什么"), ["偏好"]),
]

# 时间衰减率：隔一轮，权重乘以 (1 - DECAY_RATE)。这是一个工程经验初始值，
# 没有真实多轮对话数据验证过合不合适，跟阶段一 RERANK_FLOOR 刚接入精排时
# 一样——先给一个合理默认值让流水线跑起来，再用批量测试脚本的真实结果去
# 检验、调整。
DECAY_RATE = 0.15
L2_RERANK_FLOOR = 0.5
L2_BM25_FLOOR = 5.0
# 翻聊天记录改写成陈述句重查时的门槛（2026-10-01，见_retry_history_with_rewrites()）：
# 起初跟知识库改写路径取同一个值0.75；2026-10-02按run_l2_recall_eval.py（含同话题干扰）
# 降到0.65（用户定）：改写后正确记录最低0.66（"我买了……车"），同话题干扰最高0.37，
# 其它最高0.58（问"数据可视化"碰到"向量数据库"）。0.75挡掉了"买了什么车"隔5/20轮；
# 0.65跟0.58留了点余量。万一误放相近记录，后面还有生成步骤的可答性判断兜底（测试里
# 原问法放进来的0.66/0.79两条相近干扰，生成都正确识别了）。样本小，以后有数据再校准。
L2_REWRITE_RERANK_FLOOR = 0.65

# route="history"判断对了、但L2实际检索/gate没能翻出可信内容时的兜底回复
# ——固定文案，不是LLM生成，见"L2语料该不该收录route="general"命中的发言"
# 一节后续更新：宁可老实说"没翻到"，也不让LLM在没有真实历史证据的情况下
# 自由发挥去猜一个"之前说过什么"，那样风险跟没有防幻觉的general闲聊一样，
# 但"以为在翻旧账"的场景比单纯闲聊更容易让人误以为是真的记忆内容。
_HISTORY_MISS_REPLY = "这个我这边没翻到相关的记录，方便再跟我说一下吗？"

# 【2026-10-01】滚动摘要原来每条消息都整份重写一遍（真机13~31秒，摘要越长越慢，同一
# 对话的下一条要等它）。改成滑出窗口的轮次攒够这么多条（4问4答，用户定的"攒4轮"）
# 才合并一次；没合并的原文留在上下文窗口里，见get_context_window()
SUMMARY_BATCH_TURNS = 8
GROUP_SUMMARY_PREFIX = "群聊身份版本2\n"

# 需要查证的问题（general+needs_search），联网搜索和大模型自身知识都没能生成回答时
_LOOKUP_FAILED_REPLY = "这个问题需要查证，但这次查找和整理回答时模型接口没有正常响应，没能给出回答，请稍后再问一次。"


def _ago(created_at: Optional[str]) -> str:
    """给用户看的"多久之前"（L2生成失败时列资料来源用，用户2026-10-01定的说法）：
    1小时内"x分钟前"，1天内"x小时前"，一周内"x天前"，更早写日期（北京时间）。
    "第N轮"是给生成步骤看的内部编号，用户看不懂。"""
    from datetime import datetime, timedelta, timezone
    try:
        t = datetime.fromisoformat(created_at)
    except (TypeError, ValueError):
        return "之前"
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    secs = (datetime.now(timezone.utc) - t).total_seconds()
    if secs < 3600:
        return f"{max(1, int(secs // 60))}分钟前"
    if secs < 86400:
        return f"{int(secs // 3600)}小时前"
    if secs < 7 * 86400:
        return f"{int(secs // 86400)}天前"
    local = t.astimezone(timezone(timedelta(hours=8)))
    return f"{local.month}月{local.day}日"

# L2检索范围放开（跟业界一致：Letta的recall memory检索全部消息、Generative
# Agents全部存储只按重要性降权）之后，闲聊不再被排除，而是降权：检索分数
# 乘以这个系数。0.5是初始经验值，跟DECAY_RATE刚引入时一样，先让流水线
# 跑起来，再用真实多轮对话数据校准。完整设计见模块开头"L2检索范围放开"一节。
GENERAL_CHAT_WEIGHT = 0.5

# 助手回复里系统自己加的格式文字——检索前要去掉，否则这些每轮都出现的固定
# 措辞会互相匹配、干扰检索（这正是当初只检索用户发言的原因）。只匹配系统
# 自己的标签（【L1...】【L2...】【L3...】【未找到可靠信息】），不能把所有
# 【】都删掉：wiki正文里本身就有"【种族】鲁珀"这类有信息量的写法。
_ASSISTANT_HEADER_RE = re.compile(r"【(?:L\d[^】]*|未找到可靠信息)】")
_ASSISTANT_NOISE_LINE_RE = re.compile(
    r"^\s*(?:引用来源：\s*|\[\d+\]\s+\S.*|（这个问题拆成了\d+部分分别查证）\s*)$")
_CITATION_MARK_RE = re.compile(r"\[(?:\d+|画像)\]")
_NO_INFO_ASSISTANT_REPLIES = (
    "没有检索到置信度足够的相关内容，不做猜测性回答。",
    _HISTORY_MISS_REPLY,
)


def clean_assistant_text(text: str) -> str:
    """把一条助手回复清洗成只剩正文、适合拿去检索的样子。返回空字符串表示
    这条回复不含任何信息（纯拒答/没翻到这类固定文案），不该进入检索语料。"""
    text = _ASSISTANT_HEADER_RE.sub("", text)
    lines = [ln for ln in text.splitlines() if not _ASSISTANT_NOISE_LINE_RE.match(ln)]
    text = _CITATION_MARK_RE.sub("", "\n".join(lines)).strip()
    for no_info in _NO_INFO_ASSISTANT_REPLIES:
        text = text.replace(no_info, "").strip()
    return text


def _row_get(row, key):
    """sqlite3.Row 和普通 dict 都能用的取值；旧代码/测试构造的行可能没有新加的列。"""
    try:
        return row[key]
    except (KeyError, IndexError):
        return None


class ConversationSession:
    """一个用户的一次多轮对话会话。

    【群聊（阶段三）】conversation_id 不传时等于 user_id——私聊/chat_cli 的
    行为跟加这个参数之前完全一样。群聊时（用户2026-09-25定的方案，跟业界
    "短期上下文按对话场所、长期记忆按人"的做法一致）：
      - user_id = 群成员（带群号前缀，同一个人在不同群里是不同的人，免得
        一个群的画像带到另一个群）——只用于 L1 画像；
      - conversation_id = 群——对话记录、L0窗口、L0摘要、L2检索都按群，
        群友A刚问的问题是群友B提问的上下文，也能翻到群里别人问过的内容
        （群聊是公开场景）。
    群里一段对话有多个说话人，对话记录多存一列 speaker；给LLM看的文本里
    用户发言前面标上"[称呼]"（画像里有"称呼"就用，否则按第一次出现的顺序
    叫群友A/B/…）。摘要和L2索引是整个群共用、会存下来的，所以只用固定称呼、
    不用"你"；只有当场的规划背景里额外说明"当前提问的是谁"。
    """

    def __init__(self, kb_pipeline: MemorySystemPipeline, store: MemoryStore,
                 user_id: str, window_size: int = 6, llm_client=None, auto_intro: bool = False,
                 conversation_id: Optional[str] = None):
        self.kb_pipeline = kb_pipeline
        self.store = store
        self.user_id = user_id
        self.conversation_id = conversation_id or user_id
        self.is_group = self.conversation_id != self.user_id
        self.window_size = window_size  # L0：最近几轮算"当前上下文"

        # 不传就是2.1行为（占位版，is_real=False）——见模块开头"这个文件
        # 同时是2.1和2.2的入口"一节的说明。传入 build_llm_client() 的真实
        # 返回值才会真正走LLM路径。
        self.llm_client = llm_client or NoLLMClient()

        # 开场白（bot_profile.py）：这个用户第一次对话时，在回复前面加上自我
        # 介绍。默认关闭——批量测试脚本按回复开头判断来源（run_multiturn_tests.py
        # 的classify_source），开场白会干扰判断；chat_cli.py和以后的QQ接入层
        # 这种真正面向用户的前端才打开。
        self.auto_intro = auto_intro

        # 联网搜索（QQ群压力测试准备（二））：wiki/本地知识库查不到、或者路由
        # 判定是需要事实依据的客观问题时，先联网搜索、再退到大模型自身知识，
        # 每一级都标注来源，见 _answer_beyond_sources()。只有真实LLM可用时才
        # 用得上（搜索结果要靠LLM整理成回答），没接LLM就不初始化。
        self.web_search = build_web_search() if self.llm_client.is_real else None

        # 复用L3已经加载好的embedding模型实例（如果是真实embedding的话），
        # 见模块开头"必须避开的性能坑"的说明；kb_pipeline.dense 如果是
        # TfidfDenseRetriever（占位版），没有 .model 属性，getattr 会拿到 None，
        # L2检索时会自动退回现建现用的TF-IDF（语料很小，没有性能问题）。
        self._embedding_model = getattr(kb_pipeline.dense, "model", None)
        # 一条消息里并行的几个子问题可能同时翻旧账：Chroma索引的对账/查询在这把锁里
        # 做（HistoryVectorIndex在内存里维护"已写入哪些轮次"，同时对账会互相干扰）
        self._index_lock = threading.RLock()
        # user_says(defer_summary=True)时推迟到after_reply()再更新对话摘要
        self._summary_pending = False
        # 同一对话同时只跑一个摘要更新（qq_bot发完回复就放开对话锁、摘要在后台写，
        # 下一条消息可能在摘要写完之前进来，见_maybe_update_l0_summary()）
        self._summary_lock = threading.Lock()
        self._history_index = self._build_history_index()
        if self._history_index is not None:
            self._reconcile_history_index()

    def _build_history_index(self):
        """L2稠密检索的Chroma索引（接入Chroma之后新增，见chroma_index.py）。
        落盘的数据库 -> 本机索引目录（chroma_index.default_chroma_dir()，不在
        同步盘里；换电脑时按memory.db原件重建）；":memory:"测试数据库 -> 内存
        集合，进程结束即消失，不会往真实索引里写测试数据。

        返回None时 _dense_search() 走改动前的内存现算路径——两种情况：
        没有真实embedding模型（L3已经降级到TF-IDF，那边已经打印过降级提示，
        这里不重复）；或者Chroma本身不可用（这里打印降级提示）。
        """
        if self._embedding_model is None:
            return None
        from chroma_index import HistoryVectorIndex, announce_degrade
        db_path = self.store.db_path
        from chroma_index import default_chroma_dir
        persist_dir = None if db_path == ":memory:" else default_chroma_dir()
        try:
            return HistoryVectorIndex(
                self._embedding_model,
                getattr(self.kb_pipeline.dense, "model_name", "unknown"),
                persist_dir,
            )
        except Exception as e:
            announce_degrade(
                "对话历史检索（L2）", f"Chroma不可用（{type(e).__name__}: {e}）",
                "退回内存现算：结果不变，但每次翻旧账都要重新计算全部历史的向量",
            )
            return None

    @staticmethod
    def _history_index_rows(turns, names: Optional[Dict[str, str]] = None) -> List[Dict]:
        """L2检索语料：把SQLite的对话记录转成检索用的行——助手回复先用
        clean_assistant_text() 去掉系统自己加的格式文字，清洗后为空（纯拒答/
        没翻到这类固定文案）的直接不进语料。BM25（_search_history()）和Chroma
        索引（_reconcile_history_index()）共用这一个函数，保证两路检索看到的
        是同一份文本。

        助手回复跟它回答的那句用户提问拼成一条（"（你当时问：……）我回答：……"），
        跟LangChain VectorStoreRetrieverMemory把一问一答存成一条是同一个做法。
        原因是真机验证撞出来的："你上次推荐我晚饭吃什么来着？"跟当时的提问
        "今天晚饭吃什么？"很像（精排0.741），跟助手那段具体菜单反而不像（0.215，
        另一次措辞不同时约0.69）——助手回复单独存一条，能不能翻到取决于当时
        怎么措辞；带上问题之后，追问靠问题匹配上，答案内容跟着一起出来。用户
        自己的发言仍然单独一条（那是用户陈述的事实本身）。"""
        rows = []
        last_user_text, last_asker, last_speaker = "", "你", None
        for r in turns:
            speaker = _row_get(r, "speaker")
            if r["role"] == "user":
                text = last_user_text = r["text"]
                last_speaker = speaker
                if names and speaker in names:
                    # 群聊：用户发言带上说话人（固定称呼，见类说明）
                    last_asker = names[speaker]
                    text = f"[{last_asker}] {text}"
            else:
                text = clean_assistant_text(r["text"])
                if text and last_user_text:
                    text = f"（{last_asker}当时问：{last_user_text}）我回答：{text}"
            if text:
                rows.append({"turn_index": r["turn_index"], "role": r["role"], "text": text,
                             "is_general_chat": int(r["is_general_chat"]), "speaker": speaker,
                             "asker": last_speaker if r["role"] != "user" else speaker})
        return rows

    # ---------- 群聊：说话人标注 ----------

    def _speaker_names(self, turns) -> Dict[str, str]:
        """群聊：说话人 -> 群里的固定称呼。私聊返回空字典（不标注）。
        优先级：主人本人（按账号ID绑定，见bot_profile.is_owner()）→ 画像里的"称呼"
        （群友自己说过"叫我…"，是本人明确的意愿）→ 平台给的昵称（QQ群消息里的
        username，实测是QQ账号昵称，qq_bot.py每条消息刷新）→ 按第一次出现的顺序
        叫群友A/B/…。"""
        if not self.is_group:
            return {}
        order = []
        for r in turns:
            sp = _row_get(r, "speaker")
            if r["role"] == "user" and sp and sp not in order:
                order.append(sp)
        if self.user_id not in order:
            order.append(self.user_id)
        names = {}
        # 编号按所有说话人第一次出现的顺序排（有名字的人也占号）：这样某个人后来有了
        # 名字，其他人的编号不会跟着变——摘要和L2索引里存的是当时的称呼，编号漂移会让
        # 前后对不上。代价是可能出现"只有一个没名字的人却叫群友C"，能拿到QQ昵称后很少见
        for i, sp in enumerate(order):
            if is_owner(sp):
                names[sp] = OWNER_NAME
                continue
            nick = self.store.get_profile(sp).get("称呼") or self.store.get_display_name(sp)
            names[sp] = nick or (f"群友{chr(65 + i)}" if i < 26 else f"群友{i + 1}号")
        return names

    def _labeled(self, rows, names: Dict[str, str]) -> List[Dict]:
        """群聊：给用户发言的文本前面标上"[称呼]"，其余字段原样保留。私聊原样返回。"""
        if not names:
            return rows
        labels = self._participant_labels(names)
        all_turns = self.store.get_all_turns(self.conversation_id)
        recipients = {}
        speaker = None
        for turn in all_turns:
            if turn["role"] == "user":
                speaker = _row_get(turn, "speaker")
            elif turn["role"] == "assistant":
                recipients[turn["turn_index"]] = speaker
        out = []
        for r in rows:
            d = {k: r[k] for k in r.keys()}
            if r["role"] == "user" and d.get("speaker") in names:
                d["text"] = f"[{labels[d['speaker']]}] {d['text']}"
            elif r["role"] == "assistant" and recipients.get(r["turn_index"]) in labels:
                d["text"] = f"[助手回复给{labels[recipients[r['turn_index']]]}] {d['text']}"
            out.append(d)
        return out

    @staticmethod
    def _participant_labels(names):
        import hashlib
        return {speaker: f"群友-{hashlib.sha256(speaker.encode('utf-8')).hexdigest()[:10]}（称呼：{name}）"
                for speaker, name in names.items()}

    def _usable_summary(self):
        row = self.store.get_l0_summary(self.conversation_id)
        if row and self.is_group and not row["summary"].startswith(GROUP_SUMMARY_PREFIX):
            return None
        return row

    @timed("聊天记录写索引")
    def _reconcile_history_index(self) -> bool:
        """让Chroma里这个用户的记录跟SQLite原件一致，见chroma_index.py模块开头。
        失败只影响这一次：打印降级提示，本次检索走内存现算，下次调用再试。"""
        try:
            with self._index_lock:
                turns = self.store.get_all_turns(self.conversation_id)
                rows = self._history_index_rows(turns, self._speaker_names(turns))
                self._history_index.reconcile(self.conversation_id, rows)
            return True
        except Exception as e:
            from chroma_index import announce_degrade
            announce_degrade(
                "对话历史检索（L2）", f"Chroma同步失败（{type(e).__name__}: {e}）",
                "这一次退回内存现算，下次会重新尝试同步",
            )
            return False

    # ---------- L0：会话上下文窗口 ----------

    def get_context_window(self):
        """最近 window_size 轮，模拟"不用检索、直接在上下文里"的部分。
        2.1没有生成模型，这个窗口在2.1下只用于 cli.py 展示"当前还记得的最近
        几句"；2.2下 `user_says()` 还会把它传给 `_l1_step()` -> `llm_l1_turn()`
        辅助L1判断——见模块开头"L0上下文窗口接入L1判断"一节。
        """
        # 【2026-10-01攒几轮再摘要】滑出窗口、还没并进摘要的轮次也留在上下文里（摘要+
        # 缓冲区，跟LangChain的ConversationSummaryBufferMemory同一个思路），攒够
        # SUMMARY_BATCH_TURNS条才合并一次，中间不会有"既不在窗口也不在摘要里"的空档。
        # 上限window_size+2*批量：摘要一直失败时不让上下文无限变长。
        existing = self._usable_summary()
        summarized_up_to = existing["summarized_up_to"] if existing else -1
        recent = self.store.get_recent_turns(self.conversation_id, limit=self.window_size + 2 * SUMMARY_BATCH_TURNS)
        unsummarized = sum(1 for r in recent if r["turn_index"] > summarized_up_to)
        limit = min(max(self.window_size, unsummarized), len(recent))
        rows = recent[len(recent) - limit:]
        if not self.is_group:
            return rows
        return self._labeled(rows, self._speaker_names(self.store.get_all_turns(self.conversation_id)))

    def get_session_summary(self) -> Optional[str]:
        """L0窗口滑出去的内容的滚动摘要，见 _maybe_update_l0_summary()。
        2.1（NoLLMClient）下永远返回None——2.1从不写入l0_summary表，这里
        如实返回"没有"，不伪造一个空字符串假装"有摘要但是空的"。
        """
        row = self._usable_summary()
        return row["summary"] if row else None

    def _maybe_update_l0_summary(self) -> None:
        """L0窗口每往前滑一轮，检查是否有新滑出窗口、还没并入摘要的内容，
        有的话用LLM滚动合并进摘要。只有真实LLM可用时才做，2.1下这个方法
        整体是空操作——完整设计见 llm_l0_summary.py 模块开头的说明。

        这一步排在`user_says()`最后（本轮回复已经算完、存完历史之后）才
        执行，失败与否不影响本轮已经生成、已经返回给用户的那句回复——
        完整背景见2026-09真机排查"苇草的异格"时发现的一处误诊：`llm_
        update_summary()`因为内容安全分类器拒绝而失败，被误以为是
        `synthesize()`合并步骤在编答案，两次失败的日志长得几乎一样，
        当时`llm_update_summary()`那次调用还没标`caller`，才会被混淆，
        见`query_planner.py`的`synthesize()`文档字符串。

        **失败时是"这轮没刷新，沿用旧摘要"，不是"这一段对话被丢弃"**：
        `summarized_up_to`这次没有推进，`new_rows`覆盖的这些轮次原始
        文本本身早就已经存进`self.store`了（`append_turn()`是每轮无条件
        执行的，不依赖LLM）——不会丢，下一次`_maybe_update_l0_summary()`
        被调用时会连同新滑出窗口的部分一起重新尝试摘要。真正的影响仅限于：
        如果用户这时候问起一件已经滑出L0窗口、又还没被摘要收进去的旧
        事情，`decompose()`/`route_query()`等靠"L0窗口+摘要"理解上下文的
        判断可能因为缺这段摘要而解析不出指代——L2（情景记忆检索）本身
        直接搜索`self.store`里的完整历史，不经过这份摘要，不受影响；L1
        画像也是独立的一套结构，同样不受影响。
        """
        if not self.llm_client.is_real:
            return
        # 上一次摘要还在后台写：这次跳过，没合并的轮次留在上下文窗口里，下次再合并
        if not self._summary_lock.acquire(blocking=False):
            return
        try:
            self._update_l0_summary_locked()
        finally:
            self._summary_lock.release()

    def _update_l0_summary_locked(self) -> None:
        all_turns = self.store.get_all_turns(self.conversation_id)
        if len(all_turns) <= self.window_size:
            return  # 还没溢出窗口，没有需要摘要的内容
        all_turns = self._labeled(all_turns, self._speaker_names(all_turns))  # 群聊：摘要里写清是谁说的

        existing = self._usable_summary()
        summarized_up_to = existing["summarized_up_to"] if existing else -1
        old_summary = existing["summary"] if existing else None

        # 已经滑出当前窗口、但还没并入摘要的那些轮次——滚动摘要每次只处理
        # "新滑出去的这一小段"，不重新摘要已经压缩过的旧内容，见模块开头
        # llm_l0_summary.py 的说明。
        overflow_cutoff = len(all_turns) - self.window_size
        new_rows = [r for r in all_turns[:overflow_cutoff] if r["turn_index"] > summarized_up_to]
        if len(new_rows) < SUMMARY_BATCH_TURNS:
            return  # 攒够一批再合并（这些轮次在get_context_window()里仍以原文出现）

        new_turns_text = "\n".join(f"{r['role']}：{r['text']}" for r in new_rows)
        updated = llm_update_summary(old_summary, new_turns_text, self.llm_client)
        if updated is not None:
            if self.is_group:
                updated = GROUP_SUMMARY_PREFIX + updated.removeprefix(GROUP_SUMMARY_PREFIX)
            self.store.set_l0_summary(self.conversation_id, updated, new_rows[-1]["turn_index"])
        else:
            # 明确打出"这轮没刷新"这件事本身，不要让它安静地被吞掉——
            # 上面刚发生的`[llm_client] ... 调用失败`日志本身不会说明
            # "所以接下来会怎样"，这行是专门补这一环的。
            print("[conversation_memory] 本轮L0滚动摘要没能刷新（沿用旧摘要），"
                  "不影响本轮已经生成的回答；这批未摘要的对话会在下次滑窗时"
                  "重新尝试摘要，不会丢失。")

    # ---------- L1：核心记忆 ----------

    def _build_planner_context(self, recent_context):
        """画像可见性修复——给要传入`_dispatch_with_planner()`的背景顺手
        注入一行画像描述，见模块开头"画像可见性"一节。只在这一条链路使用，
        不传给`_l1_step()`（它已经直接拿到`profile`本身，重复喂一份没有
        意义）。

        【⑳L0摘要接入编排层背景】同一个通道现在还顺手带上L0滚动摘要（如果
        有）——完整设计动机见模块开头"⑳L0摘要接入编排层背景"一节。摘要行
        不带"用户画像"那个特殊标记，llm_generate.py不会把它当成可引用的
        证据，跟recent_context里的普通历史待遇一致：只帮LLM理解指代，不
        能被当成回答依据。顺序上摘要在最前（比recent_context更早的对话），
        画像仍在最后（代表当前最新状态）。

        画像和摘要都为空时原样返回`recent_context`，不引入多余的标记行。
        """
        summary = self.get_session_summary()
        profile = self.store.get_profile(self.user_id)
        group_note = None
        if self.is_group:
            names = self._speaker_names(self.store.get_all_turns(self.conversation_id))
            labels = self._participant_labels(names)
            me = labels[self.user_id]
            directory = "；".join(labels.values())
            group_note = {"role": "system", "text": (
                f"（系统提供的参与者名册：{directory}。这些称呼确实可见，不要声称看不到昵称。"
                f"编号用于区分身份，即使昵称相同也不能合并；助手发言前注明了回复对象。）"
                f"（这是QQ群聊：多位群友轮流@你，最近对话里每条user发言前的方括号是说话人。"
                f"当前提问的是【{me}】——回答时对当前提问的人说\"你\"，提到其他群友用方括号里的称呼（不要带方括号，也不要用\"他\"\"她\"，不知道性别）。"
                f"下面的用户画像只属于【{me}】）")}
        owner_note = None
        if is_owner(self.user_id):
            # 绑定过的主人本人在说话：第一人称就是长安（用户2026-09-28要求绑定）
            owner_note = {"role": "system", "text": (
                f"（当前跟你说话的就是{OWNER_NAME}本人——开发这个机器人的人。TA说的\"我\"\"我的课题\""
                f"\"我的开题报告\"\"我讲过的JC\"，指的就是{OWNER_NAME}的，本地知识库里{OWNER_NAME}的资料"
                f"就是TA自己的资料，检索时把\"我\"理解成{OWNER_NAME}）")}
        if not summary and not profile and not group_note and not owner_note:
            return recent_context

        rows = [r for r in (group_note, owner_note) if r]
        if summary:
            print(f"[conversation_memory] L0摘要接入编排层背景：{summary!r}")
            rows.append({
                "role": "system",
                "text": f"（更早对话的滚动摘要，仅用于理解指代，不构成证据）{summary}",
            })
        rows.extend(recent_context or [])
        if profile:
            profile_text = "；".join(f"{k}：{v}" for k, v in profile.items())
            rows.append({
                "role": "system",
                "text": f"（用户画像，代表当前已确认的最新状态）{profile_text}",
            })
        return rows

    def _try_answer_from_profile_fallback(self, query: str) -> Optional[SubAnswer]:
        """编排链路快要判定"没有查到可信的答案"之前的最后一道画像兜底，
        只被`_try_answer_from_history()`的两处give-up分支调用，见模块开头
        "画像可见性"一节第2点的完整设计动机。

        故意不经过`llm_generate_answer()`重新生成措辞——画像内容本身已经
        是确定、简明的事实，直接展示比"再让LLM读一遍、重新组织语言"更
        准确，这跟`_try_answer_from_cargo_filter()`"不用LLM重新表述确定性
        结果"是同一个原则。

        返回None表示画像里确实没有字段答得上这句话——调用方应该维持原有
        "没有查到可信的答案"这个结论，不引入新的失败模式。
        """
        profile = self.store.get_profile(self.user_id)
        hits = find_profile_answer(query, profile, self.llm_client)
        if not hits:
            return None
        lines = [f"{k}：{v}" for k, v in hits.items()]
        display = "【L1核心记忆——你之前告诉过我（编排层兜底）】\n" + "\n".join(lines)
        core = "；".join(lines)
        return SubAnswer(display=display, core=core, ok=True, source="profile", refs=["你之前告诉过我的信息"])

    def _try_answer_from_profile(self, query: str) -> Optional[str]:
        for pattern, keys in _PROFILE_QUESTION_PATTERNS:
            if pattern.search(query):
                profile = self.store.get_profile(self.user_id)
                hits = [(k, profile[k]) for k in keys if k in profile]
                if hits:
                    lines = [f"{k}：{v}" for k, v in hits]
                    return "【L1核心记忆——你之前告诉过我】\n" + "\n".join(lines)
                # 命中了"像是在问画像"这个句式，但画像里还没有对应字段——
                # 不在这里瞎猜，交给L2/L3继续尝试，都没有再统一走"未找到"
                return None
        return None

    def _l1_write(self, text: str, recent_context=None) -> List[Tuple[str, str, Optional[str]]]:
        """L1只负责"写"（2026-09-29改版，用户选的"记忆写入和回答分开"，见llm_l1.py
        模块开头）：抽出这句话里要记进画像的信息并写入，不决定这句话怎么回答——
        回答每条都照常走编排层，画像作为背景交给路由/生成（"我是谁"由路由读着画像
        直接答）。改版前这里是三选一分类器，陈述句/画像问题会在这里当场截住整条
        消息，"说自己的信息+顺带一个问题"的后半句就没人回答了。

        返回实际写入的 [(key, 新value, 旧value或None)]，值没变的不算，调用方据此在
        回复末尾加"（已记下：…）"。LLM不可用/调用失败/格式不对时退回2.1的正则抽取
        （extract_facts）——同样只写不答。
        """
        profile = self.store.get_profile(self.user_id)
        if self.is_group:
            names = self._speaker_names(self.store.get_all_turns(self.conversation_id))
            label = self._participant_labels(names)[self.user_id]
            recent_context = [{"role": "system", "text": f"当前消息属于{label}。只能抽取此人明确陈述的个人事实，不能迁移其他人的事实。"}] + list(recent_context or [])
        updates = llm_l1_extract(text, profile, self.llm_client, recent_context=recent_context)
        if updates is None:
            updates = extract_facts(text)
        written = []
        for key, value in updates:
            if value == PROFILE_DELETE:  # 这一项全部不成立了（见llm_l1.py："不喜欢"里唯一一项被移走等）
                if key in profile:
                    if self.store.delete_profile_fact(self.user_id, key) is not False:
                        written.append((key, PROFILE_DELETE, profile[key]))
                continue
            if profile.get(key) == value:
                continue
            from profile_facts import supported_profile_fact
            if not supported_profile_fact(key, value, text):
                print(f"[conversation_memory] 拒绝无依据画像字段：{key}")
                continue
            if self.store.upsert_profile_fact(self.user_id, key, value, source_text=text) is not False:
                written.append((key, value, profile.get(key)))
        return written

    @staticmethod
    def _memory_note(written) -> str:
        """回复末尾的"（已记下：…）"：由代码根据实际写入的结果生成，不交给LLM说
        （LLM说"记住了"但写入失败，用户不会知道）——跟ChatGPT"记忆已更新"的提示一样，
        是系统告诉用户的，不是回答的一部分，所以不存进对话历史。"""
        parts = [f"{k}已清空" if v == PROFILE_DELETE else f"{k}改为{v}" if old else f"{k}：{v}"
                 for k, v, old in written]
        return f"（已记下：{'；'.join(parts)}）"

    def _l1_step(self, text: str, recent_context=None) -> Optional[str]:
        """2.1规则版L1（LLM不可用时的legacy链路专用）：正则抽取+正则识别画像问题，
        能答就当场返回。LLM可用时不再走这里，见`_l1_write()`。"""
        new_facts = extract_facts(text)
        written = []
        for key, value in new_facts:
            if self.store.upsert_profile_fact(self.user_id, key, value, source_text=text) is not False:
                written.append((key, value))

        reply = self._try_answer_from_profile(text)
        if reply is None and written and not looks_like_question(text):
            lines = [f"{k}：{v}" for k, v in written]
            reply = "【L1核心记忆——已记住】\n" + "\n".join(lines)
        return reply

    # ---------- L2：情景记忆 ----------

    def _dense_search(self, query: str, chunks: List[Chunk], top_k: int):
        """chunks 的 chunk_id 就是 turn_index（见 _search_history()）。
        优先查Chroma索引；Chroma不可用/这次同步失败时退回下面的内存现算，
        两条路径返回形状一样：[(chunks里的位置, 余弦相似度)]。"""
        if self._history_index is not None and self._reconcile_history_index():
            try:
                with self._index_lock:
                    return self._history_index.search(
                        self.conversation_id, query, {c.chunk_id: pos for pos, c in enumerate(chunks)}, top_k)
            except Exception as e:
                from chroma_index import announce_degrade
                announce_degrade(
                    "对话历史检索（L2）", f"Chroma查询失败（{type(e).__name__}: {e}）",
                    "这一次退回内存现算",
                )
        if self._embedding_model is not None:
            texts = [c.text for c in chunks]
            doc_emb = self._embedding_model.encode(texts, normalize_embeddings=True)
            query_emb = self._embedding_model.encode([query], normalize_embeddings=True)[0]
            sims = doc_emb @ query_emb
            ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
            return ranked[:top_k]
        # 没有真实embedding模型：退回TF-IDF，对话历史语料很小，现建现用没有
        # 性能问题（不像embedding模型那样有加载耗时）
        return TfidfDenseRetriever(chunks).search(query, top_k=top_k)

    def _history_source_label(self, row, names, when) -> str:
        """聊天记录片段的来源标签，生成步骤据此区分"你说的"和"我（助手）回答的"，
        也会出现在给用户看的引用来源里。2026-10-01从"第N轮你说的话"改成"x分钟前
        你说的话"：轮次编号是内部编号，用户看不懂（用户定的说法）。

        【2026-10-02重跑主人绑定测试】群聊里助手回答别人的话要在标签上写明回答的是谁
        （"我（助手）回答长安的话"）：最近窗口当证据（_recent_window_passages()）不过
        门槛之后，群友问"我的开题报告"会读到助手刚回答长安的那条，只有正文开头的
        "（长安当时问：……）"不够醒目，生成步骤把它说成了"你的开题报告"。"""
        ago = _ago(when.get(row["turn_index"]))
        if row["role"] != "user":
            asker = row.get("asker")
            if names and asker and asker != self.user_id and asker in names:
                return f"{ago}我（助手）回答{names[asker]}的话"
            return f"{ago}我（助手）的回答"
        if names and row["speaker"] != self.user_id and row["speaker"] in names:
            return f"{ago}{names[row['speaker']]}说的话"  # 群聊：别的群友说的
        return f"{ago}你说的话"

    def _recent_window_passages(self) -> List[RetrievedPassage]:
        """最近对话窗口里的发言（不含当前这句），作为翻聊天记录时的证据。

        【2026-10-01真机】"我不喜欢游泳，只喜欢骑行"刚说完两轮，问"我不喜欢什么"
        答成"没翻到"：检索确实把那句排在第一，但"问句对陈述句"精排只有0.38，没过
        0.5的门槛；最近对话虽然也传给了生成步骤，但只作为"理解指代用、不能当证据"
        的背景。这条规则对知识库/wiki是对的（聊天内容不能代替资料原文），对"我之前
        说过什么"不对——最近几轮本身就是聊天记录，而且是最新最可靠的那部分。跟之前
        "画像那一行可以当依据"是同一个思路（用户指出的）。所以这几条不经过检索门槛
        直接作为候选证据，是否答得上由生成步骤的可答性判断（NO_ANSWER）决定。"""
        all_turns = self.store.get_all_turns(self.conversation_id)
        if len(all_turns) <= 1:
            return []
        window_ids = {r["turn_index"] for r in self.get_context_window()}
        names = self._speaker_names(all_turns)
        when = {r["turn_index"]: r["created_at"] for r in all_turns}
        rows = [r for r in self._history_index_rows(all_turns[:-1], names) if r["turn_index"] in window_ids]
        return [RetrievedPassage(chunk=Chunk(chunk_id=r["turn_index"],
                                             source_file=self._history_source_label(r, names, when),
                                             text=r["text"]),
                                 rrf_score=0.0, bm25_score=0.0, dense_score=0.0)
                for r in rows]

    def _personal_history(self, query):
        if re.search(r"我的(?:朋友|父母|妈妈|爸爸|同事|家人|同学|老师|导师)|(?:谁|哪位|哪个)(?:问过|说过|群友)", query):
            return False
        return self.is_group and bool(re.search(r"我是谁|我的|我(?:之前|上次|以前|曾经|说过|喜欢|不喜欢|讨厌)|(?:给我|向我|为我)推荐", query))

    def _filter_personal_passages(self, query, passages):
        if not self._personal_history(query):
            return passages
        turns = self.store.get_all_turns(self.conversation_id)
        rows = self._history_index_rows(turns[:-1], self._speaker_names(turns))
        own_ids = {row["turn_index"] for row in rows
                   if (row.get("asker") if row["role"] == "assistant" else row.get("speaker")) == self.user_id}
        return [passage for passage in passages if passage.chunk.chunk_id in own_ids]

    @timed("聊天记录检索")
    def _search_history(self, query: str, top_n: int = 2, personal_only=False) -> List[RetrievedPassage]:
        """在这个用户"当前这句话之前"的历史里做混合检索+时间衰减。
        故意排除掉刚追加进去的这一轮（当前问题本身），否则它会跟自己完全
        匹配、分数虚高，压过所有真正的历史片段。

        检索范围：用户和助手的全部历史发言，闲聊也在内（只是降权，乘以
        GENERAL_CHAT_WEIGHT）——跟业界做法一致，见模块开头"L2检索范围放开"
        一节。早期只检索role="user"且排除闲聊，原因有两个：助手回复里的
        固定格式文字（"【L1核心记忆——...】"、引用来源列表）会互相干扰匹配
        ——现在检索前用 clean_assistant_text() 去掉；闲聊会自我延续——那是
        L2抢在路由前面回答导致的，四选一路由之后根因已经不存在。
        来源标签区分"你说的话"和"我（助手）的回答"，生成步骤据此把助手
        说过的话表述成"我之前回答过"，不当成用户说过的事实。

        返回值不保证一定凑够top_n条：排序之后会先用跟`_try_answer_from_
        history()`同一套及格线逐条过滤掉没有单独过线的候选，过滤后不足
        top_n条、甚至一条都不剩，都直接如实返回，不做"至少留一条"的特殊
        处理，见模块开头"㉓"一节。
        """
        query = strip_length_request(query)  # "50字以内"这类要求不参与检索，见length_limit.py
        all_turns = self.store.get_all_turns(self.conversation_id)
        if len(all_turns) <= 1:
            return []
        names = self._speaker_names(all_turns)
        history_rows = self._history_index_rows(all_turns[:-1], names)
        if personal_only or self._personal_history(query):
            history_rows = [row for row in history_rows
                            if (row.get("asker") if row["role"] == "assistant" else row.get("speaker")) == self.user_id]
        if not history_rows:
            return []

        when = {r["turn_index"]: r["created_at"] for r in all_turns}

        def source_label(row):
            return self._history_source_label(row, names, when)

        history_chunks = [
            Chunk(chunk_id=row["turn_index"], source_file=source_label(row), text=row["text"])
            for row in history_rows
        ]
        is_general_chat = {row["turn_index"]: bool(row["is_general_chat"]) for row in history_rows}
        # 时间衰减要按"隔了几轮用户发言"算距离，不能直接拿 turn_index 相减——
        # turn_index 是 user/assistant 共用的计数器，一问一答就会跳2，直接相减
        # 会让衰减速度变成设计意图的2倍多（真实多轮测试里实测到过：本该"隔2轮"
        # 只掉到约0.72倍权重，实际因为把assistant的回复也算进了间隔，变成"隔6轮"，
        # 掉到约0.38倍，一个原本该被正确翻出来的历史发言被判成"分数不够、不存在"，
        # 详见README"阶段二2.1"一节的踩坑记录）。这里改用"这句话是历史里第几句
        # 用户发言"这个位置来算距离，只数用户发言，不数助手回复。
        # 助手回复跟它所回答的那句用户发言算同一个位置（它之前有几句用户
        # 发言、减1），这样一问一答两条的衰减权重一样；闲聊的用户发言也
        # 计入距离（放开检索范围之前闲聊被整条排除，不参与计数）。
        user_turn_position = {}
        users_seen = 0
        for row in all_turns[:-1]:
            if row["role"] == "user":
                users_seen += 1
            user_turn_position[row["turn_index"]] = max(users_seen - 1, 0)
        current_user_position = users_seen

        bm25 = BM25Index(history_chunks)
        top_k = min(10, len(history_chunks))
        bm25_ranking = bm25.search(query, top_k=top_k)
        dense_ranking = self._dense_search(query, history_chunks, top_k=top_k)
        bm25_raw = dict(bm25_ranking)
        dense_raw = dict(dense_ranking)
        fused = reciprocal_rank_fusion([bm25_ranking, dense_ranking], k=60)

        # 注意：fused 里的下标是"history_chunks 列表里的位置"（BM25Index/dense
        # 检索都是按传入列表的位置编号的），不是 chunk_id（=turn_index，因为
        # 只挑了role='user'的历史，两者之间会有空洞，不能拿positional index
        # 直接当chunk_id用）——这里显式按位置取，避免搞混。
        pool = [
            RetrievedPassage(
                chunk=history_chunks[pos],
                rrf_score=score,
                bm25_score=bm25_raw.get(pos, 0.0),
                dense_score=dense_raw.get(pos, 0.0),
            )
            for pos, score in fused
        ]

        if self.kb_pipeline.reranker.is_real:
            with span("精排", f"聊天记录{len(pool)}条候选"):
                self.kb_pipeline.reranker.rerank(query, pool, top_n=len(pool))

        # 相关性和时间衰减分开用（跟Generative Agents的做法一致：相关性决定
        # "能不能用"，时间/重要性只决定"相关的里面谁排前面"）：先记下衰减前的
        # 原始相关性，下面的逐条过滤和_try_answer_from_history()的gate都看它；
        # 衰减和闲聊降权只作用于排序。之前gate看的是衰减后的分数，"相不相关"
        # 和"新不新"混成一个数——阶段二2.1的turns_ago踩坑、以及L2检索范围放开
        # 后"你上次推荐我吃什么来着"翻不到（原始相关性约0.69，衰减到0.425过不了
        # 0.5），都是同一个结构问题，见模块开头"L2检索范围放开"一节。
        for p in pool:
            p.relevance_score = p.rerank_score if p.rerank_score is not None else p.bm25_score

        # 时间衰减：turns_ago 是"隔了几轮用户发言"，越大权重打折越多
        for p in pool:
            turns_ago = current_user_position - user_turn_position[p.chunk.chunk_id]
            decay = (1 - DECAY_RATE) ** max(turns_ago, 0)
            if is_general_chat[p.chunk.chunk_id]:
                decay *= GENERAL_CHAT_WEIGHT
            if p.rerank_score is not None:
                p.rerank_score *= decay
            p.bm25_score *= decay

        ranked = sorted(
            pool,
            key=lambda p: ((p.relevance_score if re.search(r"之前|上次|先前|前面|当时", query) else 0),
                           p.rerank_score if p.rerank_score is not None else p.bm25_score),
            reverse=True,
        )

        # 逐候选置信度过滤（㉓）：排序只保证"矮子里拔将军"排出了名次，不
        # 保证凑够top_n条时那条陪跑候选本身真的跟query相关——之前直接
        # ranked[:top_n]返回，哪怕最后一名分数远低于及格线，也会被塞进
        # 返回值，连带出现在_try_answer_from_history()的证据列表和"引用
        # 来源"里。这里复用_try_answer_from_history()已有的同一套及格线
        # 常量，对排完名次的列表再做一遍逐条门槛过滤，不合格的直接砍掉，
        # 不再因为"凑数到top_n"硬塞。完整背景见模块开头"㉓"一节。
        floor = L2_RERANK_FLOOR if self.kb_pipeline.reranker.is_real else L2_BM25_FLOOR
        ranked = [p for p in ranked if p.relevance_score is not None and p.relevance_score >= floor]
        print(f"[conversation_memory] 聊天记录候选（轮次/原始相关性）："
              f"{[(p.chunk.chunk_id, round(p.relevance_score, 3)) for p in ranked[:top_n]]}")

        # 过滤后可能不足top_n条、甚至一条都不剩，这里不做"至少留一条"的
        # 特殊处理：一条都不剩时，调用方看到空列表会跟"压根没检索到候选"
        # 走同一条兜底路径（画像兜底），效果上完全等价，见模块开头"㉓"
        # 一节。
        return ranked[:top_n]

    def _try_answer_from_history(self, query: str, extra_context=None, retrieval_query=None) -> Optional[SubAnswer]:
        """extra_context：可选，透传给生成步骤（`llm_generate.py`），格式跟
        `route_query()`的`recent_context`一致，见模块开头"检索架构升级"
        一节。默认None，不传时行为跟改动前逐字一致。返回值从裸字符串改成
        了`SubAnswer`（display/core/ok），同一节有完整说明。

        画像可见性修复：这个方法原来两处"确定翻不到"的地方（0候选/gate
        没过）都是直接return None，交给调用方展示"没翻到"的兜底文案——
        现在改成先试一次`_try_answer_from_profile_fallback()`，画像里真的
        有答案才用，没有依然return None，维持原有行为，不引入新的失败
        模式。完整设计动机见模块开头"画像可见性"一节；只接在这个方法上，
        不接`_try_answer_from_local_kb()`，理由同一节末尾的说明。
        """
        if self._personal_history(query):
            candidates = self._search_history(retrieval_query or query, personal_only=True)
        else:
            candidates = self._search_history(retrieval_query or query)
        candidates = self._filter_personal_passages(query, candidates)
        floor = L2_RERANK_FLOOR if self.kb_pipeline.reranker.is_real else L2_BM25_FLOOR

        def passed(cands):
            scored = [p.relevance_score for p in cands if p.relevance_score is not None]
            return bool(scored) and max(scored) >= floor

        rewrites_tried = False
        if (not passed(candidates) and self.llm_client.is_real and self.kb_pipeline.reranker.is_real
                and len(self.store.get_recent_turns(self.conversation_id, limit=2)) > 1):  # 没有历史就不用重查
            candidates = self._filter_personal_passages(query, self._retry_history_with_rewrites(query, extra_context)) or candidates
            rewrites_tried = True

        def answer_with_retry(evidence):
            result = self._answer_history_with(query, evidence, extra_context)
            if (result is not None or rewrites_tried or not self.llm_client.is_real
                    or not self.kb_pipeline.reranker.is_real):
                return result
            print("[conversation_memory] 历史候选通过门槛但未能回答，补做一次上下文改写检索。")
            retry = self._filter_personal_passages(query, self._retry_history_with_rewrites(query, extra_context))
            old_ids = {passage.chunk.chunk_id for passage in evidence}
            if not retry or not any(passage.chunk.chunk_id not in old_ids for passage in retry):
                return result
            return self._answer_history_with(query, self._with_following_replies(retry), extra_context)
        # 最近窗口里的发言直接作为证据（见_recent_window_passages()）：只在真实LLM下
        # 加，因为是否答得上要靠生成步骤判断；没配LLM的2.1链路维持原样
        window = self._recent_window_passages() if self.llm_client.is_real else []
        window = self._filter_personal_passages(query, window)
        if window:
            # 检索到的记录照旧要过门槛（没过的不带）；最近窗口不用过门槛，去重后接在后面
            older = self._with_following_replies(candidates) if passed(candidates) else []
            seen = {p.chunk.chunk_id for p in older}
            return answer_with_retry(older + [p for p in window if p.chunk.chunk_id not in seen])
        if not candidates:
            return self._try_answer_from_profile_fallback(query)

        # gate看衰减前的原始相关性（relevance_score），见_search_history()里
        # "相关性和时间衰减分开用"的说明
        floor = L2_RERANK_FLOOR if self.kb_pipeline.reranker.is_real else L2_BM25_FLOOR
        scored = [p.relevance_score for p in candidates if p.relevance_score is not None]
        if not scored or max(scored) < floor:
            return self._try_answer_from_profile_fallback(query)

        # 置信度gate（上面这两段if）已经通过——只有过了这一关，才轮到LLM
        # 决定"怎么把这些证据说出来"；gate没过直接return None，不会走到这里，
        # 也就不会调用LLM，见 llm_generate.py 模块开头"置信度gate是硬性防线"
        # 的说明。生成失败（LLM不可用/调用出错）时往下走，退回2.1原本的
        # "展示证据"格式。
        if self.llm_client.is_real:
            candidates = self._with_following_replies(candidates)
        return answer_with_retry(candidates)

    def _with_following_replies(self, cands: List[RetrievedPassage]) -> List[RetrievedPassage]:
        """命中一句用户发言时，带上紧跟着的那条助手回复（检索里常说的邻近上下文扩展）。

        【2026-10-01测试题F7】"你上次推荐我看的电影是哪部"：改写后命中的是用户当时的
        提问"周末想看部电影，推荐一个呗"（0.78~0.82），答案《星际穿越》在下一条助手
        回复里（0.53）没带进来，生成只能说"当时推荐了哪部没有保留下来"。问题和答案
        往往分在相邻两条里，命中问题就把答案一起给生成。只带紧跟的下一条、且是助手
        回复的（群聊里下一条可能是别的群友）。"""
        all_turns = self.store.get_all_turns(self.conversation_id)
        names = self._speaker_names(all_turns)
        when = {r["turn_index"]: r["created_at"] for r in all_turns}
        by_turn = {r["turn_index"]: r for r in self._history_index_rows(all_turns[:-1], names)}
        out, seen = [], set()
        for p in cands:
            if p.chunk.chunk_id not in seen:
                out.append(p)
                seen.add(p.chunk.chunk_id)
            row = by_turn.get(p.chunk.chunk_id)
            nxt = by_turn.get(p.chunk.chunk_id + 1)
            if row and row["role"] == "user" and nxt and nxt["role"] == "assistant" and nxt["turn_index"] not in seen:
                out.append(RetrievedPassage(
                    chunk=Chunk(chunk_id=nxt["turn_index"], source_file=self._history_source_label(nxt, names, when),
                                text=nxt["text"]),
                    rrf_score=0.0, bm25_score=0.0, dense_score=0.0))
                seen.add(nxt["turn_index"])
        return out

    def _retry_history_with_rewrites(self, query: str, recent_context=None) -> List[RetrievedPassage]:
        """翻聊天记录第一次没过门槛：问题改写成"当时可能是怎么说的"陈述句再检索一次
        （query_planner.rewrite_for_history()，跟知识库没过门槛时改写重查同一个思路）。
        改写路径用更高的门槛L2_REWRITE_RERANK_FLOOR，理由同pipeline.REWRITE_RERANK_FLOOR：
        多给了几次机会，偶然撞出高分的概率变大。返回过了这道门槛的候选（同一条取最高
        分），一条都没有就返回空列表。"""
        rewrites = rewrite_for_history(query, self.llm_client, recent_context=recent_context)
        best = {}
        for rw in rewrites:
            found = self._search_history(rw, personal_only=True) if self._personal_history(query) else self._search_history(rw)
            for p in found:
                if p.relevance_score is not None and p.relevance_score >= L2_REWRITE_RERANK_FLOOR:
                    old = best.get(p.chunk.chunk_id)
                    if old is None or p.relevance_score > old.relevance_score:
                        best[p.chunk.chunk_id] = p
        hits = sorted(best.values(), key=lambda p: -p.relevance_score)[:2]
        top = f"{hits[0].relevance_score:.3f}" if hits else "无"
        print(f"[conversation_memory] 翻聊天记录没过门槛，改写成陈述句重查：{rewrites} -> "
              f"过改写门槛{L2_REWRITE_RERANK_FLOOR}的{len(hits)}条（最高{top}）")
        return hits

    def _answer_history_with(self, query: str, candidates: List[RetrievedPassage],
                             extra_context=None) -> Optional[SubAnswer]:
        """拿聊天记录候选（检索到并过了门槛的 + 最近窗口里的）生成回答。

        【2026-10-02重跑主人绑定测试】群聊里问题前面写明是谁问的：长安刚问完"介绍一下我的
        开题报告"，群友问同一句，最近窗口里助手回答长安的那条直接成了证据；背景里虽然有
        "当前提问的是【某某】"、来源也写着"回答长安的话"，生成步骤3次都还是答成"你的开题
        报告是……"。把提问人写在问题这一行之后，3次都答成"没有你的记录/那是长安的"。"""
        gen_query = query
        candidates = self._filter_personal_passages(query, candidates)
        if not candidates:
            return self._try_answer_from_profile_fallback(query)
        if self.is_group:
            names = self._speaker_names(self.store.get_all_turns(self.conversation_id))
            me = self._participant_labels(names)[self.user_id]
            gen_query = f"（当前提问的是{me}）{query}"
        if self.llm_client.is_real:
            generated = llm_generate_answer(gen_query, candidates, self.llm_client, extra_context=extra_context,
                                             caller_label="history")
            if is_no_answer(generated):
                # 生成步骤读完候选判断完全答不了（可答性判断，见llm_generate.py
                # NO_ANSWER的说明）：跟gate没过同样处理
                return self._try_answer_from_profile_fallback(query)
            if generated is not None:
                generated, listed = compact_citations(generated, [p.chunk.source_file for p in candidates])
                refs = "\n".join(f"[{i}] {s}" for i, s in enumerate(listed, start=1))
                display = f"【L2情景记忆——LLM生成回答，引用来源见下】\n{generated}\n\n引用来源：\n{refs}"
                return SubAnswer(display=display, core=generated, ok=True, source="history",
                                 refs=["之前的对话记录"])

        if self.llm_client.is_real:
            # 真实LLM重试后仍然生成失败：如实说明+来源，不把原始聊天记录倒给用户
            return SubAnswer(display=self._generation_failed_reply(
                "【L2情景记忆】翻到了之前聊过的相关内容", [p.chunk.source_file for p in candidates]),
                core="", ok=True, source="history", refs=["之前的对话记录"])

        # 没配LLM（离线开发/测试）：沿用2.1的展示原始记录
        lines = ["【L2情景记忆——翻到了之前聊过的相关内容（已按时间衰减加权排序）】"]
        for p in candidates:
            score_display = f"{p.rerank_score:.3f}" if p.rerank_score is not None else f"BM25={p.bm25_score:.2f}"
            lines.append(f"- ({p.chunk.source_file}，衰减后分数={score_display}) {p.chunk.text}")
        return SubAnswer(display="\n".join(lines), core="", ok=True, source="history", refs=["之前的对话记录"])

    # ---------- L3：知识库（阶段一验证过的检索流水线） ----------

    # 路由给kb_scope="project"时只查这几类文档（见pipeline.retrieve()的doc_types）
    PROJECT_DOC_TYPES = ("课题报告",)
    # 路由给kb_scope="paper"（"原文里是怎么说的"）时只查文献原文（2026-10-02真机：原文是
    # 英文，中文讲稿笔记总是排在前面，问"原文里怎么说"拿到的是讲稿）
    PAPER_DOC_TYPES = ("文献原文", "文献补充材料")

    def _jc_session_filter(self, jc_session: str) -> Optional[Tuple[List[str], str]]:
        """路由给的jc_session（"latest"/"earliest"/日期前缀）-> (这几次JC讲的文献paper_key,
        补在检索问题后面的说明)。算不出来（没有目录、没对上任何一次）返回None，照常全库检索。

        【2026-10-02真机】"介绍一下我最近一次文献分享会讲的文献"答成了6月那次，实际最近
        一次是9月12日：检索不懂"最近""第一次"这类按时间排序/筛选的条件，路由和生成也都
        看不到8次JC的日期。业界做法是self-query/元数据过滤（LangChain SelfQueryRetriever、
        LlamaIndex auto-retrieval）：LLM只把问题里的条件抽成结构化字段，哪一次由代码按目录
        里的日期算（确定、不会算错，JC次数变多也不用改prompt），检索只在那次的资料里查。
        跟kb_scope="project"只查课题报告是同一个模式（用户2026-10-02选的方案）。"""
        catalog = getattr(self.kb_pipeline, "catalog", None)
        if not jc_session or not catalog:
            return None
        today = datetime.now(timezone(timedelta(hours=8))).date().isoformat()
        sessions = sorted((x for x in catalog.get("jc_sessions") or []
                           if x.get("date") and x["date"] <= today and (x.get("match") or {}).get("key")),
                          key=lambda x: x["date"])
        if jc_session == "latest":
            picked = sessions[-1:]
        elif jc_session == "earliest":
            picked = sessions[:1]
        else:
            picked = [x for x in sessions if x["date"].startswith(jc_session)]
        if not picked:
            print(f"[conversation_memory] jc_session={jc_session!r}：目录里没对上哪一次JC，照常全库检索")
            return None
        titles = {p.get("key"): p.get("title", "") for p in catalog.get("papers") or []}
        keys = list(dict.fromkeys(x["match"]["key"] for x in picked))
        note = "；".join(f"{x['date']} JC讲的文献：{titles.get(x['match']['key'], '')}" for x in picked)
        print(f"[conversation_memory] jc_session={jc_session!r} -> {note}")
        return keys, note

    def _try_answer_from_local_kb(self, query: str, extra_context=None, scope: str = "all",
                                  jc_session: str = "", retrieval_query=None) -> SubAnswer:
        """本地论文知识库：检索+置信度gate+生成。2.1时user_says()直接调用
        `kb_pipeline.answer()`——那个方法内部把"检索+置信度判断+格式化成
        字符串"三步封在一起，拿不到中间的结构化候选片段。2.2要在"gate判定
        可信"之后插入LLM生成这一步，所以改成自己调用 `retrieve_with_gate()`
        （跟 answer() 内部调用的是同一个方法，判断逻辑完全一致），拿到
        passages自己决定怎么往下走。

        跟 `_try_answer_from_wiki_page()` 不一样，这个方法总是有确定的
        回复、不会返回None——本地知识库是整条链路最后能落脚的地方，gate判定
        不可信时返回`SubAnswer(ok=False)`（`.display`是2.1原本的拒答文案，
        本身就是这一轮最终的答案，不是"跳过交给别人再试一次"）。不调用LLM
        去生成"答不上"这件事本身——理由同`_try_answer_from_history()`，见
        llm_generate.py 模块开头的说明。

        extra_context：见`_try_answer_from_history()`同名参数的说明。
        """
        rewriter = None
        if self.llm_client.is_real:
            rewriter = lambda q: rewrite_for_kb(q, self.llm_client)
        doc_types = {"project": self.PROJECT_DOC_TYPES, "paper": self.PAPER_DOC_TYPES}.get(scope)
        jc = None if scope == "project" else self._jc_session_filter(jc_session)
        paper_keys = None
        if jc:
            # 是哪一次、讲的哪篇写进问题里：精排和生成都看得到（检索只在这几篇的片段里查）
            paper_keys, note = jc
            query = f"{query}（{note}）"
            if retrieval_query:
                retrieval_query = f"{retrieval_query}（{note}）"
        passages, passed = self.kb_pipeline.retrieve_with_gate(retrieval_query or query, query_rewriter=rewriter,
                                                               doc_types=doc_types, paper_keys=paper_keys)
        scores = [passage.relevance_score if passage.relevance_score is not None else passage.rerank_score
                  for passage in passages]
        print(f"[conversation_memory] 知识库门槛：通过={passed}，范围={scope}，候选分数={scores}")
        if not passed:
            return SubAnswer(display=self.kb_pipeline.REJECT_MESSAGE, core="", ok=False)

        if self.llm_client.is_real:
            generated = llm_generate_answer(query, passages, self.llm_client, extra_context=extra_context,
                                             caller_label="local_kb")
            if is_no_answer(generated):
                # gate过了，但生成步骤读完片段判断完全答不了（C15那类"话题相关、
                # 原文并不回答"）：按拒答处理，调用方会转去联网搜索/大模型自身知识
                print("[conversation_memory] 知识库候选通过门槛，但生成判定NO_ANSWER。")
                return SubAnswer(display=self.kb_pipeline.REJECT_MESSAGE, core="", ok=False)
            if generated is not None:
                generated, listed = compact_citations(generated, [p.chunk.source_file for p in passages])
                refs = "\n".join(f"[{i}] {s}" for i, s in enumerate(listed, start=1))
                display = f"【L3知识库——LLM生成回答，引用来源见下】\n{generated}\n\n引用来源：\n{refs}"
                return SubAnswer(display=display, core=generated, ok=True, source="kb",
                                 refs=list(dict.fromkeys(p.chunk.source_file for p in passages)))

        refs = list(dict.fromkeys(p.chunk.source_file for p in passages))
        if self.llm_client.is_real:
            # 真实LLM重试后仍然生成失败：如实说明+来源，不把原始检索片段倒给用户。
            # ok=True——gate确实通过了（资料是对的，只是没整理成回答），不去联网兜底
            return SubAnswer(display=self._generation_failed_reply("【L3知识库】找到了相关资料", refs),
                             core="", ok=True, source="kb", refs=refs)

        # 没配LLM（离线开发/测试）：退回2.1原本"展示编号证据"的格式，跟直接调用
        # kb_pipeline.answer() 在gate通过时的输出逐字一致。
        return SubAnswer(display=self.kb_pipeline.format_evidence(passages), core="", ok=True, source="kb",
                         refs=refs)

    @staticmethod
    def _generation_failed_reply(found: str, sources: List[str]) -> str:
        """【2026-10-01】资料找到了、但LLM重试后仍然没能整理成回答（接口卡住/超过总时长/
        内容安全拒绝等）时给用户看的话。原来退回展示原始检索片段/原始聊天记录（可能两千
        多字、带各项分数）；用户的核心要求是"用户看到的应当是整理过的回答，而不是原始检索
        到的一大堆内容"，所以改成如实说明+资料来源，用户可以稍后再问。只在配了真实LLM时
        用；没配LLM的离线模式照旧展示原始材料。9-30加超时之前程序会一直等，所以真实
        聊天记录里一次也没出现过原文倒出来，加了超时后才可能走到这里。"""
        listed = "\n".join(f"[{i}] {s}" for i, s in enumerate(dict.fromkeys(sources), start=1))
        return (f"{found}，但这次整理回答时模型接口没有正常响应，没能生成回答，请稍后再问一次。"
                f"\n\n找到的资料：\n{listed}")

    def _try_answer_from_wiki_page(self, query: str, keyword: str, extra_context=None,
                                    detect_candidates: bool = False,
                                    cargo_profession: str = "", cargo_rarity: str = "") -> Optional[SubAnswer]:
        """明日方舟wiki：整页查询+LLM生成。返回None表示这条路没走通（页面
        没查到/生成失败），调用方应该退回本地知识库再试一次——路由判断本身
        也可能出错，不能让wiki这一步一失手就让整轮直接拒答，完整理由见
        模块开头"L3的第二个知识来源"一节。

        extra_context：见`_try_answer_from_history()`同名参数的说明。这里
        通常不需要靠它解决代词指代问题——`keyword`已经是`route_query()`结合
        上下文解析过的结果，传进来的passage本身就是那个确定实体的整页正文，
        LLM读passage就能确定"她"指的是谁；传进来纯粹是保持三个方法签名
        一致、且多一点背景不会有坏处，不是这里要解决的主要问题。

        detect_candidates：F（多候选页面消歧）新增，默认False、不传时跟
        改动前逐字一致。只有`_answer_multi_hop()`确认"这个子查询的结果会
        被后面的子查询依赖"时才会传True——为True时，多付出一次`query_
        planner.detect_answer_candidates()`调用，核对`page["text"]`这份
        原始正文里是不是真的存在不止一个候选，结果写进返回的`SubAnswer.
        candidates`。完整设计动机见模块开头"F（多候选页面消歧）"一节。

        cargo_profession/cargo_rarity：④索引/聚合/动态更新新增，默认空
        字符串、不传时跟改动前逐字一致。`route_query()`判断出
        `wiki_query_kind=="cargo_filter"`（比如"六星狙击干员有哪些"）时，
        调用方会传其中至少一个非空值进来——这种情况下这次查询根本不是
        "给定关键词查一个具体页面"，直接转给`_try_answer_from_cargo_filter()`
        处理，不走下面`fetch_page_for_answer(keyword)`这条entity路径（这种
        情况下keyword本来就是空字符串，硬走下去也查不到页面）。完整背景见
        `wiki_router.py`模块开头"④索引/聚合/动态更新"一节和
        `prts_cargo_client.py`模块开头的说明。

        `fetch_page_for_answer()`传了一个`disambiguator`闭包——"关键词短语
        没有对应任何真实页面标题"这类问题（比如"主线第五章的最终boss是谁"）
        的候选列表消歧，闭包包住这里已经有的`query`（原始问题，判断哪条
        候选真的对应用户想问的东西离不开这个）和`self.llm_client`，真正的
        判断逻辑在`wiki_router.disambiguate_wiki_candidates()`。完整设计
        动机见`prts_wiki_client.py`模块开头"坑A遗留的一个更根本的问题"
        一节和`wiki_router.py`模块开头"新增disambiguate_wiki_candidates()"
        一节。
        """
        if cargo_profession or cargo_rarity:
            return self._try_answer_from_cargo_filter(cargo_profession, cargo_rarity)

        def _disambiguator(kw, candidates):
            return disambiguate_wiki_candidates(query, kw, candidates, self.llm_client)

        page = fetch_page_for_answer(keyword, disambiguator=_disambiguator)
        if page is None:
            return None  # 关键词搜不到页面（含消歧后仍拿不准）/网络请求失败，退回本地知识库再试

        passage = RetrievedPassage(
            chunk=Chunk(chunk_id=0, source_file=f"prts.wiki/{page['title']}", text=page["text"]),
            rrf_score=0.0, bm25_score=0.0, dense_score=0.0,
        )
        generated = llm_generate_answer(query, [passage], self.llm_client, extra_context=extra_context,
                                         caller_label=f"wiki:{page['title']}")
        if generated is None or is_no_answer(generated):
            return None  # 生成失败/页面读下来完全答不了，调用方转去联网搜索等兜底

        display = (f"【L3知识库——明日方舟wiki《{page['title']}》，LLM生成回答】\n{generated}"
                   f"\n\n引用来源：\n[1] prts.wiki/{page['title']}")
        candidates = (detect_answer_candidates(query, page["text"], generated, self.llm_client)
                      if detect_candidates else [])
        return SubAnswer(display=display, core=generated, ok=True, candidates=candidates, source="wiki",
                         refs=[f"prts.wiki/{page['title']}"])

    def _try_answer_from_cargo_filter(self, profession: str, rarity: str) -> Optional[SubAnswer]:
        """④索引/聚合/动态更新——"六星狙击干员有哪些"这类跨干员职业/稀有度
        组合筛选问题的正式实现，真正的查询逻辑在`prts_cargo_client.py`
        （直接问prts.wiki自己维护的Cargo结构化数据表要结果，不是抓页面
        清洗文本），完整验证过程见该文件模块开头和HANDOFF.md⑰。

        故意不调用LLM重新生成这个答案：Cargo查出来的干员名单本身就是
        权威数据，直接确定性拼接成一句话展示，比"再让LLM读一遍名单、
        重新组织语言"更准确——多一次LLM生成只有把名字改错/漏掉的风险，
        没有额外的价值（这里不需要总结/归纳，只需要如实列出），这跟项目
        里"能不用LLM重新表述确定性结果就不用"的一贯原则一致（比如
        `_try_answer_from_history()`翻不到内容时也是直接给固定文案，不
        调用LLM去生成"没翻到"这句话本身）。

        返回None表示查询本身没有成功（网络请求失败/职业或稀有度参数不
        合法），调用方应该退回本地知识库再试一次——跟`fetch_page_for_
        answer()`查不到页面时的降级逻辑保持一致。查询成功但结果是空列表
        （确实没有符合条件的干员）不算这种情况，会正常返回一个SubAnswer，
        因为"查到了、确实没有"本身就是一个确定的、有意义的答案，不是
        "这次没查成"。
        """
        rarity_int: Optional[int] = None
        if rarity:
            try:
                rarity_int = int(rarity)
            except ValueError:
                rarity_int = None  # 交给query_operators()的范围校验去拒绝

        results = query_operators(profession=profession or None, rarity=rarity_int)
        if results is None:
            return None  # 查询失败/参数不合法，退回本地知识库再试

        criteria_parts = []
        if profession:
            criteria_parts.append(f"职业={profession}")
        if rarity:
            criteria_parts.append(f"稀有度={rarity}星")
        criteria = "、".join(criteria_parts) if criteria_parts else "（无筛选条件）"

        if not results:
            core = f"按条件（{criteria}）没有查到符合条件的干员。"
            display = (f"【L3知识库——明日方舟Cargo结构化查询】{core}"
                       "\n\n来源：PRTS干员数据库（https://prts.wiki/w/干员一览）。")
            return SubAnswer(display=display, core=core, ok=True, source="wiki", refs=["prts.wiki干员数据库"])

        names = "、".join(r["name"] for r in results)
        core = f"符合条件（{criteria}）的干员共{len(results)}个：{names}"
        display = (f"【L3知识库——明日方舟Cargo结构化查询，直接查询干员数据库】\n{core}"
                   "\n\n来源：PRTS干员数据库（https://prts.wiki/w/干员一览）。"
                   "\n说明：此名单未排除特殊模式专属干员，数据以PRTS当前记录为准。")
        return SubAnswer(display=display, core=core, ok=True, source="wiki", refs=["prts.wiki干员数据库"])

    # ---------- 本地知识库：文献目录（列清单类问题） ----------

    def _try_answer_from_catalog(self, scope: str = "all") -> Optional[SubAnswer]:
        """"作者读过哪些文献/讲过哪些JC"这类列清单问题：直接查文献目录
        （kb_sources.py，以Zotero为数据源），确定性拼接、不经过LLM重新表述——
        跟Cargo"六星狙击干员有哪些"同一个思路：列清单靠相似度检索没法保证列全，
        让LLM重新组织语言只有漏项/改错名字的风险。没有目录（Zotero从没成功读取过）
        时返回None，调用方退回普通检索。

        scope（路由的catalog_scope）：只列问题问到的范围。原来不管问什么都列全部
        13篇——用户看测试题时指出"长安做过几次JC、都讲了哪些"也把没讲过的5篇
        列了出来；"读过CoSpar吗"这种只问一篇的问题现在由路由判成content走检索。
        "all"=全部（讲过/没讲过分组），"jc"=只列JC讲过的（按场次），"not_jc"=只列
        读过没讲的。"""
        from kb_sources import _cn_date, _short_title
        catalog = getattr(self.kb_pipeline, "catalog", None)
        if not catalog or not catalog.get("papers"):
            return None
        papers = catalog["papers"]
        told = sorted((p for p in papers if p.get("jc")), key=lambda p: p["jc"][0])
        untold = [p for p in papers if not p.get("jc")]

        def line(p, prefix=""):
            first = (p.get("summary") or "").split("。")[0]
            return (f"- {prefix}《{p['title']}》（{p['first_author']} {p['year']}，{p['journal']}）"
                    + (f"：{first}。" if first else ""))

        told_lines = [line(p, prefix=f"{'、'.join(_cn_date(d) for d in p['jc'])}讲：") for p in told]
        if scope == "jc":
            n_sessions = len(catalog.get("jc_sessions") or []) or len(told)
            lines = [f"{OWNER_NAME}一共做过{n_sessions}次文献分享会（JC），讲过的文章（{len(told)}篇）："]
            lines += told_lines
        elif scope == "not_jc":
            lines = [f"{OWNER_NAME}读过、但没在JC上讲过的文献有{len(untold)}篇：" if untold
                     else f"{OWNER_NAME}读过的文献都在JC上讲过。"]
            lines += [line(p) for p in untold]
        else:
            lines = [f"{OWNER_NAME}读过的文献共{len(papers)}篇（来自{OWNER_NAME}的Zotero文献库）。", "",
                     f"在文献分享会（JC）上讲过的（{len(told)}篇）："]
            lines += told_lines
            if untold:
                lines += ["", f"读过、但没在JC上讲过的（{len(untold)}篇）："]
                lines += [line(p) for p in untold]
        core = "\n".join(lines)
        display = f"【本地知识库——{OWNER_NAME}的文献目录（Zotero）】\n{core}"
        return SubAnswer(display=display, core=core, ok=True, source="kb", refs=[f"{OWNER_NAME}的Zotero文献目录"])

    # ---------- 兜底：知识来源都查不到时，联网搜索 -> 大模型自身知识 ----------

    _KB_MISS_NOTE = f"（{OWNER_NAME}的资料里没有找到这部分内容，以下来自其它渠道）\n"

    # wiki路由转来的兜底：问题已经确定是关于这款游戏的，但原句里常常不带游戏名
    # （"主线第五章的最终boss是谁"），原样拿去搜会搜到别的游戏（真机实测搜到了
    # 《黑神话：悟空》第五章）。搜索词前面补上游戏名，回答时也告诉模型问的是它。
    WIKI_DOMAIN = "明日方舟"

    def _answer_beyond_sources(self, query: str, extra_context=None, kb_miss: bool = False,
                               domain: str = "", search_query: str = "") -> Optional[SubAnswer]:
        """QQ群压力测试准备（二），用户的决策：wiki/本地知识库查不到时，不要
        一句"不知道"把天聊死——先联网搜索，搜索也没有可用结果再让大模型用
        自身知识回答，每一级都用用户定的标注文字（WEB_SEARCH_LABEL/
        LLM_KNOWLEDGE_LABEL，逐字，由代码加在回答后面、不交给LLM写）。

        三处会调用：wiki查不到、本地知识库拒答（kb_miss=True，回答前加一句
        "作者的课题资料里没有找到这部分内容"，免得群友以为是作者课题里写的）、
        路由判定是需要事实依据的客观问题（general+needs_search）。L2翻旧账
        查不到时不调用——网上不可能有"用户之前说过什么"。

        这跟⑬-⑮反对的"用训练记忆偷偷补答案"不矛盾：那里的问题是不声不响地
        替代检索结果，这里每一段都明确标注了来源。返回None只在LLM不可用、
        或者两级都没能生成出回答时，调用方沿用原来的拒答/兜底文案。

        search_query：路由给的独立搜索词（wiki_router的search_query字段）。【2026-10-04真机】
        聊着KFC问"现在有什么正在卖的新品或者老产品回归吗？"，原句里没有"KFC"，拿原句去搜，
        5条结果对不上，退到了大模型自身知识。搜索引擎看不到最近对话，业界做法是先把问题改写成
        单独能搜的查询（对话式查询改写/history-aware retriever）；路由那次调用本来就看得到最近
        对话，顺带给出搜索词，不多一次调用。没给时用原句。整理搜索结果、回答用的还是原句+
        最近对话。
        """
        if not self.llm_client.is_real:
            return None
        prefix = self._KB_MISS_NOTE if kb_miss else ""

        question = query
        search_query = search_query or query
        if domain and domain not in query:
            search_query = f"{domain} {query}"
            question = f"{query}（问的是游戏《{domain}》）"

        results = self.web_search.search(search_query) if self.web_search is not None else None
        if results:
            answer = llm_answer_from_web(question, results, self.llm_client, extra_context=extra_context)
            if answer is not None and not is_no_answer(answer):
                answer, listed = compact_citations(answer, [f"{r['title']} {r['link']}" for r in results])
                refs = "\n".join(f"[{i}] {s}" for i, s in enumerate(listed, start=1))
                display = f"{prefix}{answer}\n\n参考链接：\n{refs}\n\n{WEB_SEARCH_LABEL}"
                return SubAnswer(display=display, core=answer, ok=True, source="web")

        answer = llm_answer_from_own_knowledge(question, self.llm_client, extra_context=extra_context)
        if answer is None:
            return None
        return SubAnswer(display=f"{prefix}{answer}\n\n{LLM_KNOWLEDGE_LABEL}", core=answer, ok=True,
                         source="llm")

    # ---------- 统一分流：LLM可用时一次性决定 general/history/wiki/local_kb ----------

    def _dispatch_with_router(self, query: str, recent_context=None,
                              route_future=None) -> Optional[Tuple[str, bool]]:
        """真机测试暴露出的问题逼出的重新设计，完整背景见模块开头"L2语料
        该不该收录route="general"命中的发言"一节：以前是L2无条件排在路由
        判断前面先跑一次，route="wiki"/"local_kb"/"general"这三选一只在L2
        没找到任何候选时才轮得到——这个顺序会导致"只要闲聊内容跟历史里
        随便哪句长得像（哪怕是它自己刚生成的回复），就被L2抢答，压根不会
        走到路由这一步"，`is_general_chat`标记也就永远打不上，闲聊像滚雪球
        一样自我延续、修不干净。

        现在改成路由（`route_query()`，四选一：wiki/local_kb/history/
        general）在最前面先跑一次，一次性决定"这句话该谁答"：真正该翻旧账
        的问题会被分类成route="history"，这时候才去调用L2的
        `_try_answer_from_history()`；这样"判断该不该标记general"和"判断该
        谁来回答"是同一次判断，不会再出现"L2抢先答完、标记却没跟上"的
        情况。返回(回复文本, 这一轮算不算general闲聊)的二元组，general与否
        直接由这次路由结果决定，调用方不需要再自己去猜。

        LLM不可用时`route_query()`直接返回None（没有更低配的规则版可退，
        见 wiki_router.py 模块开头说明），这时返回None，调用方应该退回2.1
        原本"L2先试、答不出来再落到本地知识库"这条legacy链路——2.1本来就
        没有wiki/general/history这几条路径，不存在"退化"这回事。

        这个重新设计不是没有代价，如实记录：以前L2永远排在最前面，"这是不
        是在问历史"这件事有真正的检索结果撑腰，不会凭空猜错；现在改成先
        猜分类、猜中"history"才去检索，如果一句真正该翻旧账的话在语言上
        没有"来着"这类明显的回指线索，可能被误判成general，被自由聊天直接
        回答而不是老实去翻历史——这是主动接受的新权衡，缓解手段只有把
        wiki_router.py里"history"这一类的判断标准和示例尽量写清楚，没有
        更好的兜底手段。

        【route="general"命中时is_general_chat不再恒为True】真机测试又
        暴露一个新问题：陈述句（"我们组决定用Chroma做向量数据库"）以前
        因为route="general"的prompt/示例只覆盖问句/寒暄，容易被误判进
        wiki/local_kb，触发置信度gate拒答；给general补上陈述句覆盖之后，
        这类内容也会命中route="general"，但它们跟真正的寒暄闲聊不一样——
        是以后可能被"翻旧账"追问的具体事实，不能像闲聊一样被永久排除出
        L2候选池。所以这里不再对route="general"无条件返回True，改成读
        `route["is_chitchat"]`：真正的寒暄/闲聊/请求建议仍然是True（维持
        原有排除行为，避免闲聊自我延续这个已经修过的问题重新出现）；
        陈述新信息的场景是False（不排除，正常留在L2候选池里，将来可以被
        翻出来）。完整背景、这个字段的设计取舍见 wiki_router.py 模块开头
        "真机测试暴露的第5个问题"一节。

        【recent_context：解决代词/省略指代解析不出来被误判成history的问题】
        真机测试又暴露一个新场景："能天使的三技能专三的材料是什么"答对之后，
        紧接着问"她精二的材料是什么"，"她"单独看没有任何线索，被误判成了
        route="history"；换成"能天使精二的材料是什么"（把"她"换成明确的
        名字）就能正确走route="wiki"。根因跟`llm_l1.py`那次"L0上下文窗口
        未接入L1判断"是同一类——`route_query()`判断时只看得到这一句话，
        看不到L0维护的最近对话，没有信息能补全代词指代的是谁。修法完全
        对称：`route_query()`新增`recent_context`参数，传入时能帮它把"她"
        解析成"能天使"、正确判给wiki并给出正确的keyword，不传时行为跟
        改动前逐字一致。完整背景见 wiki_router.py 模块开头"真机测试暴露的
        第6个问题"一节。

        【检索架构升级：recent_context现在也顺带传给了L2/L3的生成步骤】
        以前recent_context只用来帮route_query()解析代词/指代，判断出正确
        的route/keyword之后就不再往下传了——`_try_answer_from_history()`/
        `_try_answer_from_local_kb()`内部真正做检索匹配、以及调用LLM生成
        回答这几步完全看不到这段上下文。现在这几个方法都新增了
        `extra_context`参数，这里顺手把recent_context传进去，让生成措辞时
        也能借助这段背景——这是HANDOFF"L0上下文窗口未接入路由判断"一节末尾
        记录的遗留缺口的一部分修复，完整设计见`query_planner.py`模块开头。

        【㉑单查询代词解析缺口，以及㉔：resolve_for_retrieval()的调用点
        从`_dispatch_with_planner()`搬到了这里】真机验证⑳时发现：单查询
        （`decompose()`判定不需要拆分）分支以前直接把原始文本转交到这里，
        `route_query()`能借助`recent_context`正确把"它现在进展到哪一步
        了？"分类成route="history"，但`_try_answer_from_history()`实际
        检索用的还是带代词的原句，跟"烛龙"毫无字面重合，查不到（完整
        现象见模块开头"⑳"一节"真机验证结果"）。

        第一版修复（㉑）把`resolve_for_retrieval()`加在了
        `_dispatch_with_planner()`里、路由判断之前，对每一句单查询
        （不管闲聊、陈述句还是真正需要检索的问题）无差别生效。真机验证
        ㉑时（同一轮测试里紧接着的一句纯陈述闲聊"今天天气有点冷，中午
        吃了碗泡面。"）暴露出这个位置选错了：`resolve_for_retrieval()`
        在"当前这句话本身完整、但`recent_context`里恰好有一句内容更
        丰富/更具体的旧话"这种场景下，会把当前这句话整个替换成旧话内容
        （"今天天气有点冷，中午吃了碗泡面。"被真实改写成了上一句"我们
        实验室这周启动了...烛龙..."），而这段被污染的文本又被原样转交
        到这里，连`route_query()`生成的闲聊回复内容都被带偏了——这次
        真机测试里最终回复"好的，这个我记下了～"够通用、侥幸没有露出
        破绽，但生成这句回复时`route_query()`实际看到的输入已经不是
        用户真正说的那句话，跟㉒里`synthesize()`可能被污染的q1内容"混进"
        最终回答是同一类风险，只是这次发生在general闲聊路径上。

        根因：`resolve_for_retrieval()`这个函数的设计动机，从一开始就
        只是为了服务`_try_answer_from_history()`/`_try_answer_from_
        local_kb()`这两个不经过LLM、只能做字面检索匹配的方法——general
        路由的回复由`route_query()`自己生成，wiki路由靠`route_query()`
        自己抽取的keyword检索，两者都不需要一份额外改写过的检索文本。
        把调用放在路由判断**之前**，等于对general/wiki这两条原本用不上
        它的路由也无差别生效，扩大了这个函数本该覆盖的范围；扩大范围
        之后又叠加了一个真实存在的prompt弱点——模型在"一句朴素、跟上下文
        无关的当前输入"和"上下文里一句更突出、更具体的旧话"之间，容易把
        两者搞混，这跟㉒里`decompose()`的失败模式是同一种"抽象指令写得
        对，具体上下文形状下模型泛化不稳"。

        修复：`resolve_for_retrieval()`的调用点搬到这里，改成路由判断
        **之后**，只在确定要走route="history"/route="local_kb"这两条
        字面检索路径时才对`query`做一次改写，再把改写后的文本交给
        `_try_answer_from_history()`/`_try_answer_from_local_kb()`；
        route="general"/route="wiki"两条分支完全不再调用它，用的还是
        `route_query()`自己接收到的原始`query`——这两条路径本来就不需要
        它，也是这次真机测试暴露问题之前从未出过错的两条路径。
        `_dispatch_with_planner()`对应改回"单查询原样委托给这里"，不再在
        路由之前做任何改写。

        代价：比㉑第一版实现更省——以前是"每一句单查询都多一次LLM调用"，
        现在收窄成"只有真的路由到history/local_kb时才多一次"，wiki/
        general这两条更常见的路径反而比㉑第一版更便宜（回到了改动前的
        调用次数）。

        已知没有一并处理的边界（如实记录）：route="wiki"分支内部
        `_try_answer_from_wiki_page()`没命中、回退到本地知识库时
        （紧接着这一行的`_try_answer_from_local_kb(query, ...)`），用的
        依然是未经`resolve_for_retrieval()`改写的原始`query`——这个具体
        分支从㉑之前到现在都没有做过这一步改写，不属于这次真机测试暴露
        的范围，也没有真机证据表明它有问题，这次不顺带扩大范围去改。

        route_future：推测执行时（`_write_and_dispatch()`）已经跟画像写入、拆分判断
        同时发出去的那次`route_query()`，这里直接取它的结果，不再重新调一次；
        输入跟下面这行完全相同（同一句话、同一份背景）。
        """
        if route_future is not None:
            route = route_future.result()
        else:
            route = route_query(query, self.llm_client, recent_context=recent_context)
        if route is None:
            return None

        print(f"[conversation_memory] 路由结果：{route['route']}，"
              f"needs_search={route.get('needs_search', False)}，"
              f"检索主题={route.get('retrieval_query', '')!r}")
        if route["route"] == "general":
            if route.get("needs_search"):
                # 客观问题（主观/客观区分见wiki_router.py的needs_search）：走联网
                # 搜索兜底链并标注来源；不算闲聊（是真正的信息问答）
                sub = self._answer_beyond_sources(query, extra_context=recent_context,
                                                  search_query=route.get("search_query", ""))
                if sub is not None:
                    return sub.display, False
                # 联网和自身知识都没能生成：这时route给的reply只是"我帮你查一下～"这种
                # 占位语，发出去就没下文了（2026-10-01实测），如实说明
                return _LOOKUP_FAILED_REPLY, False
            return route["reply"], route["is_chitchat"]

        if route["route"] == "history":
            resolved_query = route.get("retrieval_query") or resolve_for_retrieval(query, recent_context, self.llm_client)
            if resolved_query != query:
                print(f"[conversation_memory] 路由到route=\"history\"，检索文本已改写："
                      f"{query!r} -> {resolved_query!r}")
            sub = self._try_answer_from_history(query, extra_context=recent_context, retrieval_query=resolved_query)
            if sub is None:
                # L2没能翻出可信内容：老实说"没翻到"，不调用LLM去凭空猜
                # "之前说过什么"——见 _HISTORY_MISS_REPLY 定义处的说明。
                reply = _HISTORY_MISS_REPLY
            else:
                reply = sub.display
            return reply, False

        if route["route"] == "wiki":
            sub = self._try_answer_from_wiki_page(query, route["keyword"], extra_context=recent_context,
                                                   cargo_profession=route.get("cargo_profession", ""),
                                                   cargo_rarity=route.get("cargo_rarity", ""))
            if sub is None:
                # wiki查不到：先联网搜索/大模型自身知识（明日方舟的问题退回作者的
                # 生信课题资料没有意义）；兜底也走不通时才沿用原来的本地知识库兜底
                sub = self._answer_beyond_sources(query, extra_context=recent_context, domain=self.WIKI_DOMAIN)
            if sub is None:
                sub = self._try_answer_from_local_kb(query, extra_context=recent_context)
            return sub.display, False

        # route["route"] == "local_kb"
        if route.get("kb_query_kind") == "catalog":
            sub = self._try_answer_from_catalog(route.get("catalog_scope", "all"))
            if sub is not None:
                return sub.display, False
        resolved_query = route.get("retrieval_query") or resolve_for_retrieval(query, recent_context, self.llm_client)
        if resolved_query != query:
            print(f"[conversation_memory] 路由到route=\"local_kb\"，检索文本已改写："
                  f"{query!r} -> {resolved_query!r}")
        sub = self._try_answer_from_local_kb(query if route.get("retrieval_query") else resolved_query, extra_context=recent_context,
                                             scope=route.get("kb_scope", "all"),
                                             jc_session=route.get("jc_session", ""), retrieval_query=resolved_query)
        if not sub.ok:
            fallback = self._answer_beyond_sources(resolved_query, extra_context=recent_context, kb_miss=True)
            if fallback is not None:
                sub = fallback
        return sub.display, False

    # ---------- 检索架构升级：多跳/复合问题的编排层 ----------
    # 完整设计动机见 query_planner.py 模块开头，这里只有落地到
    # ConversationSession 这一侧的编排逻辑——之所以没有整个放进
    # query_planner.py，是因为执行每一跳需要用到 self.kb_pipeline/
    # self.llm_client 这些 ConversationSession 已有的状态和方法
    # （_try_answer_from_history() 等），query_planner.py 只放不依赖
    # ConversationSession 状态的纯逻辑（分解、改写、合并这几次LLM调用）。

    @staticmethod
    def _log_resolve_for_retrieval(original: str, resolved: str, depends_on: List[str]) -> None:
        """跟`prts_wiki_client`那些"关键词xxx命中页面"提示同一个模式的
        console-only调试打印——纯print()，不会拼进返回给用户的回复文本。
        补这一条的原因：2026-09第二轮真机验证里，用户想专门验证"有依赖的
        子查询"会不会正确触发`resolve_for_retrieval()`改写，但这条路径本身
        没有像wiki查找那样天然可观察的输出，光看最终回答质量很难跟"这一跳
        压根没走到这个分支"或者"L1在更前面就抢答了"这些情况区分开——加一行
        打印直接把"改写前/改写后"亮出来，比靠回答内容反推可靠。

        2026-09第三轮真机测试后的更新（F真机测试第一轮里的Fix A，见模块开头
        同名小节）：`resolve_for_retrieval()`原来只在子查询路由到local_kb/
        history时调用，wiki路由完全依赖`route_query()`自己解析代词/依赖信息，
        结果在"能天使的同事"这个fan-out场景里选错了关键词。修复后只要
        `depends_on`非空，不管最终会路由到哪个分支，都统一先调用一次
        `resolve_for_retrieval()`——所以这里的打印和文档也要跟着改口径，
        不能再说"路由到local_kb/history"，而是"只要这个子查询有依赖"就会
        触发。`depends_on`为空时`resolved==original`（没有调用
        `resolve_for_retrieval()`），这种情况不打印，避免刷屏。
        """
        if depends_on and resolved != original:
            print(f"[query_planner] 有依赖的子查询，"
                  f"检索文本已改写：{original!r} -> {resolved!r}")
        elif depends_on:
            print(f"[query_planner] 有依赖的子查询，"
                  f"resolve_for_retrieval()原样返回（没有需要改写的代词/省略）："
                  f"{original!r}")

    def _execute_sub_query(self, query_text: str, depends_on: List[str],
                            recent_context, dep_results: Dict[str, SubAnswer],
                            detect_candidates: bool = False) -> Optional[SubAnswer]:
        """执行多跳编排器拆出的一个子查询，返回结构化的`SubAnswer`（而不是
        `_dispatch_with_router()`那种"最终展示文本+is_general_chat"二元组）
        ——这个方法服务的是编排器内部，编排器需要知道这一跳有没有查到确定
        的东西，才能决定后面依赖它的子查询要不要继续、以及要把什么内容
        当背景注入下一步。

        依赖注入：如果这个子查询有depends_on，把它依赖的每个子查询的结果
        （优先core，core为空退回display）格式化成"已查到的信息"，追加到
        recent_context后面，一起传给route_query()——跟真实的L0对话历史
        走的是同一套指代解析机制，不是另外发明一套占位符替换逻辑，完整
        理由见 query_planner.py 模块开头"依赖怎么解析"一节。

        【F真机测试暴露的真实bug：route_query()自己解析代词，会在依赖
        信息和更早的L0历史之间选错】原来这里只在route="local_kb"/"history"
        时才调用`resolve_for_retrieval()`把query文本先改写一遍，route="wiki"
        被认为不需要——因为keyword提取本来就是route_query()自己在做，
        理论上它能直接结合context_rows解析代词。真机测试"能天使的同事的
        种族包含哪些"暴露了这个假设不成立：`_answer_multi_hop_with_fanout()`
        对"大帝/德克萨斯/空"三个候选分别执行依赖它的"他们的种族分别是
        什么？"，每个分支context_rows里明明只注入了一个候选名字，但因为
        这次对话此前一轮已经问过"能天使的同事包含哪些种族"、L0历史里能天使
        的份量更重，route_query()三次都错误地把keyword解析回了"能天使"，
        没有采信这一跳新注入的那个候选。根因是route_query()要同时做"分类
        +解析代词+提取keyword"这三件事，指代消解只是这个复合判断的一个
        分支，遇到新注入信息和旧历史抢显著性时不够稳。

        修复：不再假设route="wiki"不需要改写，改成不管路由到哪一条，
        只要depends_on非空，一律先用`resolve_for_retrieval()`把代词/省略
        指代显式替换进query文本本身（比如把"他们的种族分别是什么？"配上
        背景"已查到的信息：大帝"，改写成"大帝的种族是什么？"），再拿改写
        后的文本去调用route_query()——把"该信任哪个候选"这个判断从
        route_query()复合判断里的一个隐藏分支，变成显式写进文本里的
        确定内容，不用再指望route_query()自己在竞争的上下文里选对。
        独立子查询（depends_on为空）本身就是完整问句，不需要改写，这部分
        行为完全不变。

        detect_candidates：F（多候选页面消歧）新增，默认False、不传时跟
        改动前逐字一致。由调用方（`_answer_multi_hop()`）根据"这个子查询
        的结果会不会被后面的子查询依赖"决定要不要传True；本轮只在
        route="wiki"这条路径接了检测，见`_try_answer_from_wiki_page()`
        同名参数的说明。
        """
        context_rows = list(recent_context) if recent_context else []
        for dep_id in depends_on:
            dep = dep_results.get(dep_id)
            if dep is not None:
                dep_text = dep.core if (dep.ok and dep.core) else dep.display
                context_rows.append({"role": "assistant", "text": f"（已查到的信息）{dep_text}"})

        resolved_text = (resolve_for_retrieval(query_text, context_rows, self.llm_client)
                         if depends_on else query_text)
        self._log_resolve_for_retrieval(query_text, resolved_text, depends_on)

        route = route_query(resolved_text, self.llm_client, recent_context=context_rows)
        if route is None:
            return None

        if route["route"] == "general":
            # 分解器理应只拆信息性问题，这里出现general是防御性场景（比如
            # 某个子查询意外被判成闲聊）——route那次LLM调用里已经顺带生成
            # 好了可以直接用的自然语言回复，直接原样当这一跳的结果。
            if route.get("needs_search"):
                sub = self._answer_beyond_sources(resolved_text, extra_context=context_rows,
                                                  search_query=route.get("search_query", ""))
                if sub is not None:
                    return sub
                return SubAnswer(display=_LOOKUP_FAILED_REPLY, core="", ok=False)
            return SubAnswer(display=route["reply"], core=route["reply"], ok=True)

        if route["route"] == "wiki":
            sub = self._try_answer_from_wiki_page(resolved_text, route["keyword"], extra_context=context_rows,
                                                   detect_candidates=detect_candidates,
                                                   cargo_profession=route.get("cargo_profession", ""),
                                                   cargo_rarity=route.get("cargo_rarity", ""))
            if sub is not None:
                return sub
            # wiki没查到：先走联网搜索/大模型自身知识兜底，走不通才退回本地
            # 知识库（用的是同一份已经改写过的文本，不需要再改写一遍）
            sub = self._answer_beyond_sources(resolved_text, extra_context=context_rows, domain=self.WIKI_DOMAIN)
            if sub is not None:
                return sub
            return self._try_answer_from_local_kb(resolved_text, extra_context=context_rows)

        if route["route"] == "history":
            return self._try_answer_from_history(resolved_text, extra_context=context_rows,
                                                  retrieval_query=route.get("retrieval_query"))

        # route["route"] == "local_kb"
        if route.get("kb_query_kind") == "catalog":
            sub = self._try_answer_from_catalog(route.get("catalog_scope", "all"))
            if sub is not None:
                return sub
        sub = self._try_answer_from_local_kb(resolved_text, extra_context=context_rows,
                                             scope=route.get("kb_scope", "all"),
                                             jc_session=route.get("jc_session", ""),
                                             retrieval_query=route.get("retrieval_query"))
        if not sub.ok:
            fallback = self._answer_beyond_sources(resolved_text, extra_context=context_rows, kb_miss=True)
            if fallback is not None:
                return fallback
        return sub

    def _answer_multi_hop(self, original_text: str, plan: DecomposePlan, recent_context) -> str:
        """执行`query_planner.decompose()`拆出的多个子查询：按列表顺序
        （`_validate_plan()`已经保证这是一个合法的拓扑序，见该函数说明）
        逐个执行，独立的子查询（depends_on为空）互不影响；有依赖的子查询
        会拿到它依赖的前面子查询的结果当背景，见`_execute_sub_query()`。

        失败处理区分两种情况：`referenced_ids`预先算出"哪些子查询id被后面
        的子查询依赖"——如果失败的这一跳被后面依赖，前提缺失，继续往下会
        变成带着缺失前提瞎猜，诚实终止整个流程；如果没有任何子查询依赖它
        （独立并列场景里失败的那一个，或者压根就是最后一跳），只把这一跳
        记成失败，不影响其它独立子查询继续执行，最后合并回答时如实说明
        "这部分没查到"，不是全部推倒重来。

        F（多候选页面消歧）：`referenced_ids`里的每个子查询同时也会传
        `detect_candidates=True`，一旦发现某个被依赖的子查询实际上有不止
        一个候选（`SubAnswer.candidates`），立刻停止按原计划往下执行、
        转交给`_answer_multi_hop_with_fanout()`处理，不再走下面这条默认
        的单分支`synthesize()`路径——完整设计见`query_planner.py`模块开头
        "F正式实现"一节。

        【F真机测试第二轮暴露的真实问题：发现歧义前，依赖它的子查询会先
        白跑一次】早期实现是先把`plan["sub_queries"]`整个列表按顺序执行
        一遍、全部跑完才检查`ambiguous_ids`，这意味着即使q1（比如"能天使
        的同事有哪些"）产生了歧义，依赖它的q2（"他们的种族分别是什么"）
        还是会在主循环里先老老实实执行一次——而这次执行用的`dep_results`
        是q1"挑了一个候选说"的那句完整生成回答（不是具体候选名字本身），
        执行完`_answer_multi_hop_with_fanout()`发现歧义后又要重新按每个
        候选各执行一次q2，这一次主循环里的执行完全是白费的LLM调用，还会
        在真机日志里多打印一次容易让人误以为是重复bug的"改写"记录（用户
        真机测试"能天使的同事的种族包含哪些"时看到5条改写记录、以为拆解
        乱了，其中1条就是这次白跑，另外4条才是4个候选各自的fan-out分支）。
        修复：改成边执行边检查——一旦某个被依赖的子查询自己就产生了歧义，
        当场跳出主循环、交给fan-out处理，不再执行任何依赖它的子查询；跟
        这次歧义无关的独立子查询（比如同一个plan里还有别的互不依赖的
        信息需求）不受影响，仍然在主循环里正常执行到底。
        """
        referenced_ids = {dep for sq in plan["sub_queries"] for dep in sq["depends_on"]}

        if parallel_enabled():
            outcome = self._run_plan_parallel(plan, recent_context, referenced_ids)
            if isinstance(outcome, str):
                return outcome  # 被依赖的子问题失败，诚实终止
            sub_results, ambiguous_ids = outcome
            if ambiguous_ids:
                return self._answer_multi_hop_with_fanout(original_text, plan, recent_context,
                                                            sub_results, ambiguous_ids)
            merged = self._merge_sub_results(original_text, plan, sub_results)
            return merged if merged is not None else self._format_multi_hop_fallback(plan, sub_results)

        sub_results: Dict[str, SubAnswer] = {}
        ambiguous_ids: List[str] = []
        for sq in plan["sub_queries"]:
            if ambiguous_ids and set(sq["depends_on"]) & set(ambiguous_ids):
                # 这个子查询依赖的某个前置结果已经确认有多个候选，执行它
                # 只会用到"还没拆分成具体候选"的过渡态背景，注定要在
                # fan-out里按候选重新执行一遍——不在这里白跑，见上面新增
                # 的docstring小节。
                continue

            answer = self._execute_sub_query(sq["query"], sq["depends_on"], recent_context, sub_results,
                                              detect_candidates=(sq["id"] in referenced_ids))
            if answer is None or not answer.ok:
                if sq["id"] in referenced_ids:
                    return (f"要回答这个问题，得先查清楚「{sq['query']}」，"
                            f"但没能找到可信的答案，没法继续往下查了。")
                sub_results[sq["id"]] = answer if answer is not None else SubAnswer(
                    display="没有查到可信的答案", core="", ok=False)
                continue
            sub_results[sq["id"]] = answer
            if sq["id"] in referenced_ids and len(answer.candidates) > 1:
                ambiguous_ids.append(sq["id"])

        if ambiguous_ids:
            return self._answer_multi_hop_with_fanout(original_text, plan, recent_context,
                                                        sub_results, ambiguous_ids)

        merged = self._merge_sub_results(original_text, plan, sub_results)
        if merged is not None:
            return merged

        # 合成失败（LLM不可用/调用失败）：退回更朴素的拼接格式，老实展示
        # 每个子查询各自的结果，不隐藏"没能合并成一句话"这件事本身——跟
        # llm_generate.py"生成失败退回展示证据"是同一套降级哲学。
        return self._format_multi_hop_fallback(plan, sub_results)

    def _run_plan_parallel(self, plan: DecomposePlan, recent_context, referenced_ids):
        """第二档（2026-09-28）：按依赖图调度子问题——依赖都有结果了就开跑，互不
        依赖的同时跑（LLMCompiler的做法：不按"层"一批批等，谁的前置先完成谁先走）。
        每个子问题的执行逻辑、失败/歧义的判断规则跟上面的串行循环完全一样，只是
        执行顺序变了：
          - 被依赖的子问题失败：不再启动新的子问题，已经在跑的等它们结束（Python
            线程没法强行中断；等完再返回，也保证这条消息处理完之前同一个对话的
            下一条不会插进来），然后诚实终止，返回终止提示（str）；
          - 被依赖的子问题出现多个候选：依赖它的子问题不启动（交给fan-out按候选
            各跑一遍），跟它无关的照常跑完。
        返回 (sub_results, ambiguous_ids)，sub_results按plan顺序排好——合并回答时
        引用编号按这个顺序排，不能随完成先后变。
        """
        sub_queries = plan["sub_queries"]
        sub_results: Dict[str, SubAnswer] = {}
        ambiguous_ids: List[str] = []
        pending = list(sub_queries)
        running = {}  # Future -> sq
        abort_reply = None

        while True:
            if abort_reply is None:
                for sq in list(pending):
                    deps = sq["depends_on"]
                    if set(deps) & set(ambiguous_ids):
                        pending.remove(sq)  # 见串行循环里同一条注释：留给fan-out，不白跑
                        continue
                    if all(d in sub_results for d in deps):
                        pending.remove(sq)
                        fut = submit(self._execute_sub_query, sq["query"], deps, recent_context,
                                     dict(sub_results), detect_candidates=(sq["id"] in referenced_ids),
                                     tag=sq["id"])
                        running[fut] = sq
            if not running:
                break
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for fut in done:
                sq = running.pop(fut)
                answer = fut.result()
                if answer is None or not answer.ok:
                    if sq["id"] in referenced_ids:
                        if abort_reply is None:
                            abort_reply = (f"要回答这个问题，得先查清楚「{sq['query']}」，"
                                           f"但没能找到可信的答案，没法继续往下查了。")
                        continue
                    sub_results[sq["id"]] = answer if answer is not None else SubAnswer(
                        display="没有查到可信的答案", core="", ok=False)
                    continue
                sub_results[sq["id"]] = answer
                if sq["id"] in referenced_ids and len(answer.candidates) > 1:
                    ambiguous_ids.append(sq["id"])

        if abort_reply is not None:
            return abort_reply
        ordered = {sq["id"]: sub_results[sq["id"]] for sq in sub_queries if sq["id"] in sub_results}
        ambiguous_ids.sort(key=[sq["id"] for sq in sub_queries].index)
        return ordered, ambiguous_ids

    def _merge_sub_results(self, original_text: str, plan: DecomposePlan,
                           sub_results: Dict[str, SubAnswer]) -> Optional[str]:
        """多跳/复合问题的合并，按来源分段标注（用户的要求：一个问题的答案
        分别来自多种来源时，要在回答中分段做好标注）。

        全部来自知识来源（wiki/本地知识库/历史/画像）：跟以前一样交给
        synthesize()合并成一段连贯的话。有部分来自联网搜索/大模型自身知识时：
        知识来源的部分照旧synthesize()合并（只把这几部分的问题交给它，免得它
        对着完整原问题说"另一部分没查到"），联网搜索/大模型自身知识的部分各自
        单独成段、原样带上自己的参考链接和标注，不交给LLM重新合并——合并时
        标注容易丢，这里用代码保证每一段的来源标注一字不差地留在它自己那一段后面。
        返回None表示synthesize()合并失败，调用方沿用原来的朴素拼接兜底。
        """
        unverified_ids = [sq["id"] for sq in plan["sub_queries"]
                          if sq["id"] in sub_results and sub_results[sq["id"]].source in ("web", "llm")]
        if not unverified_ids:
            merged = synthesize(original_text, plan, sub_results, self.llm_client)
            if merged is None:
                return None
            # 全部来自知识来源：synthesize()合并成一段话后，各部分原本的"引用来源"
            # 会丢（知识库扩充的真机验证里，"作者做过几次JC、都讲了哪些"拆成两部分、
            # 合并后回答里一条来源都没有）。单一来源的回答照旧只附引用、不加
            # 【以上内容来自…】（用户的要求），这里补一份去重后的引用列表
            refs = list(dict.fromkeys(r for sub in sub_results.values() if sub.ok for r in sub.refs))
            if not refs:
                return merged
            return merged + "\n\n引用来源：\n" + "\n".join(f"[{i}] {r}" for i, r in enumerate(refs, start=1))

        parts = []
        verified_sqs = [sq for sq in plan["sub_queries"] if sq["id"] not in unverified_ids]
        if verified_sqs:
            verified_plan: DecomposePlan = {"sub_queries": verified_sqs}
            verified_results = {sq["id"]: sub_results[sq["id"]] for sq in verified_sqs if sq["id"] in sub_results}
            focus = "；".join(sq["query"] for sq in verified_sqs)
            merged = synthesize(focus, verified_plan, verified_results, self.llm_client)
            text = merged if merged is not None else self._format_multi_hop_fallback(verified_plan, verified_results)
            # 知识来源那一段也单独标出来源（用户的要求）：否则后面网络搜索那段末尾的
            # "以上内容来自网络搜索"会让人以为整个回答都是搜来的；而且synthesize()
            # 合并后这一段原本的"引用来源"会丢，这里顺带补回来
            label = self._verified_source_label(verified_results.values())
            parts.append(f"{text}\n\n{label}" if label else text)
        for sq in plan["sub_queries"]:
            if sq["id"] in unverified_ids:
                parts.append(f"关于「{sq['query']}」：\n{sub_results[sq['id']].display}")
        return "\n\n".join(parts)

    @staticmethod
    def _verified_source_label(subs) -> str:
        """复合问题里知识来源那一段的来源标注，比如
        "【以上内容来自明日方舟wiki：prts.wiki/能天使】"。只在多来源的回答里用，
        单一来源的回答照旧只附引用（用户的要求）。"""
        kinds = {"wiki": "明日方舟wiki", "kb": f"{OWNER_NAME}的资料", "history": "之前的对话记录",
                 "profile": "你之前告诉过我的信息"}
        groups: Dict[str, List[str]] = {}
        for sub in subs:
            if sub.ok and sub.source in kinds:
                groups.setdefault(sub.source, [])
                groups[sub.source] += [r for r in sub.refs if r not in groups[sub.source]]
        if not groups:
            return ""
        parts = []
        for kind, refs in groups.items():
            detail = [r for r in refs if r != kinds[kind]]  # history/profile的ref就是类别名本身，不重复写
            parts.append(f"{kinds[kind]}：{'、'.join(detail)}" if detail else kinds[kind])
        return f"【以上内容来自{'；'.join(parts)}】"

    @staticmethod
    def _format_multi_hop_fallback(plan: DecomposePlan, sub_results: Dict[str, SubAnswer]) -> str:
        """`synthesize()`合成失败时的朴素拼接兜底格式，`_answer_multi_hop()`
        和`_answer_multi_hop_with_fanout()`共用同一份——两处失败降级的
        展示逻辑本该一致，抽出来避免维护两份几乎一样的拼接代码。
        """
        lines = [f"（这个问题拆成了{len(plan['sub_queries'])}部分分别查证）"]
        for sq in plan["sub_queries"]:
            answer = sub_results.get(sq["id"])
            text = answer.display if answer is not None else "没有查到可信的答案"
            lines.append(f"- {sq['query']}\n  {text}")
        return "\n".join(lines)

    def _answer_multi_hop_with_fanout(self, original_text: str, plan: DecomposePlan, recent_context,
                                       sub_results: Dict[str, SubAnswer], ambiguous_ids: List[str]) -> str:
        """F（多候选页面消歧）："都查一遍、合并回答"——用户确认的处理策略。
        完整设计动机、范围限制见`query_planner.py`模块开头"F正式实现"
        一节，这里是落地执行的部分。

        只处理"恰好一个被依赖的子查询产生歧义"这一种拓扑（目前唯一的真实
        案例"玛恩纳的侄女的职业是什么"就是这个形状）。如果同一个plan里
        不止一个子查询同时产生歧义，为避免组合爆炸，保守退化成每个都只
        取第一个候选、按原来的单分支流程处理——不报错、不崩溃，只是回到
        "不做消歧"的旧行为，如实打印一条提示。

        【F真机测试第二轮暴露的真实bug：fan-out分支里，多轮L0历史会把
        当前分支的候选名字"挤"没】Fix A让`_execute_sub_query()`不管路由到
        哪条分支都会先调用`resolve_for_retrieval()`把代词换成具体名字，
        理论上足够了；但"能天使的同事的种族包含哪些"真机测试证明还不够：
        这次对话前一轮刚问过一次同义问法，L0历史里已经完整提到过"德克萨斯
        /大帝/空/可颂"这四个名字，`resolve_for_retrieval()`拿到的背景
        同时有这段更早、更完整的历史和这一分支新注入的"已查到的信息）
        德克萨斯"，两者都摆在同一个"最近对话："文本块里、没有优先级之分，
        LLM有时候会被更显眼、更完整的历史带跑，把"他们"解析回"德克萨斯、
        大帝、空、可颂"这一整份名单，而不是这个分支该用的那一个名字——
        4个分支里只有1个真的解析对了，其余分支查到的候选各自的种族全部
        错误或缺失。

        根因是`_execute_sub_query(..., recent_context, ...)`这里的调用
        原样把`recent_context`（这一整段对话更早的历史）传了进去，而fan-out
        分支的每个下游子查询实际上完全不需要它：这个分支要问的是谁，已经
        在上面`branch_context[ambiguous_id] = SubAnswer(..., core=candidate, ...)`
        这一行用代码直接确定好了，不需要也不应该再让LLM去多轮历史里自己
        猜——多喂给它的这段历史只有添乱的风险，没有额外帮助。修复：fan-out
        分支执行依赖它的下游子查询时，不再传`recent_context`（传`None`），
        只保留`branch_context`这份已经写死了具体候选名字的背景，把"这一步
        该问的是谁"从"LLM在两段互相竞争的文本里挑一个"变成"代码已经确定、
        没有歧义可言"。`_RESOLVE_SYSTEM_PROMPT`本身也补了一条"背景里如果
        同时出现`已查到的信息`标记和普通历史、两者指向不一样的内容，以
        `已查到的信息`为准"的优先级说明，作为非fan-out路径（`_answer_
        multi_hop()`主循环里正常执行的依赖子查询）同样可能遇到类似竞争
        时的第二道防线——但fan-out这里能直接去掉噪声源，比只靠一条prompt
        指令更可靠，所以对fan-out分支采用更彻底的做法。
        """
        if len(ambiguous_ids) > 1:
            print(f"[conversation_memory] 多个子查询同时出现候选歧义"
                  f"（{ambiguous_ids}），这种组合场景不在当前fan-out范围内，"
                  f"退回只用各自第一个候选处理。")
            for sid in ambiguous_ids:
                sub_results[sid].candidates = sub_results[sid].candidates[:1]
            merged = self._merge_sub_results(original_text, plan, sub_results)
            return merged if merged is not None else self._format_multi_hop_fallback(plan, sub_results)

        ambiguous_id = ambiguous_ids[0]
        ambiguous_sq = next(sq for sq in plan["sub_queries"] if sq["id"] == ambiguous_id)
        candidates = sub_results[ambiguous_id].candidates
        dependents = [sq for sq in plan["sub_queries"] if ambiguous_id in sq["depends_on"]]

        # 歧义子查询本身的展示替换成"查到了几个候选"，不再是原来那句只
        # 挑了一个候选说的生成回答——这样交给synthesize()时，最终回答里
        # 会带上"玛恩纳的侄女有瑕光和临光两位"这类交代候选来源的内容。
        candidates_joined = "、".join(candidates)
        fanout_results: Dict[str, SubAnswer] = dict(sub_results)
        fanout_results[ambiguous_id] = SubAnswer(
            display=f"查到{len(candidates)}个符合条件的候选：{candidates_joined}",
            core=f"有{len(candidates)}个候选：{candidates_joined}",
            ok=True,
        )
        fanout_sub_queries: List = [ambiguous_sq]

        # 各分支互不依赖：先全部发出去同时跑（第二档，2026-09-28；并行关闭时submit()
        # 当场同步执行，等于原来的串行），再按候选顺序收结果，合并时顺序固定
        branch_futures = {}
        for candidate in candidates:
            branch_context = dict(sub_results)
            branch_context[ambiguous_id] = SubAnswer(display=candidate, core=candidate, ok=True)
            for sq in dependents:
                # 故意不传recent_context（传None）——这个分支该问的是谁已经
                # 在上面一行用代码写死进branch_context了，不需要也不应该
                # 再让LLM去更早的多轮历史里自己猜，见本方法上面新增的
                # docstring小节。
                branch_futures[(candidate, sq["id"])] = submit(
                    self._execute_sub_query, sq["query"], sq["depends_on"], None, branch_context,
                    tag=f"{sq['id']}:{candidate}")

        for candidate in candidates:
            for sq in dependents:
                answer = branch_futures[(candidate, sq["id"])].result()
                branch_id = f"{sq['id']}__{candidate}"
                fanout_sub_queries.append({"id": branch_id, "query": f"[{candidate}] {sq['query']}",
                                            "depends_on": [ambiguous_id]})
                fanout_results[branch_id] = answer if (answer is not None and answer.ok) else SubAnswer(
                    display=f"没有查到{candidate}对应的可信答案", core="", ok=False)

        # 跟这次歧义无关的独立子查询原样保留，不受fan-out影响。
        for sq in plan["sub_queries"]:
            if sq["id"] != ambiguous_id and ambiguous_id not in sq["depends_on"]:
                fanout_sub_queries.append(sq)

        fanout_plan: DecomposePlan = {"sub_queries": fanout_sub_queries}
        merged = self._merge_sub_results(original_text, fanout_plan, fanout_results)
        if merged is not None:
            return merged
        return self._format_multi_hop_fallback(fanout_plan, fanout_results)

    def _dispatch_with_planner(self, query: str, recent_context=None, plan_future=None,
                               route_future=None) -> Optional[Tuple[str, bool]]:
        """`_dispatch_with_router()`之上新加的入口——先问
        `query_planner.decompose()`这句话要不要拆成多个子查询。绝大多数
        情况（不需要拆）时，`decompose()`返回单子查询的plan，原样委托给
        `_dispatch_with_router()`。

        需要拆分时才走`_answer_multi_hop()`，返回(合并后的回复, False)——
        多跳问题的答案是事实性内容，不是闲聊，`is_general_chat`给False，
        正常留在L2候选池里，跟其它wiki/local_kb/history命中保持一致。

        `decompose()`内部已经处理了"client不可用"的退化（直接返回单查询
        plan，不发起LLM调用），这里不需要重复判断；`_dispatch_with_router()`
        在client不可用时会返回None，这个方法在单查询分支下会原样把None
        传出去，`user_says()`看到None就退回2.1的legacy链路，行为链路
        完整、不需要在这里单独处理"LLM不可用"这一种情况。

        【㉑单查询代词解析缺口，以及㉔：这次修复现在放在`_dispatch_with_
        router()`里，不在这里】真机验证⑳时发现单查询分支需要
        `resolve_for_retrieval()`才能让检索用上代词解析（完整背景见模块
        开头"⑳"一节）；但第一版实现（㉑）把这次调用直接放在这个方法里、
        路由判断之前，对每一句单查询（包括闲聊/陈述句）无条件生效，真机
        验证㉑时又暴露出这个位置选错了——完整现象、根因和修复见
        `_dispatch_with_router()`同名小节，这里只记结论：这个方法现在
        恢复成㉑之前"单查询原样委托给`_dispatch_with_router()`"的样子，
        不再在路由判断之前做任何改写。

        plan_future/route_future：推测执行时已经提前发出去的`decompose()`/
        `route_query()`，见`_write_and_dispatch()`。要拆分时，提前做的那次
        整句路由用不上（每个子问题各自再路由），直接丢掉。
        """
        if plan_future is not None:
            plan = plan_future.result()
        else:
            plan = decompose(query, self.llm_client, recent_context=recent_context)
        if len(plan["sub_queries"]) == 1 and not plan["sub_queries"][0]["depends_on"]:
            return self._dispatch_with_router(query, recent_context=recent_context, route_future=route_future)

        if route_future is not None:
            route_future.cancel()  # 还没开始就不跑了；已经在跑的让它跑完，结果不用
        reply = self._answer_multi_hop(query, plan, recent_context)
        return reply, False

    # ---------- 主入口 ----------

    def _write_and_dispatch(self, text: str, recent_context) -> Tuple[list, Optional[Tuple[str, bool]]]:
        """LLM可用时每条消息的主流程（2026-09-29改版）：写画像（`_l1_write()`）和
        回答（拆分→路由→检索/生成）互不截断、各做各的——记忆写入和回答分开，业界
        做法，见llm_l1.py模块开头。

        并行模式下三件事同时开始（推测执行，2026-09-28）：写画像、拆分判断、整句
        路由；要拆分时整句路由的结果丢掉。回复返回之前等画像写完，下一条消息一定
        用得上新画像。回答用的背景（含画像）在写画像之前取：这句话里新说的信息，
        回答时这句话本身就在输入里，不需要等画像写完；串行模式也按同样的顺序取，
        两种模式的输入完全一致。

        返回 (实际写入画像的内容, `_dispatch_with_planner()`的返回值)。
        """
        planner_context = self._build_planner_context(recent_context)
        l1_future = submit(self._l1_write, text, recent_context, tag="画像写入")
        if parallel_enabled():
            plan_future = submit(decompose, text, self.llm_client, recent_context=planner_context, tag="拆分判断")
            route_future = submit(route_query, text, self.llm_client, recent_context=planner_context,
                                  tag="路由判断")
            dispatched = self._dispatch_with_planner(text, planner_context, plan_future=plan_future,
                                                     route_future=route_future)
        else:
            dispatched = self._dispatch_with_planner(text, planner_context)
        return l1_future.result(), dispatched

    def _log_length_check(self, text: str, reply: str) -> None:
        """用户提了字数要求时，生成后量一下正文字数、只打日志（2026-09-29加的检查，见
        length_limit.py）。

        【2026-10-01去掉了超出就压缩那一步】原来正文超出上限20%以上会再调一次LLM压缩。
        实测压缩是最容易超时的一步：同一个压缩任务4~99秒不等，推理模型在思考里反复
        逐字数字数（"Let me count more carefully"），一次输出1.1万token最后只留130字。
        用户决定：字数靠大模型读问题自己控制，目的是"别长篇大论"，大致符合就行——
        这也是常见做法（字数要求原样交给模型，不做逐字精确控制）。日志留着，方便看
        实际达标情况。"""
        req = parse_length_request(text)
        if req is None:
            return
        _, body, _ = split_reply(reply)
        n = count_units(body, req.unit)
        if req.min_units and n < req.min_units:
            print(f"[length_limit] 用户要求{req.raw}，正文{n}{req.unit}，比要求短（不自动扩写）")
        elif exceeds_limit(req, body):
            print(f"[length_limit] 用户要求{req.raw}，正文{n}{req.unit}，超出较多（不压缩，原样回复）")

    def after_reply(self) -> None:
        """`user_says(defer_summary=True)`推迟下来的对话摘要更新。前端在回复发出去
        之后调用（qq_bot.py在同一个对话的锁里调，下一条消息不会读到更新到一半的
        摘要）。成熟的记忆系统也是回复之后在后台整理记忆，不让用户等。"""
        if self._summary_pending:
            self._summary_pending = False
            self._maybe_update_l0_summary()

    def user_says(self, text: str, defer_summary: bool = False) -> str:
        """见`_user_says()`。外面没有分步计时的trace时（终端版、演示脚本）在这里建一个，
        回复算完打印各步耗时（timing.py）；qq_bot自己建trace，把发送也算进去。只对
        真实联网的LLM打印（有provider_label的那种），测试替身不打印。"""
        if current_trace() is not None or not getattr(self.llm_client, "provider_label", None):
            return self._user_says(text, defer_summary)
        trace = Trace(f"{self.conversation_id} 消息 {text[:30]!r}")
        reply = trace.run(self._user_says, text, defer_summary)
        trace.emit()
        return reply

    def _user_says(self, text: str, defer_summary: bool = False) -> str:
        """处理用户的一句话：先存历史，再依次尝试L1->统一分流（LLM可用时）
        或L1->L2->L3本地知识库（LLM不可用时的legacy链路），最后把回复也
        存回历史（这样回复本身也成为"过去发生过的事"，虽然当前L2检索只看
        role='user'，这条assistant记录目前只用于get_context_window()展示，
        不参与检索）。

        L1：LLM可用时只负责写画像，跟回答并行、不截住消息（2026-09-29改版，见
        `_write_and_dispatch()`）；没接LLM时是2.1的规则版`_l1_step()`，能答就当场答。

        `is_general_chat`标记直接读`_dispatch_with_planner()`（内部绝大多数
        情况直接委托给`_dispatch_with_router()`，见该方法说明）的返回值
        决定，不再需要一个跨方法的临时标志位去事后确认——完整原因见
        `_dispatch_with_router()`的说明。

        recent_context一定要在`append_turn`存入当前这句话**之前**取，取的是
        "这句话发生之前"的L0窗口——见模块开头"L0上下文窗口接入L1判断"一节；
        顺序反过来的话，L1判断这句话时会把这句话自己也当成"最近对话"递给
        自己，没有意义。传给`_dispatch_with_planner()`的是`_build_planner_
        context()`包过一层的版本（原始recent_context+一行画像描述，见模块
        开头"画像可见性"一节），一路往下传给`route_query()`/`decompose()`
        解析代词/省略指代、以及生成步骤——`_l1_step()`用的还是原始
        recent_context，不需要重复喂一份画像（它已经直接拿到`profile`
        本身）。
        """
        recent_context = self.get_context_window()
        first_contact = self.auto_intro and not recent_context
        user_turn_index = self.store.append_turn(self.conversation_id, "user", text,
                                                 speaker=self.user_id if self.is_group else None)

        is_general_chat = False
        written = []
        if self.llm_client.is_real:
            reply = None
            written, dispatched = self._write_and_dispatch(text, recent_context)
        else:
            # 2.1（没接LLM）：规则版L1能答就当场答，答不了走下面的legacy链路
            reply = self._l1_step(text, recent_context)
            dispatched = None
        if reply is None:
            if dispatched is not None:
                reply, is_general_chat = dispatched
            else:
                # 2.1 legacy链路（LLM不可用）：L2先试，答不出来再落到本地
                # 知识库，跟这次重新设计之前的行为完全一致。
                sub = self._try_answer_from_history(text)
                if sub is None:
                    sub = self._try_answer_from_local_kb(text)
                reply = sub.display

        self._log_length_check(text, reply)

        if is_general_chat:
            self.store.mark_turn_general_chat(self.conversation_id, user_turn_index)

        self.store.append_turn(self.conversation_id, "assistant", reply)
        # 本轮的一问一答（以及刚补上的is_general_chat标记）同步进Chroma索引
        if self._history_index is not None:
            self._reconcile_history_index()
        if defer_summary:
            self._summary_pending = True
        else:
            self._maybe_update_l0_summary()
        if written:
            # "（已记下：…）"跟开场白一样只加在返回给用户的文本上，不存进对话历史
            reply = f"{reply}\n\n{self._memory_note(written)}"
        if first_contact:
            # 开场白只加在返回给用户的文本上，不存进对话历史——它不是这一轮
            # 真正的回答，存进去会被L2翻旧账检索到、也会挤占L0窗口
            return f"{intro_message(self.llm_client.is_real)}\n\n{reply}"
        return reply


if __name__ == "__main__":
    import os
    from pipeline import MemorySystemPipeline

    here = os.path.dirname(os.path.abspath(__file__))
    kb_dir = os.path.join(here, "..", "data", "knowledge_base")

    kb_pipeline = MemorySystemPipeline(kb_dir)
    store = MemoryStore(db_path=":memory:")
    # 不传 llm_client -> 自动用 NoLLMClient -> 走2.1的规则版逻辑。
    # 这个自测脚本本来就是2.1时代写的，用来验证"不接LLM"这条路径不回归，
    # 所以这里故意不传 llm_client，而不是接入 build_llm_client()。
    session = ConversationSession(kb_pipeline, store, user_id="demo_user")

    script = [
        "我叫长安，我的研究方向是单细胞谱系追踪",
        "我是谁？",                      # 应该命中L1，答出"称呼"
        "我的研究方向是什么？",           # 应该命中L1，答出"研究方向"
        "我刚才说的研究方向是什么来着？",  # 措辞不同，可能命中L1也可能落到L2，用来观察规则边界
        "我们组打算用Chroma做向量数据库先跑通逻辑",  # 不是profile问题，纯粹一句陈述，存进L2
        "我们组打算用什么做向量数据库来着？",         # 不该命中L1（没有对应规则），应该落到L2翻出上一句
        "变异检测常用什么工具？",         # 知识库里没把握的问题，应该落到L3并拒答
        "多组学数据整合有哪些方法？",     # 知识库能答的问题，应该落到L3并给出片段
    ]

    for turn in script:
        print("=" * 60)
        print(f"你说：{turn}\n")
        print(session.user_says(turn))
        print()
