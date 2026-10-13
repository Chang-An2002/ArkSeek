# -*- coding: utf-8 -*-
"""
统一分流路由判断 —— 决定这句话该查本地论文知识库、实时查prts.wiki、翻
之前对话里聊过的内容（L2），还是根本不需要检索任何证据（当成普通聊天来
回应），阶段二2.2专属功能（LLM不可用时直接跳过，没有规则版兜底，见下方
说明）。最初只是`conversation_memory.py`接入wiki查询这个L3第二知识来源的
第一块拼图（另一块是`prts_wiki_client.py`的`fetch_page_for_answer()`，见
README"进阶目标"一节），后来随着route="general"的加入、又随着这里说明的
route="history"的加入，已经演变成`conversation_memory.py`里"这句话到底该
谁答"这件事的唯一决策点，不再只是"wiki专属"的路由了（模块名没有跟着改，
是因为改名字要动的地方比较多，收益不成比例，这里用文档说明代替改名）。

【为什么要专门做路由，而不是"两边都查，比比看谁分数更可信"】
本地知识库（两篇生信论文）和prts.wiki（明日方舟游戏百科）是两个完全不相干的
领域——一个问题不可能同时跟"单细胞谱系追踪"和"干员天赋数值"都相关。硬要
两边都跑一遍（本地检索+精排，prts.wiki还要多发一次真实网络请求）纯粹浪费
开销，而且两边的置信度阈值本来就是分开标定的（README"知识库分开"这条设计
决定），分数不能跨领域比较、也没必要比。更合理的做法是先判断"这问题该去
哪查"，只跑那一条路径。

【route="general"的由来：从"chitchat/none各管一段"到"防幻觉只该管两个知识
库领域，其他一律像正常聊天助手一样回应"】
这条路由分两轮真机测试才走到现在这个样子，记录一下演变过程，比直接呈现
最终版本更能说明白为什么这么设计：

第一轮真机测试发现，单纯一句"你好"会被系统当成"知识性问题"处理——路由
判断不出该查哪个知识库，最终落到本地知识库的置信度gate，判定"检索不到
可靠内容"，回复一句冷冰冰的"没有检索到置信度足够的相关内容"。当时的修复
是加了route="chitchat"，命中就不再检索，直接返回同一次LLM调用里顺带生成
好的回复；同时保留route="none"，专门处理"问了一个具体问题、但两个知识库
都答不上来"的情况（比如"今天晚饭吃什么"），依然让它落到本地知识库gate走
正常拒答——这一版的逻辑是"chitchat不是在提问，none是在提问但答不上来"。

第二轮真机测试，用户指出这个区分本身就站错了立场：这个系统不应该只是一个
"分层记忆系统查询器"，遇到查不到证据的问题就一律拒答；它首先应该是一个
正常的对话助手，像任何一个大模型聊天产品一样能自然接住寒暄、闲聊、请求
建议/推荐这类不需要"证据支撑"的内容，只有当问题确实落在本地知识库或明日
方舟这两个专门做了检索的领域时，才需要用"检索不到就不瞎编"这条防幻觉原则
约束住。换句话说，"防幻觉"该防的是"在自己承诺过要可靠的领域里编造事实"，
不该被泛化成"任何答不出确切依据的话都不能说"——现实中"今天晚饭推荐吃什么"
根本不存在什么需要验证的"事实"，拒答反而是把系统的能力边界收得比实际应该
的更窄。

所以这一版把route="chitchat"和route="none"合并成一个更宽泛的route="general"：
不属于本地知识库、也不属于明日方舟wiki这两个特定领域的所有内容——寒暄、
闲聊、请求建议/推荐、观点性问题、跟这两个知识库都不相关的通用问题——统一
处理成"不需要检索，直接像普通聊天助手一样自然回应"，reply字段在同一次LLM
调用里顺带生成好。这不是放弃防幻觉，而是把防幻觉的适用范围收紧到它真正该
管的地方：route="wiki"/"local_kb"这两条路径的行为完全没变，查不到证据时
一样会经过置信度gate、一样会老实拒答（`_try_answer_from_router()`对这两条
路由的处理逻辑没有任何改动）——一句关于"明日方舟最新干员是谁"这种明确落在
wiki领域但查不到证据的问题，依然应该拒答而不是让LLM自由发挥瞎猜，这跟
route="general"要解决的问题不是同一类，完整的边界讨论见
`conversation_memory.py`模块开头"防幻觉的适用范围"一节。

这个改动也带来一个诚实的权衡，需要记录下来：route="general"比之前的
route="none"承担的风险更大——以前分类错了（该判wiki/local_kb却判成none），
最坏结果是"多余的拒答"，是一个安全但保守的失败模式；现在分类错了，最坏
结果是"该走检索防幻觉流程的问题被当成general自由回答"，可能产生编造的
内容。目前只能靠prompt里的分类标准和示例尽量把边界画清楚，没有更好的
兜底手段——如果后续测试发现误判频繁，需要回头收紧这条路由的判断标准，
而不是简单地又加一个新分支去打补丁。

【第三轮真机测试逼出的第4个分类：route="history"，把"L2该不该管"这件事
从"L2先跑再看"改成"路由先分清楚"】
route="general"合并完成之后，真机测试又暴露出一个更深的问题——完整讨论见
`conversation_memory.py`模块开头"L2语料该不该收录route="general"命中的
发言"一节，这里只记落地后的结论：只要`conversation_memory.py`维持"L2永远
排在路由判断前面无条件先跑一次"这个顺序，"吃了吗"这类闲聊只要跟历史里随便
哪一句长得像（哪怕是它自己之前的回复），就会被L2抢答，压根不会有机会走到
这个模块的路由判断——事后想给它打`is_general_chat`标记也无从打起，闲聊会
把自己的新回复也变成下一次同类问题的"历史证据"，滚雪球一样自我延续。

真正的修法不是在L2生成完之后打补丁，而是让路由判断本身能够分清"这看起来
是在追问之前对话里的具体内容"和"这就是眼下的闲聊/请求建议"——把原来
`route="general"`兜底覆盖的"跟两个知识库都不相关的内容"再拆出一块，专门
留给"像是在翻旧账"的问题，交给L2去试。这样`conversation_memory.py`就能把
"L2无条件先跑"这条硬顺序，换成"先问一次路由该谁答，四选一里选中history才
去跑L2"——路由判断和"这一轮算不算general闲聊"这件事在同一个地方一次性
定下来，不再需要L2抢答成功之后才知道。

【设计：照搬llm_l1.py同款的"一次LLM调用、严格JSON、解析失败就返回None"】
判断"这句话该怎么处理"本质上是个语义分类问题，跟L1判断"是不是在问自己的
画像"是同一类任务，所以直接复用同一个模式：一次LLM调用，输出严格JSON，
四选一路由：
    route="wiki"      —— 问的是明日方舟游戏里一个具体的、可查证的事实
                          （干员数值/天赋/技能/剧情/道具等），keyword给出
                          应该拿去查prts.wiki的关键词（通常是问题里的干员名/
                          道具名等专有名词）
    route="local_kb"  —— 问的是本地知识库能答的内容（单细胞/谱系追踪相关）
    route="history"   —— 问的是"之前这段对话里说过/提到过的具体内容"，
                          不是问用户自己稳定的身份信息（那是L1管的事），
                          典型标志是"来着""我刚才说的""你还记得（某件事）"
                          这类回指过去对话的表达——真正的答案要交给L2去
                          翻历史，这一步只负责识别"该去L2试试"，keyword/
                          reply都给空字符串
    route="general"   —— 以上都不是：寒暄/问候/客套话/致谢/告别、闲聊、
                          请求建议或推荐、笼统的观点性问题（包括"明日方舟
                          好玩吗"这种不是在问具体游戏机制的泛泛问题）、或者
                          任何跟这两个知识库、也跟"之前聊过的内容"都不相关
                          的通用问题——reply给出可以直接回复用户的自然语言
                          文字，不需要援引任何证据，像普通聊天助手一样自然
                          回应即可

这个改动带来一个新的、需要如实记录的权衡：以前L2无条件排在最前面，"这是
不是在问历史"这件事有真正的检索结果撑腰，不会凭空猜错；现在改成先猜分类
再决定要不要检索，如果一句真正该翻旧账的话，语言上又没有"来着"这类明显的
回指线索，会被误判成general，被自由聊天直接回答，而不是老实地去翻历史——
这是主动接受的新权衡，不是没考虑到，缓解手段只有把prompt里"history"这一类
的判断标准和示例尽量写清楚（尤其是"general"分类里明确提醒"拿不准就判给
history，不要在general里凭空编答案"），没有更好的兜底手段。

【这一步只在llm_client.is_real时才生效，没有更低配的版本可以退】
判断"这句话该怎么处理"本身就需要语义理解，不像L1的_l1_step()那样还有
fact_extractor.py的正则版可以退——路由这件事本来就是2.2才有的新能力，2.1
（NoLLMClient）根本没有"查prts.wiki"和"general直接回复"这两条路径存在过，
谈不上"退化"，直接返回None，调用方（conversation_memory.py）看到None就
跳过这整块、只走本地知识库L3，这跟2.1原本的行为完全一致（包括"你好"在2.1
下依然会被拒答——2.1本来就没有生成能力，这个体验缺口只能靠接LLM解决，不是
这次改动引入的新退步）。

【给以后再加route分类定的一条纪律：先问"这是不是一条全新的答题策略"】
从两选一（wiki/local_kb）一路加到现在的四选一（+general/history），每一次
都是真机测试暴露了具体问题之后加的——用户当时直接问了一句"这样一直往上堆
route是不是有屎山代码的嫌疑"，这条追问问得对，值得把界限定清楚，不能真的
无限加下去。

现在这四个分类不是任意的历史堆叠，而是对应四种本质不同的**答题策略**：
不需要证据、自由生成（general）；证据来自这次对话历史，用L2检索（history）；
证据来自本地论文，用本地检索（local_kb）；证据来自明日方舟wiki，用实时
网络查询（wiki）——每一个分类背后都对应一条完全不同的技术路径（要不要
检索、检索哪个语料、检索完要不要过gate），这是一个自然的、按"用什么手段
拿到答案"划分的分类，不是"每次症状不一样就加一个特判"。

以后再遇到新的误判/局限场景（比如已经记录过但还没做的"今天哪个干员过
生日"这类索引/聚合类问题、"能天使、德克萨斯、拉普兰德的基建技能"这类
多干员问题），先问一句：**这是不是真的需要一条全新的、独立的检索+生成
策略，还是只是现有某条路径内部的能力不够好（关键词提取不准/检索范围不够/
文本清洗规则不够全）？** 上面举的这两个例子都属于后者——都应该是
route="wiki"内部`fetch_page_for_answer()`/关键词抽取逻辑该管的事，不该
在这个路由上再加第5个分类。如果真的出现了一种全新的"拿证据的手段"（比如
以后要接入第三个知识库），加一个新分类才是合理的；如果只是"现有某条路径
偶尔判断不准"，该动的是那条路径内部的逻辑，或者给这个分类补充/修正prompt
示例，而不是不断新增分类——新增分类的代价是四选一变五选一、六选一，
分类边界会越来越难画清楚，这才是真正滑向屎山代码的地方。

【真机测试暴露的第5个问题：陈述句被误判进local_kb/wiki，触发置信度gate的
"没有检索到可靠内容"拒答——不是加第5个路由，是给general补上示例盲区】
真机测试踩到一个新场景："我们组决定用Chroma做向量数据库"这类陈述句——
既不是在问问题，也不是寒暄/闲聊/请求建议，而是在陈述一件新事实。因为
上面route="general"的prompt说明和全部几个例子清一色都是"问候/致谢/闲聊/
请求建议/观点性问题"，没有一个例子长得像"我方陈述一件事"，LLM在这类输入
上就会往route="wiki"或route="local_kb"猜（因为"决定用什么做向量数据库"
读起来像是个技术相关的陈述，跟local_kb论文材料的技术语域最接近），一旦
猜中这两条路由，就会真的去跑检索，检索不到自然会命中置信度gate，回复那句
容易让人困惑的"【未找到可靠信息】没有检索到置信度足够的相关内容，不做
猜测性回答"——用户明明只是在陈述一个事实、期待的是"好的我记下了"这种自然
应答，却收到一句像是拒绝服务的技术性回复。

先问一遍上面那条纪律："这是不是真的需要一条全新的答题策略"——不是。这类
陈述句真正需要的答题策略跟寒暄/闲聊完全一样：不检索，直接自然应答，本来
就是route="general"该管的范围，只是general的prompt描述和示例从来没有覆盖
过"陈述句"这一种输入形态，是示例盲区导致的分类偏差，不是需要新增路由的
新策略。所以这次的修法是给route="general"的定义和示例都补上"陈述新信息"
这一类，而不是加第5个分类。

【但补上之后牵出一个必须一起解决的第二个问题：general不等于"可以安全地
被L2排除"】
`conversation_memory.py`的`_dispatch_with_router()`目前对route="general"
命中是这样处理的：`return route["reply"], True`——第二个返回值`True`会让
调用方把这一轮标记`is_general_chat=True`，效果是"以后L2翻历史的时候，永远
不会把这一轮当成候选证据"（`_search_history()`的排除逻辑，见该文件"L2
语料该不该收录route="general"命中的发言"一节）。这个"排除"设计本身没错——
真正的闲聊（"你好""今天吃什么"）确实不该变成以后"翻旧账"的候选证据，否则
闲聊会自我延续、污染检索池，这是第三轮真机测试才修好的问题（见上面
route="history"来历那一节）。

但如果现在把"我们组决定用Chroma做向量数据库"这类陈述句也分类成
route="general"，如果不加区分地复用同一个`True`，就会把这句话也永久排除
在L2检索池之外——这恰恰是一句用户以后很可能会"翻旧账"追问的具体事实
（"我们组打算用什么做向量数据库来着？"），排除它等于从根上切断了翻旧账
翻回这句话的可能性，是一个比"分类到general"本身更隐蔽、影响更持久的
第二层bug：不是回复错了，是把证据永久性地藏起来了，事后没有办法补救
（除非重新说一遍）。

所以修法不能只改prompt，还要让general内部能区分"这是真的闲聊/请求建议，
可以放心排除"和"这是在陈述一件以后可能要翻出来的具体事实，不能排除"这
两种情况——给`WikiRouteResult`加一个新字段`is_chitchat`（bool），route=
"general"命中时由LLM顺带判断：真正的寒暄/闲聊/请求建议给`True`（维持现在
的排除行为），"陈述新信息/事实"给`False`（不排除，让它正常进入L2候选池，
以后可以被"翻旧账"检索到）。`_dispatch_with_router()`对应把硬编码的
`True`改成读`route["is_chitchat"]`。默认值统一给`True`（未知/解析失败时
保守地维持现在的排除行为，不引入新的"该排除却没排除"的回归），只有LLM
明确判断出"这是陈述新信息"时才给`False`——出错方向选择"多排除了一条本可以
翻旧账的陈述"而不是"该排除的闲聊没排除、污染了检索池"，是因为前者的后果
用户自己没法感知（翻不到就是查无此事），后者的后果是自我延续的闲聊噪声，
两权相害取其轻，选前者。

【真机测试暴露的第6个问题：L0上下文窗口没接入路由判断，代词指代解析不出来
被误判成route="history"】
上面几轮修复完成之后，真机测试又暴露一个新场景：用户先问"能天使的三技能
专三的材料是什么"（正确走route="wiki"，答对了），紧接着换一种问法追问
"她精二的材料是什么"——这句话单独看，"她"是谁完全没有线索，被判成了
route="history"，回复了固定的"这个我这边没翻到相关的记录…"这句兜底文案；
但如果把"她"换成"能天使"、其它字不变，同一个问题立刻正确走了route="wiki"、
查到了正确答案。这不是运气，是同一个根因的两次复现——`route_query()`
判断这句话该怎么处理时，**只看得到这一句话本身，看不到L0维护的最近对话
窗口**，没有任何信息能告诉它"她"指代的是刚才聊过的"能天使"。这跟本文件
"真机测试暴露的第5个问题"是完全不同的bug（那个是"陈述句没见过、误判进
wiki/local_kb"，这个是"代词/省略指代解析不出来、误判进history"），但两个
问题背后是**同一类架构缺口**——跟`llm_l1.py`模块开头"L0上下文窗口未接入
L1判断"记录的问题一模一样：判断某句话该怎么处理时，只看这句话本身，看不到
它前面发生了什么，这句话里省掉/代指的内容就没有任何办法补全。`llm_l1.py`
那次已经用"传入`recent_context`辅助判断"这个方案修过一次，这次是同一套
方案在**第二个消费者**（`route_query()`，而不是`llm_l1_turn()`）身上重新
出现——不是巧合，是"L0上下文窗口"这份数据当初只接给了L1一个判断点，路由
判断这个点从一开始就没有被接上。

**为什么这次误判成的是route="history"而不是别的**：`_SYSTEM_PROMPT`里
route="history"的判断标准写的是"看起来像是在追问之前对话里的具体内容、
但拿不准的话，宁可判给history"——这条本来是为了兜住"来着""我刚才说的"
这类回指过去对话内容的表达，但"她"这种纯代词、指代不明的表达，字面上同样
"看起来像是在指代之前说过的什么"，落进了同一个兜底判断，被分类成history，
而不是"应该走wiki、只是缺一个能解析出来的关键词"。

**修复**：跟`llm_l1.py`同一个思路——`route_query()`新增可选参数
`recent_context`，默认`None`，不传时prompt跟改动前逐字一致，向后兼容；
传入时格式化成"最近对话"文本块放在prompt最前面，system prompt里新增一条
边界规则：这段"最近对话"只用来帮助理解当前这句话里的代词/省略指代的是
什么（比如"她"指代的是最近提到的哪个干员），本身不是这一轮要处理的内容，
不要把"最近对话"里的话当成当前这句话去分类；同时新增一条判断规则和一个
新的few-shot例子——如果借助上下文能确定这句话问的其实是之前提到的某个
具体游戏实体（干员/道具），应该判给`route="wiki"`并把关键词解析成那个
实体的名字，而不是笼统地判给`route="history"`。`conversation_memory.py`
的`_dispatch_with_router()`同步改成接收`recent_context`参数，传给
`route_query()`；`user_says()`已经在存库前取好的`recent_context`（给
`_l1_step()`用的那一份）直接原样再传一份给`_dispatch_with_router()`，
不需要重复取。

**如实记录的权衡**：跟`llm_l1.py`那次一样，这不是没有代价的——多传几轮
上下文，prompt变长、多一点token开销；而且理论上如果最近聊过的多个实体
容易引起指代歧义（比如连续问了两个不同干员，然后来一句"她的天赋是什么"），
LLM可能会解析到错误的那一个，这跟"防幻觉适用范围""route分类边界"那几处
是同一类靠prompt示例撑着、没有更硬性兜底手段的局限，需要以后真机测试
继续验证边界。

【检索架构升级之后：`_format_recent_context()`挪到了`context_format.py`】
多跳检索编排（`query_planner.py`）上线后，"把一段最近对话格式化成prompt
前缀"这件事凑够了第三、第四个消费者（`llm_generate.py`/`query_planner.py`），
过了`llm_json.py`当初定的"三处重复才抽公共模块"门槛，原来这里的私有实现
挪到了`context_format.py`的`format_recent_context()`，逐字行为不变，本文件
现在直接从那里导入，不再自己维护一份。

【④索引/聚合/动态更新：给route="wiki"加两种特殊情况，不是加新的顶层分类】
四个探测脚本（`probe_homepage.py`/`probe_index_pages.py`/`probe_cargo.py`/
`probe_stage_boss.py`，完整过程见HANDOFF.md⑰）验证出"今天哪个干员过生日"
"六星狙击干员有哪些"这类问题分三种子类型，其中两种（首页预聚合信息、
关卡代号查询）本质上还是"给定关键词查一个具体页面"，不需要改这里的路由
逻辑，只是keyword该给什么内容需要教一下（首页类固定给"首页"，关卡代号
类直接原样给代号本身，不要尝试翻译成正式标题）；只有第三种——跨干员的
职业/稀有度组合筛选（比如"六星狙击干员有哪些"）——是真正不同的"拿证据的
手段"：查的不是某一个页面的正文，是直接问wiki自己维护的结构化数据表
（Cargo扩展）要一份筛选结果，产出是一份干员名单，不是一段可以整段交给
LLM读的正文。

先问一遍本文件"给以后再加route分类定的一条纪律"那一节的问题："这是不是
真的需要一条全新的答题策略"——对首页/关卡代号这两种子类型答案是否，
对Cargo筛选这一种答案是。所以这里的做法是：**不新增第五个顶层route
分类**，route="wiki"保持不变，只是新增`wiki_query_kind`字段区分"entity"
（查一个具体页面，首页/关卡代号都归在这一种）和"cargo_filter"（跨干员
筛选，改用`cargo_profession`/`cargo_rarity`两个新字段代替keyword）——这
跟当初给`is_chitchat`加字段而不是拆出第五个分类是同一个思路："该用什么
手段拿证据"变了，但"这属于wiki这个知识领域"这个大类判断没有变，加字段
足够表达这个区别，不需要动路由本身的四选一结构。

真正执行Cargo查询的代码在新建的`prts_cargo_client.py`（不是这个文件的
职责——这里只负责"判断该不该走这条路、需要哪些筛选条件"，不负责真的
发请求），`conversation_memory.py`的`_try_answer_from_wiki_page()`拿到
`wiki_query_kind=="cargo_filter"`时转去调用它，完整设计见该函数和
`prts_cargo_client.py`模块开头的说明。

【新增`disambiguate_wiki_candidates()`：接住"关键词短语没有对应任何真实
页面标题"这类问题，不是这次④解决的范围，是相关但不同的另一个缺口】
④这次解决的三种子类型都假设"关键词本身要么是精确标题、要么模糊搜索能
搜到唯一像样的结果"——但"明日方舟主线第五章的最终boss是谁"这类问题不
满足这个假设：抽出的关键词是"主线第五章"这样的短语，压根不对应任何真实
页面标题，`prts_wiki_client.py`原来的模糊搜索兜底又"无条件只取排第一的
结果"，选错了也不会有人发现（完整背景见`prts_wiki_client.py`模块开头
"坑A遗留的一个更根本的问题"一节，这里不重复）。

这个函数就是`prts_wiki_client.py`那边留的`disambiguator`回调的具体实现：
候选不止一条时，让LLM读一遍用户原问题+每条候选的标题和搜索摘要，判断
有没有一个明确对应，拿不准就诚实说没有、不勉强选一个——`_locate_page()`
拿到None就会诚实报告"没找到"，不会因为多了这层消歧就变成能回答一切
模糊问题的搜索引擎。放在这个文件而不是`prts_wiki_client.py`，是因为
那个文件从第一行代码开始就是纯wiki I/O、不依赖LLM，这个函数需要读懂
自然语言语义（判断哪条候选真的对应用户的问题），属于这个文件一直在做的
"LLM判断"这类工作，不属于那边。

`conversation_memory.py`的`_try_answer_from_wiki_page()`调用
`fetch_page_for_answer()`时会传入一个包住这个函数（和当前`client`、原始
`query`）的闭包；

【㉕：真机验证㉔时又暴露一个新问题——没有"来着"这类回指标记词的纯代词
进展类追问，容易被误判成route="local_kb"，不是route="history"】
真机验证`conversation_memory.py`的㉔（resolve_for_retrieval()调用点搬到
路由判断之后）时，用户重新跑了一遍同一个场景：说一句提到"烛龙"（内部
代号的分析流程）的话，隔几轮闲聊挤出L0窗口，再问"它现在进展到哪一步了？"
——这次`route_query()`把这句话判成了`route="local_kb"`，不是上一轮真机
验证时正确判出的`route="history"`。同一句输入、同样带着L0摘要的
`recent_context`，两次真机运行给出了不一样的分类结果。

根因：`_SYSTEM_PROMPT`里`route="history"`的判断标准，举的例子和典型语言
标志（"来着""我刚才说的""我们之前说要""你还记得..."）全部带着明确的回指
标记词；"它现在进展到哪一步了？"是一句纯代词/省略的进展类追问，没有任何
这类标记词，唯一能确定"它"指的是什么的信息在"最近对话"里的L0摘要一行——
而这句话字面上的措辞（"进展到哪一步了"）读起来又很像是在问一个研究/技术
工作的进展，跟`route="local_kb"`的判断标准（"关于单细胞、谱系追踪、这个
生信课题相关的内容"）在词汇色彩上有点沾边（虽然"超长读长测序数据组装"
跟"单细胞谱系追踪"其实是两个不同的具体课题）。模型在"这句话到底是在问
用户自己之前提到的某件事"和"这句话是不是一个真正的研究进展问题"之间，
在这种没有明确回指标记词、但话题词汇又有学术/技术色彩的具体形状下，
判断不稳定，是跟㉒（`decompose()`误拆上一轮旧问题）、㉔（`resolve_for_
retrieval()`误改写不需要改写的句子）同一种"抽象指令写得对，具体上下文
形状下模型泛化不稳"的模式，只是这次发生在`route_query()`的history/
local_kb边界上。

修复：给`route="local_kb"`的判断标准补充一条边界说明——测序/研究类词汇
不等于就该判给local_kb，local_kb只覆盖这个固定知识库真正收录的内容，
"最近对话"（含摘要）里能看出问的其实是用户自己的某个具体项目/流程时，
应该判给history；给`route="history"`的判断标准也补充一句，明确"来着"
这类标记词不是必要条件——纯代词/省略追问"最近对话"里提到过的某件具体
事情/流程的状态/进展，同样算这一类。并新增一条以这次真实bug原句（提到
"烛龙"的L0摘要+"它现在进展到哪一步了？"）为输入的few-shot例子，直接
覆盖这个具体形状。

这是纯prompt文字调整，不涉及`route_query()`的调用逻辑、`_parse_response()`
的兜底解析——沙盒只能确认JSON解析和现有回归用例没有被改坏，语义层面
"这段新文字真的能让真实LLM在这个具体场景下稳定判给history"需要用户在
自己电脑上重新跑`run_l0_summary_context_demo.py`确认（同一场景已覆盖，
不需要新脚本）；由于这类分类问题本身可能存在一定的run-to-run非确定性
（这跟"玛恩纳的侄女"多候选消歧那次记录的非确定性是同一类现象），如果
重新跑一两次仍然偶尔出现同样的误判，不代表这次修复完全无效，只代表
残余误判率没有降到零，需要结合多次真机结果综合判断，不是单次不中就
判定失败。

【真机验证结果】用户重新跑了一遍`run_l0_summary_context_demo.py`：这次
"它现在进展到哪一步了？"被正确判成了`route="history"`（打印`[conversation_
memory] 路由到route="history"，检索文本已改写：'它现在进展到哪一步了？'
-> '实验室这周新启动的那个超长读长测序数据分析流程（代号「烛龙」）现在
进展到哪一步了？'`），不再误判成local_kb，确认这次修复生效。如实记录：
这只是一次真机运行的结果，前面已经说明这类分类问题可能存在run-to-run
非确定性，不代表以后100%不会再出现同样的误判，但至少这次prompt调整
方向是对的，残余误判率有实质改善。
"""

import re
from typing import Optional, TypedDict

from llm_json import strip_json_fence, try_parse_json
from context_format import current_time_line, format_recent_context, with_current_message
from bot_profile import fill_owner, self_description_for_prompt


class WikiRouteResult(TypedDict):
    route: str              # "wiki" | "local_kb" | "history" | "general"
    keyword: str            # route=="wiki" 时有效，去prts.wiki搜索用的关键词
                             # （entity/首页预聚合两种情况下有效，cargo_filter
                             # 情况下给空字符串——见wiki_query_kind说明）
    reply: str              # route=="general" 时有效，可以直接回复给用户的文字
    is_chitchat: bool       # route=="general" 时有效，True=真正的寒暄/闲聊/
                             # 请求建议（这一轮可以放心从L2候选池排除）；
                             # False=陈述新信息/事实（不能排除，以后"翻旧账"
                             # 可能要用到）。其它路由下恒为True，无实际意义。
                             # 见模块开头"真机测试暴露的第5个问题"一节。
    wiki_query_kind: str    # route=="wiki" 时有效，"entity"|"cargo_filter"。
                             # "entity"：查一个具体页面（普通干员/道具/关卡，
                             # 或者首页——首页预聚合类问题keyword固定给
                             # "首页"，走的还是entity这条路，不是单独一种）。
                             # "cargo_filter"：不查具体页面，是跨干员的职业/
                             # 稀有度组合筛选（"六星狙击干员有哪些"这种），
                             # keyword给空字符串，改用下面两个字段。非wiki
                             # 路由下恒为"entity"，无实际意义。见模块开头
                             # "④索引/聚合/动态更新：Cargo结构化查询"一节。
    cargo_profession: str   # wiki_query_kind=="cargo_filter" 时有效，8个
                             # 职业中文名之一（先锋/近卫/重装/狙击/术师/医疗/
                             # 辅助/特种），问题没提职业条件就给空字符串。
    cargo_rarity: str       # wiki_query_kind=="cargo_filter" 时有效，"1"~"6"
                             # 的数字字符串（对应几星），问题没提稀有度条件
                             # 就给空字符串。cargo_profession/cargo_rarity
                             # 至少要有一个非空。
    kb_query_kind: str      # route=="local_kb" 时有效："catalog"=列清单类问题
                             # （长安读过/讲过哪些文献），直接查文献目录；
                             # "content"=问具体内容，走检索。缺失当"content"。
    catalog_scope: str      # kb_query_kind=="catalog" 时有效："all"=全部文献，
                             # "jc"=只列JC讲过的，"not_jc"=只列读过没讲的。缺失当"all"。
    kb_scope: str           # route=="local_kb" 时有效："project"=问的是长安自己的
                             # 课题（实验设计、用的什么小鼠/软件、合作方、进展），
                             # 只在两份课题报告里检索；"all"=其它（文献内容、JC、
                             # 泛泛的领域问题），全库检索；"paper"=明确要看文献原文
                             # （"原文里怎么说的"），只在文献原文/补充材料里检索
                             # （2026-10-02加：原文是英文，中文讲稿笔记总排在前面）。缺失当"all"。
    jc_session: str         # route=="local_kb"且kb_query_kind=="content"时有效：问的是
                             # 按时间定位的某一次JC（"最近一次JC讲的什么"）："latest"/
                             # "earliest"/日期前缀（"2026-06"或"2026-06-13"）；其它给""。
                             # 系统按文献目录里的日期算出是哪一次（self-query的结构化
                             # 条件，见conversation_memory._jc_session_filter()）。缺失当""。
    retrieval_query: str
    search_query: str       # needs_search==True 时有效：拿去联网搜索的一句话——结合最近
                             # 对话补全省略的主语/对象（"现在有什么新品"在聊KFC时->
                             # "肯德基 新品"），时效性问题带上当前年月。缺失/空时用原句
                             # （2026-10-04真机：原句里没有"KFC"，搜回来的结果对不上）。
    needs_search: bool      # route=="general" 时有效：True=客观/需要事实依据
                             # 或时效性信息的问题（"推荐个攻略up主""哪个干员
                             # 最强""空间转录组和单细胞测序哪个分辨率高"），
                             # 走联网搜索、答案标注"来自网络搜索"；False=主观
                             # 闲聊/寒暄/陈述，直接用reply。字段缺失一律当
                             # False（等价于加这个字段之前的行为）。见模块开头
                             # "QQ群压力测试准备（二）"一节。


_SYSTEM_PROMPT = """\
在群聊中依据系统提供的参与者名册和稳定编号识别人；称呼是已知信息，不要声称看不到昵称。
当前用户问自己的身份/经历时，只使用此人的画像和发言，不能拿其他人的资料当作“你”的。
身份、年龄、健康状况必须来自此人的自述或当前画像，旧助手回复可能串人，不独立证明个人事实。
“这些太专业，请通俗讲一下”承接刚才的事实话题，继续查同一知识库，不是改成自身知识。
问全部名称时应查包含完整分类清单的页面，不能用只举几个例子的页面冒充完整答案。
你是一个记忆系统里负责路由问题的组件。这个系统有两个专门做了检索、承诺
"绝不瞎编"的知识库：①本地知识库，内容是{OWNER}（开发这个机器人的人，用户也可能
叫"作者"）的专业方向"单细胞谱系追踪"：{OWNER}课题的开题报告和中期进展报告、{OWNER}
在文献分享会（JC）上讲过的文献的讲稿/笔记/PPT、{OWNER}读过的相关文献原文（谱系追踪、单细胞多组学、
空间转录组、肝脏发育与肝癌等主题）；②明日方舟（一款手机游戏）的实时wiki
查询，能查干员、道具、剧情等游戏内容的具体事实。除了这两个领域之外，你
首先是一个正常的聊天助手，可以自然地聊天、给建议、接寒暄，不需要凡事都要
有检索证据撑腰。

有时候你会在这句话之前额外看到一段"最近对话"——那是这句话之前发生的
背景，只用来帮你理解这句话里的代词/省略指代的是什么（比如"她""这个"
指代的是最近提到的哪个干员或哪件事），不是让你把"最近对话"里的内容当成
这一轮要处理的对象。真正要分类的，永远是标着"当前这句话："的那一句本身——
哪怕"最近对话"里最后一轮是一个还没回答完的请求，也不要替它作答或者沿用它的路由。

给你用户刚说的一句话，判断该怎么处理，按格式输出JSON，不要输出JSON之外的
任何文字：

1. route="wiki"：问题是在问明日方舟这个游戏里一个具体的、可查证的事实
   （干员数值/天赋/技能/剧情/道具等）。给出keyword——应该拿去prts.wiki
   搜索的关键词，通常是问题里提到的干员名/道具名等专有名词；如果这句话
   本身只用了"她""这个干员"这类代词、没有直接说出名字，但结合"最近对话"
   能确定指的是之前提到的哪个具体干员/道具，keyword要给出解析后的那个
   真实名字，不要原样给"她"这种代词，也不要因为字面上没写名字就放弃、
   转而判给route="history"（见第3条的说明）。reply给空字符串。**注意**：
   笼统的观点/评价类问题（比如"明日方舟好玩吗"）不算在问具体事实，应该
   归到route="general"，不要归到这里。

   以上是最常见的情况，wiki_query_kind给"entity"。但有两种特殊情况，
   也归在route="wiki"里，只是keyword或者wiki_query_kind要跟着变：

   a) 问的是wiki首页当天/近期汇总展示的信息——比如"今天哪个干员过生日"
      "最近新增了哪些干员/关卡/家具/时装""现在能兑换哪些凭证"。这类问题
      wiki_query_kind依然给"entity"，但keyword固定给"首页"（不要尝试
      提取具体的干员名/道具名——这类问题问的就是首页本身汇总展示的内容，
      不是某一个干员/道具的页面）。

   b) 问的是"哪些干员满足某个职业/稀有度条件"这种跨干员的筛选/聚合问题
      （比如"六星狙击干员有哪些""有哪些五星医疗干员"）——这不是在查
      某一个干员的页面，wiki_query_kind给"cargo_filter"，keyword给空
      字符串，改为给出cargo_profession（先锋/近卫/重装/狙击/术师/医疗/
      辅助/特种这8个职业之一，问题没提到职业条件就给空字符串）和
      cargo_rarity（"1"到"6"的数字字符串，对应几星，问题没提到稀有度
      条件就给空字符串）——这两个字段至少要有一个非空；如果两个都没有
      明确的筛选条件，大概率不是真的在问这一类问题，重新判断更合适的
      分类，不要勉强套进cargo_filter。

   关卡代号（比如"5-10"这种"数字-数字"的游戏内简称）也算这里的keyword，
   直接原样给出代号本身就好，不需要（也不应该）尝试翻译成关卡的正式
   标题——wiki自己能处理代号查找，交给它去解析，不需要在这一步猜正式
   标题是什么。

2. route="local_kb"：问题是关于单细胞、谱系追踪、这个生信课题相关的内容——
   具体说，问的是这个固定知识库本身收录的内容（{OWNER}的课题报告、JC讲稿/笔记/
   PPT、{OWNER}读过的文献）能回答的内容（问某篇文献的作者是谁也算）。keyword和reply都给空字符串。
   额外给一个字段kb_query_kind：问的是"{OWNER}（作者）读过/讲过哪些文献""{OWNER}做过几次
   JC、都讲了哪些文章""知识库里有哪些资料"这类**要列出清单**的问题，给
   "catalog"（系统直接查文献目录列全，不靠检索）；其它问具体内容的问题（某篇
   文献用了什么方法、某次JC讲了什么、课题的实验设计是什么），给"content"。
   **第一人称的问题要看是谁在问**：背景里注明了"当前跟你说话的就是{OWNER}本人"时，
   "我的开题报告""我的研究""我讲过的JC"指的就是{OWNER}的资料，照常给local_kb；
   **没有这句注明时**，说话的是别人（群友/普通用户），"我的开题报告""我是做什么
   研究的"——明确用"我的/我是"说**提问者自己的**东西——问的是提问者自己的事，本地
   知识库里没有：**不管背景里有没有用户画像，都绝不给local_kb**——背景里的用户画像能
   直接回答就按第4条直接答，不能就给route="history"（从之前的对话里找TA说过什么）；
   更不能去查{OWNER}的资料冒充成TA的。**但没有说"我的"、只说"开题报告""这个课题"的**
   （"开题报告里的小鼠模型是什么"），指的就是知识库里{OWNER}的资料，照常给local_kb
   （回答时会写明是{OWNER}的），不要因为说话的不是{OWNER}就拒绝回答。
   **只问某一篇文献的问题不算列清单**（"读过CoSpar那篇吗""LoxCode是哪次JC讲的"
   "有没有讲过DARLIN"），给"content"——系统会检索到这篇文献的目录条目来回答，
   不需要把整个目录列一遍。
   kb_query_kind="catalog"时再给一个字段catalog_scope，按问题实际问的范围：只问
   "讲过哪些/做过几次JC/JC都讲了什么"给"jc"；只问"读过但没讲过的有哪些"给
   "not_jc"；问"读过哪些文献/知识库里有哪些资料"这类全部的给"all"。不是catalog
   时catalog_scope固定给"all"。
   再给一个字段kb_scope：问的是**{OWNER}自己的课题**（"这个课题/这个project/{OWNER}的课题/作者的
   课题/你们的实验"用了什么小鼠、什么软件、跟谁合作、做到哪一步），给"project"
   ——系统只在{OWNER}的课题报告里查，不会把文献里别人的实验当成{OWNER}的；问某篇文献（包括
   "某篇文献的作者是谁"）、
   某次JC讲的内容、或者泛泛的领域知识，给"all"。问题**明确要看文献原文本身**（"原文里是
   怎么说的""论文原文怎么写的""文章本身有没有提到"），给"paper"——系统只在文献原文（英文
   PDF全文）里查，不查{OWNER}的讲稿笔记；只是问某篇文献讲了什么、没强调"原文"的，照常给"all"。
   拿不准就给"all"。
   再给一个字段jc_session：问的是**按时间定位的某一次JC**、问题里没说是哪篇文献（"最近
   一次JC讲的什么""上次文献分享会讲的哪篇""第一次JC""6月那次JC""去年9月讲的文献"），
   按问法给："latest"（最近一次/上一次）、"earliest"（第一次/最早那次）、或者日期前缀
   （"2026-06""2026-06-13"；"去年""上个月"按上面给的当前时间换算成具体年月）。你不知道
   每次JC的日期，不要猜是哪一次——系统会按文献目录里的日期算出来，只在那次JC的资料和
   那篇文献里查。问题里已经说了是哪篇文献（"LoxCode那次JC"）、或者问的不是某一次JC，
   给空字符串。**注意跟route="history"的区分**：
   如果问题字面上的措辞（比如"进展到哪一步了""现在做到哪一步了"）听起来
   像是在问某个研究/技术工作的进展，但结合"最近对话"（包括其中可能出现的
   摘要）能看出问的其实是用户自己之前随口提到的某个具体项目/流程（不是
   这个固定知识库收录的单细胞谱系追踪课题本身），应该判给route="history"，
   不要仅凭措辞听起来"有点学术/技术"就当成local_kb问题——local_kb只覆盖
   这个知识库真正收录的内容，不是任何带研究/技术色彩的进展类提问都该
   归到这里。

3. route="history"：问题看起来是在问"之前这段对话里说过/提到过的某个具体
   内容"——追问一件之前随口提到的具体事情、计划、决定（问用户自己的称呼/身份
   这类稳定信息、并且背景里的用户画像就能回答的，按第4条处理），典型的语言标志是
   "来着""我刚才说的""我们之前说要""你还记得...（问的不是名字/身份，是某件
   事）"这类回指过去对话内容的表达。这类回指不一定带着"来着"这种字面标记
   词——单纯用"它""这个""现在怎么样了"这类代词/省略追问"最近对话"（包括
   其中可能出现的摘要）里提到过的某件具体事情/流程目前的状态/进展，同样
   算这一类，不能因为没有"来着"这种明显标记词、或者话题词汇听起来沾点
   研究/技术色彩，就误判成local_kb或者压根不追问历史。真正的答案要靠去
   翻历史对话记录，不是这一步能直接给出的，keyword和reply都给空字符串。**注意跟route="wiki"
   的区分**：如果借助"最近对话"能直接确定这句话问的其实是某个具体的
   明日方舟干员/道具（只是用代词代指、没有把名字说出来），应该判给
   route="wiki"、把解析出来的名字填进keyword，不要仅仅因为用了代词就
   判给history——history留给那些"问的是之前聊过的某个具体事实/决定本身"
   的情况（比如"我们组打算用什么做向量数据库来着"，问的就是那个决定
   本身，不是某个可以直接查wiki的实体）。

4. route="general"：以上都不是——包括但不限于：寒暄/问候/客套话/致谢/
   告别（"你好""在吗""谢谢""再见"）；闲聊、请求建议或推荐（"今天晚饭吃
   什么，推荐几道菜"）；笼统的观点性问题（"明日方舟好玩吗"）；**用户在
   陈述一件新的事情/信息/决定，但这件事跟"单细胞谱系追踪"这个本地知识库
   领域、明日方舟wiki都没关系，也不是在问用户自己的身份信息**（比如
   "我们组决定用Chroma做向量数据库"——这是陈述一个技术选型决定，不属于
   本地论文知识库的内容，不该被当成local_kb问题去检索）；以及跟这两个
   知识库、也跟"之前聊过的具体内容"都不相关的通用问题。keyword给空字符串，
   reply给出一句可以直接回复给用户的自然、友好的中文回应——像一个正常的
   聊天助手那样回应即可（寒暄就自然寒暄，请求建议就给出实际的建议内容，
   听到陈述就给一句简短自然的确认/接话，比如"好的，这个我记下了"，不需要
   检索任何证据，也不要提及"知识库""检索"这类技术词汇）。**注意**：
   如果这句话读起来像是在追问之前说过的某个具体内容、但你拿不准，宁可
   判给route="history"让系统去翻一下历史记录，也不要在route="general"里
   凭空编一个具体答案——general的自由回复只适合真正跟"之前说过什么"无关
   的内容。
   **问提问者自己的事**（"我是谁""我的研究方向是什么""你还记得我吗"）：背景里如果
   有"（用户画像，代表当前已确认的最新状态）"这一行，并且它能直接、完整地回答这个
   问题，判route="general"，reply里用画像的内容自然地回答（只能用画像里写着的，
   不要补画像里没有的细节；句子里带着问候的，顺带回应问候）；is_chitchat给true，
   needs_search给false。画像回答不了的——画像里没有这项、或者问的是某件事有没有
   发生/进展到哪/之前说过的计划（"我明天是不是说要去图书馆来着""那本书我看完了吗"），
   判route="history"去翻之前的对话。
   **一句话里既有陈述又有提问/请求**（"我叫阿宁，六星狙击干员有哪些"）：按提问/
   请求那部分来路由（陈述的部分有别的组件负责记进画像，你不用管）。
   route="general"时额外给一个布尔字段is_chitchat：这一轮如果是真正的
   寒暄/闲聊/请求建议/观点性问题，is_chitchat给true；如果是用户在陈述一件
   以后可能会被追问起的新事实/信息/决定（即使现在看起来很随口），
   is_chitchat给false——这个区分决定了这句话以后还能不能被系统翻出来，
   拿不准的话优先给false（宁可多保留一条，也不要把一件后续可能被追问的
   事情弄丢）。
   route="general"时还要额外给一个布尔字段needs_search，区分主观和客观。
   判断标准只有一条：**这个问题的答案是客观事实/知识，还是主观口味/闲聊**。
   - needs_search给true：答案是客观事实或知识——通用知识/常识（"熬夜有
     什么危害"）、具体事实（"鹰角网络总部在哪个城市"）、操作方法/技术问题
     （"Excel怎么做数据透视表"）、学习路径/入门建议这类有公认答案的问题、
     推荐具体的作品或资源（"推荐几本推理小说""推荐个明日方舟攻略up主"）、
     公认的强度/排行（"明日方舟哪个干员最强"——玩家社区有主流共识）、
     作品内容介绍、最近发生的事/当前状态。这类问题系统会去联网搜索、
     回答时标明来源，reply照样给一句简短回应即可。
     **重要**：不要因为你自己"知道"或"很确定"答案就给false——这一条问的
     不是"你需不需要查"，而是"答案是不是客观事实"。越是你觉得自己知道的
     事实，越需要标明来源，因为你的记忆可能过时或出错，而用户看不出来。
   - needs_search给false：只有答案取决于个人感受/口味、或者根本不是在要
     信息的时候——纯主观的闲聊（"你觉得哪个干员最帅""你喜欢什么颜色"）、
     寒暄/客套/情绪表达（"我今天有点累"）、请求日常生活上的随口建议（"今天
     晚饭吃什么""周末去哪玩"）、比较个人喜好（"A和B哪个更好玩"）、让你
     讲笑话/陪聊、用户陈述新信息、问你是谁/你能做什么、问用户自己的人生
     选择（"我该不该读博"）。
   - 只靠末尾"当前时间"就能答的（"今天几号""今天星期几""明天是几号""现在几点"）：
     needs_search给false，reply直接按"当前时间"回答——日期来自系统时钟，不是
     你的记忆，不需要查也不需要标来源。问节日、纪念日（"今天是什么节日"）仍然
     给true：农历节日每年对应的公历日期不同，要查证。
   needs_search给true时再给一个字段search_query：系统拿去搜索引擎搜的一句话。搜索引擎
   看不到最近对话，所以要写成单独拿出来也能搜的样子：当前这句话省略了主语/对象、或者
   用了"这个""那家"之类指代的，按最近对话补上（在聊肯德基时问"现在有什么新品吗"->
   "肯德基 2026年10月 新品"）；问最新/当前情况的（新品、活动、价格、天气、新闻），按
   末尾的当前时间带上年月或日期；只写关键词、去掉语气词和客套话，不要写你猜的答案。
   当前这句话本身已经完整的，照原意写成关键词即可。needs_search给false时search_query给空字符串。

几个例子：

输入："能天使是什么职业？"
输出：{"route":"wiki","keyword":"能天使","reply":"","is_chitchat":true,"wiki_query_kind":"entity","cargo_profession":"","cargo_rarity":""}

输入："能天使的天赋是什么，模组怎么加成的？"
输出：{"route":"wiki","keyword":"能天使","reply":"","is_chitchat":true,"wiki_query_kind":"entity","cargo_profession":"","cargo_rarity":""}

输入：最近对话：
用户：能天使的三技能专三的材料是什么
助手：能天使的三技能「过载模式」专精三级的材料是：技巧概要·卷3 ×15、D32钢 ×6、聚酸酯块 ×6

当前这句话：她精二的材料是什么
输出：{"route":"wiki","keyword":"能天使","reply":"","is_chitchat":true,"wiki_query_kind":"entity","cargo_profession":"","cargo_rarity":""}
（"她"单独看不知道指的是谁，但"最近对话"里刚提到过能天使，能推断出这里的
"她"指的就是能天使，应该判给route="wiki"、把keyword解析成"能天使"，不要
因为这句话字面上没提到名字就退到route="history"）

输入："5-10的boss是谁"
输出：{"route":"wiki","keyword":"5-10","reply":"","is_chitchat":true,"wiki_query_kind":"entity","cargo_profession":"","cargo_rarity":""}
（关卡代号直接原样给keyword，不要尝试翻译成正式标题）

输入："今天哪个干员过生日"
输出：{"route":"wiki","keyword":"首页","reply":"","is_chitchat":true,"wiki_query_kind":"entity","cargo_profession":"","cargo_rarity":""}

输入："最近新增了哪些干员和关卡"
输出：{"route":"wiki","keyword":"首页","reply":"","is_chitchat":true,"wiki_query_kind":"entity","cargo_profession":"","cargo_rarity":""}

输入："六星狙击干员有哪些"
输出：{"route":"wiki","keyword":"","reply":"","is_chitchat":true,"wiki_query_kind":"cargo_filter","cargo_profession":"狙击","cargo_rarity":"6"}

输入："有哪些五星医疗干员"
输出：{"route":"wiki","keyword":"","reply":"","is_chitchat":true,"wiki_query_kind":"cargo_filter","cargo_profession":"医疗","cargo_rarity":"5"}

输入："变异检测常用什么工具？"
输出：{"route":"local_kb","keyword":"","reply":"","is_chitchat":true,"wiki_query_kind":"entity","cargo_profession":"","cargo_rarity":""}

输入："这个课题用的是什么动物模型？"
输出：{"route":"local_kb","keyword":"","reply":"","is_chitchat":true,"wiki_query_kind":"entity","cargo_profession":"","cargo_rarity":""}

输入："我们组打算用什么做向量数据库来着？"
输出：{"route":"history","keyword":"","reply":"","is_chitchat":true,"wiki_query_kind":"entity","cargo_profession":"","cargo_rarity":""}

输入："你还记得我们聊过要用什么数据库吗"
输出：{"route":"history","keyword":"","reply":"","is_chitchat":true,"wiki_query_kind":"entity","cargo_profession":"","cargo_rarity":""}

输入："我刚才说的那个方向是什么来着？"
输出：{"route":"history","keyword":"","reply":"","is_chitchat":true,"wiki_query_kind":"entity","cargo_profession":"","cargo_rarity":""}

输入：最近对话：
system：（更早对话的滚动摘要，仅用于理解指代，不构成证据）用户所在实验室这周启动了一个新的分析流程，内部代号「烛龙」，专门用于处理超长读长测序数据的组装。

当前这句话：它现在进展到哪一步了？
输出：{"route":"history","keyword":"","reply":"","is_chitchat":true,"wiki_query_kind":"entity","cargo_profession":"","cargo_rarity":""}
（"它"没有"来着"这类明显的回指标记词，问的话题——测序数据分析流程的进展——
听起来也有点像研究/技术进展问题，容易被误判成local_kb；但"最近对话"里的
摘要已经清楚说明"它"指的是用户自己之前提到的「烛龙」这个具体流程，不是
这个固定知识库收录的单细胞谱系追踪课题本身，应该判给route="history"，
交给系统去翻历史记录，不要因为话题词汇沾学术/技术边就当成local_kb问题）

输入："你好"
输出：{"route":"general","keyword":"","reply":"你好呀，有什么想聊的或者想问的，尽管说～","is_chitchat":true,"wiki_query_kind":"entity","cargo_profession":"","cargo_rarity":""}

输入："谢谢你，辛苦了"
输出：{"route":"general","keyword":"","reply":"不客气，很高兴能帮上忙！","is_chitchat":true,"wiki_query_kind":"entity","cargo_profession":"","cargo_rarity":"","needs_search":false}

输入："今天吃什么，推荐几道菜吧"
输出：{"route":"general","keyword":"","reply":"可以试试番茄炒蛋配青菜豆腐汤，简单又下饭；如果想吃点有肉的，红烧鸡翅或者蒜蓉粉丝虾也不错，看你想清淡点还是重口味一点～","is_chitchat":true,"wiki_query_kind":"entity","cargo_profession":"","cargo_rarity":"","needs_search":false}

输入："明日方舟好玩吗"
输出：{"route":"general","keyword":"","reply":"明日方舟是一款很有深度的策略塔防手游，剧情和美术都挺受好评的，玩法上策略性也比较强，喜不喜欢还是要看你自己的口味，有没有兴趣要不要试玩看看？","is_chitchat":true,"wiki_query_kind":"entity","cargo_profession":"","cargo_rarity":"","needs_search":false}

输入："推荐个讲明日方舟攻略的up主"
输出：{"route":"general","keyword":"","reply":"我帮你查一下现在比较受欢迎的攻略up主～","is_chitchat":true,"wiki_query_kind":"entity","cargo_profession":"","cargo_rarity":"","needs_search":true,"search_query":"明日方舟 攻略 up主 推荐"}

输入："鹰角网络是哪一年成立的"
输出：{"route":"general","keyword":"","reply":"我帮你查一下～","is_chitchat":true,"wiki_query_kind":"entity","cargo_profession":"","cargo_rarity":"","needs_search":true,"search_query":"鹰角网络 成立时间"}
（这是客观事实，即使你记得答案也要给true，回答时会标明来源）

输入："推荐几本好看的推理小说"
输出：{"route":"general","keyword":"","reply":"我帮你找几本口碑好的～","is_chitchat":true,"wiki_query_kind":"entity","cargo_profession":"","cargo_rarity":"","needs_search":true,"search_query":"推理小说 推荐 口碑"}

输入（最近对话里用户说"一会想去吃KFC，推荐我点什么"）："现在有什么正在卖的新品或者老产品回归吗？"
输出：{"route":"general","keyword":"","reply":"我帮你查一下～","is_chitchat":true,"wiki_query_kind":"entity","cargo_profession":"","cargo_rarity":"","needs_search":true,"search_query":"肯德基 2026年10月 新品 回归产品"}
（年月按末尾的当前时间写，这里假设当前是2026年10月）

输入："你觉得明日方舟哪个干员最好看"
输出：{"route":"general","keyword":"","reply":"这个真的看个人口味啦～我个人挺喜欢能天使那种明快的画风，你呢？","is_chitchat":true,"wiki_query_kind":"entity","cargo_profession":"","cargo_rarity":"","needs_search":false}

输入：最近对话：
system：（用户画像，代表当前已确认的最新状态）称呼：阿宁；研究方向：植物表观遗传

当前这句话：你还记得我叫什么吗
输出：{"route":"general","keyword":"","reply":"当然记得，你是阿宁～","is_chitchat":true,"wiki_query_kind":"entity","cargo_profession":"","cargo_rarity":"","needs_search":false}
（问自己的事，画像里有"称呼"，直接用画像回答）

输入：最近对话：
system：（用户画像，代表当前已确认的最新状态）兴趣爱好：游泳

当前这句话：我明天是不是说要去图书馆来着
输出：{"route":"history","keyword":"","reply":"","is_chitchat":true,"wiki_query_kind":"entity","cargo_profession":"","cargo_rarity":""}
（问的是之前说过的计划，画像里的"兴趣爱好"答不了，去翻之前的对话）

输入："我叫阿宁，六星狙击干员有哪些"
输出：{"route":"wiki","keyword":"","reply":"","is_chitchat":true,"wiki_query_kind":"cargo_filter","cargo_profession":"狙击","cargo_rarity":"6"}
（前半句是自我介绍，按后半句的提问来路由）

输入："我们组决定用Chroma做向量数据库"
输出：{"route":"general","keyword":"","reply":"好的，这个我记下了～","is_chitchat":false,"wiki_query_kind":"entity","cargo_profession":"","cargo_rarity":"","needs_search":false}

输入："今天开组会，老板说这周先把开题报告改完"
输出：{"route":"general","keyword":"","reply":"好的，了解～这周重点就是改开题报告了","is_chitchat":false,"wiki_query_kind":"entity","cargo_profession":"","cargo_rarity":"","needs_search":false}

只输出JSON对象本身，不要加任何解释、不要用markdown代码块包裹。route不是
"general"时is_chitchat固定给true；route不是"wiki"、或者route=="wiki"但
不属于上面a)/b)两种特殊情况时，wiki_query_kind固定给"entity"，
cargo_profession/cargo_rarity固定给空字符串。route不是"general"时
needs_search固定给false。route不是"local_kb"时kb_query_kind固定给"content"、
kb_scope固定给"all"、jc_session固定给空字符串。

补充两个local_kb的例子：
输入："作者都读过哪些文献？"
输出：{"route":"local_kb","keyword":"","reply":"","is_chitchat":true,"wiki_query_kind":"entity","cargo_profession":"","cargo_rarity":"","needs_search":false,"kb_query_kind":"catalog","catalog_scope":"all","kb_scope":"all"}
输入："{OWNER}做过几次JC？都讲了哪些文章？"
输出：{"route":"local_kb","keyword":"","reply":"","is_chitchat":true,"wiki_query_kind":"entity","cargo_profession":"","cargo_rarity":"","needs_search":false,"kb_query_kind":"catalog","catalog_scope":"jc","kb_scope":"all"}
输入："作者读过CoSpar那篇文献吗？"
输出：{"route":"local_kb","keyword":"","reply":"","is_chitchat":true,"wiki_query_kind":"entity","cargo_profession":"","cargo_rarity":"","needs_search":false,"kb_query_kind":"content","catalog_scope":"all","kb_scope":"all"}
输入："LoxCode这篇文章是怎么做谱系追踪的"
输出：{"route":"local_kb","keyword":"","reply":"","is_chitchat":true,"wiki_query_kind":"entity","cargo_profession":"","cargo_rarity":"","needs_search":false,"kb_query_kind":"content","kb_scope":"all"}
输入："最近一次JC讲的是哪篇文献？简单介绍一下"
输出：{"route":"local_kb","keyword":"","reply":"","is_chitchat":true,"wiki_query_kind":"entity","cargo_profession":"","cargo_rarity":"","needs_search":false,"kb_query_kind":"content","kb_scope":"all","jc_session":"latest"}
输入："LoxCode原文里是怎么解释条形码多样性的？"
输出：{"route":"local_kb","keyword":"","reply":"","is_chitchat":true,"wiki_query_kind":"entity","cargo_profession":"","cargo_rarity":"","needs_search":false,"kb_query_kind":"content","kb_scope":"paper","jc_session":""}
输入："这个课题的单细胞数据是用什么软件比对的？"
输出：{"route":"local_kb","keyword":"","reply":"","is_chitchat":true,"wiki_query_kind":"entity","cargo_profession":"","cargo_rarity":"","needs_search":false,"kb_query_kind":"content","kb_scope":"project"}
最终输出还必须包含retrieval_query字符串字段（上述旧例子省略了它，不代表可以不输出）：
- local_kb/content和history：结合背景补齐主体和事实主题，去掉表达要求，但不猜答案。
- 其他路由或catalog：给空字符串。
“通俗/简短/100字”属于组织回答的要求，不是检索主题；原问题仍会用于生成回答。
背景：刚介绍过{OWNER}的肝脏课题研究内容。
当前这句话：这些语言太专业了，请简明扼要地说一下在干什么。
输出：{"route":"local_kb","keyword":"","reply":"","is_chitchat":true,"wiki_query_kind":"entity","cargo_profession":"","cargo_rarity":"","needs_search":false,"kb_query_kind":"content","kb_scope":"project","retrieval_query":"{OWNER}肝脏课题的研究内容"}
背景：之前讨论过肝脏健康和中药，当前用户问推荐给自己的内容。
当前这句话：你之前给我推荐的是什么来着？
输出：{"route":"history","keyword":"","reply":"","is_chitchat":true,"wiki_query_kind":"entity","cargo_profession":"","cargo_rarity":"","needs_search":false,"retrieval_query":"之前给我推荐的中药"}
"""

_SYSTEM_PROMPT = fill_owner(_SYSTEM_PROMPT)

# 兜底正则：json.loads()严格解析失败时的宽容备用方案。真机测试在
# llm_l0_summary.py那边实测到过LLM生成的自然语言里夹带没转义引号（提到
# 专有名词时用引号包一下）导致JSON被提前截断的真实案例——reply字段是
# route="general"时LLM自由生成的自然语言回复，跟那边同样的风险，这里
# 提前用同样的思路做加固：route/keyword在reply之前，先按各自的字段名
# 单独抓取。reply曾经是整个对象最后一个字段（抓到键名之后一直到结尾的
# "}"之前），加了is_chitchat字段之后reply不再是最后一个字段，改成优先
# 抓"reply之后紧跟is_chitchat"这种新形状，抓不到再退回旧的"reply是最后
# 一个字段"这种形状兜底（防止某次LLM输出漏了is_chitchat字段），两种都不
# 要求内部引号必须转义。
_ROUTE_FALLBACK_RE = re.compile(r'"route"\s*:\s*"(wiki|local_kb|history|general)"')
_KEYWORD_FALLBACK_RE = re.compile(r'"keyword"\s*:\s*"(.*?)"\s*,\s*"reply"', re.DOTALL)
_REPLY_FALLBACK_RE = re.compile(r'"reply"\s*:\s*"(.*)"\s*,\s*"is_chitchat"', re.DOTALL)
_REPLY_FALLBACK_RE_LEGACY = re.compile(r'"reply"\s*:\s*"(.*)"\s*\}\s*$', re.DOTALL)


def _fallback_parse(cleaned: str) -> Optional[dict]:
    """json.loads()失败时的宽容兜底，只处理"某个字符串字段内部有没转义引号"
    这一种具体的失败模式，不是通用JSON修复——見上方注释。

    is_chitchat不额外抓取：它是这次新加的字段，正则兜底这条路径本来就是
    给"reply里夹带没转义引号"这种具体故障兜底的，不值得为了一个新字段再
    加一条正则去解析可能同样被夹在坏格式里的布尔值。统一保守给True——
    等价于"退回到这次修复之前的行为"，不会引入新的回归，付出的代价只是
    "这一条极端情况下（LLM输出格式本身就有问题）新的陈述句可能被保守地
    排除出L2候选池"，跟"根本没走到这条修复"的旧行为一致，可以接受。

    wiki_query_kind/cargo_profession/cargo_rarity同理不额外抓取，统一给
    "entity"/空字符串——cargo_filter的JSON本身没有任何自由文本字段（不像
    reply是LLM自由生成的自然语言），几乎不会触发这条"某个字符串字段夹带
    没转义引号"的兜底路径，为这么罕见的组合再写抓取正则不值得；代价是
    这种极端情况下一次真正的cargo_filter请求会被保守地当成"没有筛选条件
    的entity查询"（keyword为空，后面`_parse_response()`会因为keyword为空
    直接判失败返回None），等价于"这次没能识别出筛选意图，退回本地知识库
    再试"，比强行拿一个不完整的筛选条件去查询更安全。
    """
    route_match = _ROUTE_FALLBACK_RE.search(cleaned)
    if route_match is None:
        return None
    route = route_match.group(1)

    if route == "general":
        reply_match = _REPLY_FALLBACK_RE.search(cleaned) or _REPLY_FALLBACK_RE_LEGACY.search(cleaned)
        reply = reply_match.group(1).strip() if reply_match else ""
        if not reply:
            return None
        return {"route": "general", "keyword": "", "reply": reply, "is_chitchat": True,
                "wiki_query_kind": "entity", "cargo_profession": "", "cargo_rarity": "",
                "needs_search": False}

    if route == "wiki":
        keyword_match = _KEYWORD_FALLBACK_RE.search(cleaned)
        keyword = keyword_match.group(1).strip() if keyword_match else ""
        if not keyword:
            return None
        return {"route": "wiki", "keyword": keyword, "reply": "", "is_chitchat": True,
                "wiki_query_kind": "entity", "cargo_profession": "", "cargo_rarity": "",
                "needs_search": False}

    # route in ("local_kb", "history")：这两种情况keyword/reply本来就该是
    # 空字符串，不需要额外抓取。
    return {"route": route, "keyword": "", "reply": "", "is_chitchat": True,
            "wiki_query_kind": "entity", "cargo_profession": "", "cargo_rarity": "",
                "needs_search": False}


def _parse_response(raw: str) -> Optional[dict]:
    """防御性解析，跟llm_l1.py同一套做法：格式不对/字段缺失一律返回None，
    不让调用方拿到一个半成品结果；json.loads()失败时先试一下`_fallback_parse()`
    这个宽容兜底，见上方注释。"剥代码块外壳+json.loads()+失败返回None"这两步
    是从llm_json.py共用的，跟llm_l1.py/llm_l0_summary.py是同一套逻辑，见
    该文件模块开头的说明。

    wiki_query_kind/cargo_profession/cargo_rarity：④索引/聚合/动态更新新增
    的三个字段，见WikiRouteResult定义处和模块开头"④索引/聚合/动态更新"
    一节。这里有一处需要特别注意、容易踩的坑：原来"route=='wiki'却没给
    keyword就判失败"这条校验，对cargo_filter这一支是不成立的——cargo_
    filter天生就不需要keyword（用cargo_profession/cargo_rarity代替），
    如果不加区分地沿用旧校验，会把每一次合法的cargo_filter请求都误判成
    解析失败，所以下面把这条校验拆成了"是不是cargo_filter"两个分支分别
    处理，不是简单地把原校验删掉。
    """
    cleaned = strip_json_fence(raw)
    data = try_parse_json(cleaned)
    if data is None:
        return _fallback_parse(cleaned)

    if data.get("route") not in ("wiki", "local_kb", "history", "general"):
        return None

    keyword = data.get("keyword", "")
    reply = data.get("reply", "")
    if not isinstance(keyword, str) or not isinstance(reply, str):
        return None
    keyword = keyword.strip()
    # LLM偶尔把回复里的换行写成字面的"\n"（JSON里多转义了一层），解析后就是反斜杠+n
    # 两个字符，QQ里原样显示成"\n\n"（2026-09-28真机：私聊让它给聊天摘要时满屏\n）
    reply = reply.replace("\\n", "\n").strip()

    # wiki_query_kind/cargo_profession/cargo_rarity：字段缺失/类型不对一律
    # 保守当"entity"/空字符串处理，跟is_chitchat同样的降级哲学——不会把一个
    # 格式有问题的结果误判成合法的cargo_filter请求继续往下走。非wiki路由下
    # 这三个字段没有意义，统一归零，不让调用方看到无意义的残留值。
    wiki_query_kind = data.get("wiki_query_kind", "entity")
    if wiki_query_kind not in ("entity", "cargo_filter"):
        wiki_query_kind = "entity"
    cargo_profession = data.get("cargo_profession", "")
    cargo_rarity = data.get("cargo_rarity", "")
    if not isinstance(cargo_profession, str):
        cargo_profession = ""
    if not isinstance(cargo_rarity, str):
        cargo_rarity = ""
    cargo_profession = cargo_profession.strip()
    cargo_rarity = cargo_rarity.strip()

    if data["route"] != "wiki":
        wiki_query_kind, cargo_profession, cargo_rarity = "entity", "", ""

    is_cargo_filter = data["route"] == "wiki" and wiki_query_kind == "cargo_filter"

    if is_cargo_filter and not (cargo_profession or cargo_rarity):
        # cargo_filter却一个筛选条件都没给，是个不完整/自相矛盾的结果，
        # 当解析失败处理——跟"route=wiki(entity)却没给keyword"是对称的
        # 校验，只是换成了这一支自己的必填字段。
        return None

    if data["route"] == "wiki" and not is_cargo_filter and not keyword:
        # 普通entity查询却没给关键词，是个不完整/自相矛盾的结果，当解析
        # 失败处理，不要带着一个空关键词去查prts.wiki（那样搜索接口大概率
        # 会报错或搜到无意义的结果）。cargo_filter这一支不受这条限制，已经
        # 在上面单独校验过。
        return None

    if data["route"] == "general" and not reply:
        # 同理，route=general却没给回复文字，没法直接用，当解析失败处理，
        # 让调用方跳过这条路径、落到本地知识库正常走一遍拒答流程，总比拿一个
        # 空字符串当回复展示给用户要诚实。
        return None

    # is_chitchat：只在route="general"时有实际意义（决定这一轮是否从L2候选
    # 池排除，见_dispatch_with_router()）。字段缺失/类型不对一律保守当True
    # 处理——等价于这次修复之前的行为，不引入"该排除的闲聊没排除"这种新的
    # 回归；只有LLM明确给出布尔值False时才采纳False。
    is_chitchat = data.get("is_chitchat", True)
    if not isinstance(is_chitchat, bool):
        is_chitchat = True

    # needs_search：只在route="general"时有意义。缺失/类型不对一律当False，
    # 等价于加这个字段之前的行为（直接用reply），不会因为解析问题多花一次
    # 搜索、也不会让主观闲聊莫名其妙带上"来自网络搜索"的标注。
    needs_search = data.get("needs_search", False)
    if not isinstance(needs_search, bool) or data["route"] != "general":
        needs_search = False
    # search_query：只在needs_search时有意义；缺失/类型不对给""，调用方退回用原句搜索
    search_query = data.get("search_query", "")
    if not isinstance(search_query, str) or not needs_search:
        search_query = ""
    search_query = search_query.strip()[:80]
    retrieval_query = data.get("retrieval_query", "")
    if not isinstance(retrieval_query, str) or data["route"] not in ("local_kb", "history"):
        retrieval_query = ""
    retrieval_query = retrieval_query.strip()[:300]

    # kb_query_kind：只在route="local_kb"时有意义，不认识的值一律当"content"（走检索，
    # 等价于加这个字段之前的行为）
    kb_query_kind = data.get("kb_query_kind", "content")
    if kb_query_kind not in ("catalog", "content") or data["route"] != "local_kb":
        kb_query_kind = "content"
    # catalog_scope：同上，不认识的值/不是catalog一律当"all"（等价于加这个字段之前的行为）
    catalog_scope = data.get("catalog_scope", "all")
    if catalog_scope not in ("all", "jc", "not_jc") or kb_query_kind != "catalog":
        catalog_scope = "all"
    # kb_scope：同上，不认识的值一律当"all"（全库检索，等价于加这个字段之前的行为）
    kb_scope = data.get("kb_scope", "all")
    if kb_scope not in ("project", "paper", "all") or data["route"] != "local_kb":
        kb_scope = "all"
    # jc_session：只在问具体内容（content）时有意义；不认识的值一律当""（不按JC过滤）
    jc_session = data.get("jc_session", "")
    if not isinstance(jc_session, str) or kb_query_kind != "content" or not (
            jc_session in ("latest", "earliest") or re.fullmatch(r"\d{4}(-\d{2}(-\d{2})?)?", jc_session)):
        jc_session = ""

    return {
        "route": data["route"],
        "keyword": keyword,
        "reply": reply,
        "is_chitchat": is_chitchat,
        "wiki_query_kind": wiki_query_kind,
        "cargo_profession": cargo_profession,
        "cargo_rarity": cargo_rarity,
        "needs_search": needs_search,
        "search_query": search_query,
        "retrieval_query": retrieval_query,
        "kb_query_kind": kb_query_kind,
        "catalog_scope": catalog_scope,
        "kb_scope": kb_scope,
        "jc_session": jc_session,
    }


def route_query(text: str, client, recent_context=None) -> Optional[WikiRouteResult]:
    """核心入口。client 是 llm_client.py 里 build_llm_client() 返回的对象。

    recent_context：可选，L0维护的最近几轮对话（调用方从`get_context_window()`
    取，格式见`context_format.format_recent_context()`），用来帮助解析当前这句话里的代词/
    省略指代（比如"她精二的材料是什么"里的"她"），不传时行为跟改动前逐字
    一致——见模块开头"真机测试暴露的第6个问题"一节。

    返回 None 表示这次没能判断出路由（LLM不可用/调用失败/解析失败），调用方
    应该跳过这一步，只走本地知识库L3——不代表"这个问题没有答案"，只是
    "路由这一步没能给出明确判断"，比强行猜一个路由更诚实。
    """
    if not getattr(client, "is_real", False):
        return None

    messages = [
        # 自我介绍跟开场白共用 bot_profile.py 里同一份，被问"你是谁/能做什么"
        # 时general回复按它说，不临时编
        {"role": "system", "content": f"{_SYSTEM_PROMPT}\n{self_description_for_prompt()}{current_time_line()}"},
        {"role": "user", "content": with_current_message(recent_context, text)},
    ]
    raw = client.chat(messages, caller="route_query()")
    if raw is None:
        return None

    parsed = _parse_response(raw)
    if parsed is None:
        print(f"[wiki_router] LLM返回的内容不是预期的JSON格式（原始输出：{raw!r}），"
              f"本轮跳过这一步，只走本地知识库。")
    return parsed


_DISAMBIGUATE_SYSTEM_PROMPT = """\
你在帮用户在明日方舟wiki（prts.wiki）上定位正确的页面。用户问了一个问题，
系统从这句话里抽出了一个关键词去搜索，但这个关键词不是任何页面的精确
标题，模糊全文搜索返回了几个候选页面（标题+命中关键词的正文摘要）。

请判断这些候选里，有没有一个明确就是用户问题真正想问的那个页面。如果有，
输出该候选的完整标题（必须跟给出的候选标题逐字一致，不要改写、不要
翻译、不要加书名号）；如果都不像、或者拿不准，输出空字符串——诚实地说
"没有找到"，比勉强选一个不对的候选更好，不要因为"总得选一个"就凑合。

只输出如下JSON，不要有任何其它文字、不要用```包裹：
{"title": "<选中的候选标题，或者空字符串>"}
"""


def disambiguate_wiki_candidates(question: str, keyword: str, candidates: list, client) -> Optional[str]:
    """`prts_wiki_client.py`里`Disambiguator`回调类型的具体实现——完整设计
    动机见模块开头"新增disambiguate_wiki_candidates()"一节。

    candidates：`search_page_candidates()`返回的候选列表，每条
    `{"title": ..., "snippet": ...}`；调用方保证长度>=2（`_locate_page()`
    只在候选不止一条时才会调用这个函数）。

    返回选中的候选标题（保证是candidates里某一条的title，已经过校验），
    或者None表示：LLM不可用/调用失败/解析失败/LLM明确判断都不确定/LLM
    返回的标题不在候选列表里（防御性校验，不盲目信任LLM的输出一定是
    候选集合里的原文）——这几种情况在这里统一处理成"诚实地找不到"，调用方
    （`_locate_page()`）不需要、也没办法区分是上面哪一种。
    """
    if not getattr(client, "is_real", False):
        return None

    candidates_block = "\n".join(
        f"- 标题：{c['title']}\n  摘要：{c['snippet']}" for c in candidates
    )
    messages = [
        {"role": "system", "content": _DISAMBIGUATE_SYSTEM_PROMPT},
        {"role": "user", "content": f"用户问题：{question}\n抽出的关键词：{keyword}\n候选列表：\n{candidates_block}"},
    ]
    raw = client.chat(messages, caller="disambiguate_wiki_candidates()")
    if raw is None:
        return None

    parsed = try_parse_json(strip_json_fence(raw))
    if not isinstance(parsed, dict):
        return None
    title = parsed.get("title")
    if not isinstance(title, str):
        return None
    title = title.strip()
    if not title:
        return None

    valid_titles = {c["title"] for c in candidates}
    if title not in valid_titles:
        print(f"[wiki_router] disambiguate_wiki_candidates()返回的标题“{title}”"
              f"不在候选列表{sorted(valid_titles)}里，按拿不准处理。")
        return None
    return title


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from llm_client import build_llm_client

    client = build_llm_client()
    print(f"当前LLM后端：{client.name}（is_real={client.is_real}）\n")

    cases = [
        "能天使是什么职业？",
        "能天使的天赋是什么，模组怎么加成的？",
        "推进之王的技能是什么？",
        "变异检测常用什么工具？",
        "这个课题用的是什么动物模型？",
        "你好",
        "谢谢你，辛苦了",
        "今天吃什么，推荐几道菜吧",
        "明日方舟好玩吗",
        "我们组打算用什么做向量数据库来着？",
        "你还记得我们聊过要用什么数据库吗",
        "我们组决定用Chroma做向量数据库",
        "你好，吃了吗，我们组决定用Chroma做向量数据库",
        # ④索引/聚合/动态更新新增的三类问题，见模块开头同名一节
        "5-10的boss是谁",
        "今天哪个干员过生日",
        "最近新增了哪些干员和关卡",
        "六星狙击干员有哪些",
        "有哪些五星医疗干员",
    ]
    for text in cases:
        result = route_query(text, client)
        print(f"消息={text!r}\n  -> {result}\n")

    # ---- 诊断：对比有无recent_context，route_query()对代词指代问题的判断
    # 差异——真机复现过的具体案例："能天使的三技能专三的材料是什么"之后
    # 紧接着问"她精二的材料是什么"，不带上下文时"她"没有任何线索，容易被
    # 误判成route="history"；带上下文时应该能解析出"她"=能天使，正确判给
    # route="wiki"并给出keyword="能天使"。见模块开头"真机测试暴露的第6个
    # 问题"一节。
    print("=" * 70)
    print("诊断：对比有无recent_context，route_query()对代词指代问题的判断差异")
    print("=" * 70)
    diag_text = "她精二的材料是什么"
    diag_context = [
        {"role": "user", "text": "能天使的三技能专三的材料是什么"},
        {"role": "assistant", "text": "能天使的三技能「过载模式」专精三级的材料是：技巧概要·卷3 ×15、D32钢 ×6、聚酸酯块 ×6"},
    ]
    print(f"消息={diag_text!r}")
    print(f"  不带recent_context -> {route_query(diag_text, client)}")
    print(f"  带recent_context   -> {route_query(diag_text, client, recent_context=diag_context)}")

    # ---- 诊断：disambiguate_wiki_candidates()——候选列表消歧，见模块开头
    # "新增disambiguate_wiki_candidates()"一节+`prts_wiki_client.py`模块开头
    # "坑A遗留的一个更根本的问题"一节。这里手工构造候选列表做诊断，不依赖
    # 真实网络请求（真正连prts.wiki搜索的完整链路验证需要在conversation_
    # memory.py接入之后，用你自己电脑上的真实网络测）。
    print("=" * 70)
    print("诊断：disambiguate_wiki_candidates()——候选列表消歧")
    print("=" * 70)

    # 案例1：复现坑A本身的场景——"能天使"精确标题命中时根本不会走到这个
    # 函数，但假设它没有精确标题（比如换一个真实存在歧义的关键词），候选里
    # 混了一个道具页面，应该选中干员本人的页面，不是被相关度更高的道具页面
    # 带偏。
    case1_candidates = [
        {"title": "能天使的中坚信物", "snippet": "……能天使的专属道具，用于……"},
        {"title": "能天使", "snippet": "……狙击干员，企鹅物流成员，速射手……"},
    ]
    chosen1 = disambiguate_wiki_candidates("能天使是什么职业", "能天使", case1_candidates, client)
    print(f"案例1（应选中《能天使》）-> {chosen1!r}")

    # 案例2：候选里没有一个真的对应问题，应该诚实返回None，不勉强选一个。
    case2_candidates = [
        {"title": "能天使", "snippet": "……狙击干员，企鹅物流成员……"},
        {"title": "德克萨斯", "snippet": "……近卫干员，企鹅物流队长……"},
    ]
    chosen2 = disambiguate_wiki_candidates("明日方舟主线第五章的最终boss是谁", "主线第五章", case2_candidates, client)
    print(f"案例2（候选都不对应，应返回None）-> {chosen2!r}")
