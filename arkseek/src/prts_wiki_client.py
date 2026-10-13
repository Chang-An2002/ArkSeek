# -*- coding: utf-8 -*-
"""
prts.wiki（明日方舟中文Wiki）实时查询客户端 —— 进阶目标"问答版明日方舟万事通"的
第一块原型，对应你确认的两条设计决定："知识库分开"+"检索方式选实时查"。

【为什么是"实时查"而不是"定期爬到本地文件"】
现有 pipeline.py 的 MemorySystemPipeline 是照着"开题报告/中期报告这种几乎不变的
静态文档"设计的——爬一次、建好索引，之后可以放心用很久。明日方舟的游戏内容
（干员数值、干员池、活动、剧情）更新频率高得多，爬到本地的快照很快就会过时，
"过时的答案"比"答不上来"更容易误导人。所以这里选择每次提问时直接查prts.wiki
的实时接口，牺牲一点响应速度换取内容始终最新——不复用现有的 kb_dir 索引，
是一条完全独立的查询路径。

【为什么是"知识库分开"而不是塞进现有的 data/knowledge_base/】
现有知识库调好的置信度阈值（BM25_FLOOR/RERANK_FLOOR等）是照着两篇论文的语料
规模、领域、检索行为标定的，明日方舟维基的内容体量、领域、更新方式都完全不同，
混在一起会让这些阈值失去意义，还可能互相干扰。所以这是第二个独立的
"检索——生成"实例，跟现有 MemorySystemPipeline 平级，不是它的扩展。

【这算不算脱离阶段一/二的主线框架——不算，是L3这一层的延伸】
方案文档里L3（知识库问答）这一层，本来的定义就是"外部知识来源+检索+回答"，
现有实现只是把"外部知识来源"具体化成了"本地论文文件"。这里做的事情本质上
还是"给定一个问题，找到相关的外部内容，交给LLM生成回答"，只是把"外部知识
来源"换成了"prts.wiki的实时查询接口"——架构角色不变，属于阶段一/二核心逻辑
demo范围内的延伸，跟阶段三（接入QQ这个具体IM平台）完全不搭边、不冲突，
不需要等QQ官方审核通过才能做。

【真实踩坑记录A/B：找页面、清洗正文】

坑A——模糊搜索找错了页面：问"能天使是什么职业"，`search_page_title("能天使")`
搜到的第一条结果是《能天使的中坚信物》（一个道具页面，标题里恰好也包含"能
天使"三个字），不是干员本人的页面。原因是MediaWiki标准全文搜索按"文本相关度"
排序，不等于"标题精确匹配优先"。修复：`_locate_page()`先尝试把关键词当成
精确页面标题直接取（`action=parse&page=<keyword>&redirects=1`，顺便处理
"简称重定向到正式页面"的情况），精确标题不存在时才退回模糊搜索。

坑B——解析原始wikitext把干员数据一起当噪音删掉了：第一版清洗逻辑是把
wikitext里所有`{{模板}}`整个删掉，但明日方舟维基的干员页面几乎全部实质内容
（职业、天赋、技能数值）恰恰是靠模板拼出来的——不是"正文夹杂着几个模板"，
而是"正文本身就是模板"。修复：改用`action=parse&prop=text`拿MediaWiki自己
渲染好的HTML（模板已经在服务器端展开成最终显示的文字/表格），清洗HTML标签
而不是清洗未展开的wikitext。

【坑C/D，以及一次架构反思：切分+精排这套工具，从一开始就不适合"已经定位到
单一页面"这个场景】

坑A/B修好、页面能正确取到干净正文之后，做了"长页面分块+精排"：复用
`chunking.py`的`chunk_text()`把整页切成候选片段，`reranker.py`的精排模型
挑最相关的几条交给LLM——这是照搬阶段一"从181个论文片段里检索"的思路。
接下来两轮真机验证暴露的问题，最后指向的是同一个更根本的判断错误：

- **坑C**：不分小节直接按字符数切分，导致"天赋"这个词在"干员档案"背景
  故事小节（日常口语用法，"有天赋"）和"天赋"技术小节（游戏机制，被动技能）
  两种截然不同的含义被混在一起——精排选出的候选片段是背景故事里"她在射击
  方面的天赋，包括动态视力……"这类叙事文字，不是真正的技能效果说明。整页
  不切分时反而答对了，因为LLM能看到"天赋"标题紧挨着效果说明的完整版面
  结构，靠位置线索自己分辨对了；切成200字小片段后，这个结构信息全部丢失。
  **尝试的修复**：新增`chunk_page_by_section()`，先按`<h1>`~`<h6>`标题把
  正文切成"小节名->小节正文"，每个片段带上"【小节名】"前缀标签消歧。

- **坑D**：加了标签之后，LLM生成时确实学会了区分两种"天赋"（回答里如实说
  "片段里只给出了效果描述，没提到具体名称"），但精排本身还是把叙事性的
  "干员档案"文字打了更高分——根因是精排模型系统性偏爱"读起来流畅像段完整
  回答"的散文，"第一天赋·快速弹匣：精英1阶段+6"这种简短但准确的技术描述
  反而容易被打低分，标签这几个字不足以扭转这个打分习惯，跟`reranker.py`
  记录的"复合提问置信度系统性偏低"是同一类模型固有局限。**尝试的修复**：
  加`_boost_by_section_match()`规则——问题原文里出现的小节名（比如"天赋"），
  比精排的语义打分更可靠，命中就整体排到前面。

- **规则修好之后，新问题**：给"天赋"命中的候选片段做硬优先，把"职业"相关
  的候选挤出了结果——这是同一个问题的另一种表现：一个"规则打补丁"只能
  照顾到它专门设计针对的那种情况（这次是"天赋"），照顾不到的情况（这次是
  "职业"）反而被新规则挤得更惨。而且即使"天赋"本身答对了，回答也丢失了
  "潜能提升""模组"这些没有被问题字面提到、但整页阅读时LLM会自己关联起来
  的加成信息——这些信息分散在别的小节里，靠"小节名是否精确出现在问题里"
  这种规则永远补不完。

**反思：三轮真机验证里，唯一一次"职业和天赋都答对、而且回答组织得有条理
（分别列出第一/第二天赋，写明潜能和模组的加成，最后总结）"的，恰恰是最早
那次完全不切分、把整页正文直接交给LLM的版本。** 这不是巧合：切分+精排是
为"从一大堆文档里挑出最相关的几条"设计的（阶段一的181个论文片段，语料
太大、不可能整个塞进一次LLM调用），但这里的场景是"已经确定就是这一个
页面"，页面本身也没有大到超出LLM一次调用的舒适区间（干员页面通常一两万
字符）——用检索去在一个已知的单一页面内部"挑重点"，反而丢掉了LLM自己
阅读连续上下文、结合标题结构做综合判断的能力，还引出"规则越打补丁越多、
但每条规则只照顾到它针对的那一种情况"这个成本持续上升、覆盖面却上不去
的坏循环。继续在这个方向上"发现一个新反例就加一条新规则"，只会越改越
复杂、越难维护（这正是"照顾了天赋却挤掉职业"这个新bug暴露出来的信号）。

**最终结论：放弃切分+精排+规则打补丁，回到"整页交给LLM"，只做一件轻量
的事——砍掉几个固定的、体量大但几乎不会被用来回答事实性问题的小节**
（语音记录/干员密录/悖论模拟/干员模型，见下方`_LOW_VALUE_SECTIONS`），
控制一下prompt长度，不做逐候选打分、不做规则消歧。`chunk_page_by_section()`
/精排/`_boost_by_section_match()`这些代码已经从这个文件里删掉——不是注释
掉、是真的删掉，免得留着一段不再被调用的复杂逻辑，让以后看代码的人（包括
以后的自己）分不清"现在实际生效的到底是哪条路径"。章节切分的能力
（`_clean_html_keep_sections()`/`_split_into_sections()`）还留着，但用途
变了：不再是"挑出最相关的几条"，而是"筛掉几个已知没用的大块"，是一个
简单得多、也稳定得多的规则（这几个小节几乎不可能回答任何事实性问题，
不需要看问题内容就能判断，不是针对某一类问题的特判）。

【重要：本文件必须在你自己的电脑上运行，不能在云端沙盒里跑通】
云端沙盒和它能连的设备通道，访问外网都要走Anthropic的出站白名单，prts.wiki
不在这个白名单里——实测直接对 prts.wiki/api.php 发请求会被挡在403（大概率是
Cloudflare一类的机器人防护），所以每一版修改都没法在这边真实调用验证，只做
语法检查，需要你在自己电脑上跑一遍 `python prts_wiki_client.py`（或
`run_prts_wiki_demo.py`）才能验证。

【一个还没有答案的问题，需要你自己确认：prts.wiki是否允许这样的自动化访问】
之前调研没能确认prts.wiki的服务条款/robots.txt对自动化查询的具体态度（同样
被403挡住，看不到页面内容）。这里的原型请求量很小（一次提问=最多两次接口
调用），而且带上了标识自己身份的User-Agent，不是高频爬取；但如果以后真要
把这个功能长期挂在QQ机器人上大量使用，请你自己先确认一下prts.wiki是否欢迎
这种用法，这不是我能替你判断的事情。

【坑E：材料图标名称在HTML清洗中丢失——图标是"纯图标链接"，名字只活在
title属性里，不是img的alt】
用户报过一个具体案例（"能天使二技能材料"）：材料这类信息在页面上是靠一个
图标表示的，但清洗完的正文里这个图标对应的材料名完全消失，只剩下一个孤零零
的数字（比如需要的数量"15"），看不出这个数量指的是哪个材料。

用户从真实页面截了一段HTML确认了根因，跟`_HTML_TAG_RE`这条"整个标签连
属性一起删"的清洗方式直接相关，但具体丢失的位置比预想的更刁钻——一开始
猜测材料名可能在`<img>`的`alt`属性里（常见的图片替代文字写法），结果实际
拿到的标签里`<img>`根本没有`alt`属性；材料名其实写在**包住这个`<img>`的
`<a>`标签的`title`属性里**（`<a href="/w/..." title="技巧概要·卷3"><img .../></a>`），
这是MediaWiki渲染"文件/图标链接"（`typeof="mw:File"`）的标准写法——图标本身
不带名字，名字挂在外面那层链接的title上，而且这个`<a>`标签内部，除了这个
`<img>`，再没有任何其它可见文字。`_HTML_TAG_RE`把`<a ...>`整个标签（连同
`title`属性）当噪音一起删掉，`<img>`标签本身也整个删掉，这段HTML唯一还剩下
的可见文字，只有旁边那个独立的`<span class="prts-item-quantity-label">15
</span>`——材料名没有任何备份能在清洗后存活。用真实截图里的这段HTML直接
喂给`_clean_html()`复现过：清洗后的结果就是孤零零一个`"15"`，材料名完全
消失，跟用户报的现象完全对得上。

**修复**：不是针对"材料表格"这一种具体场景写死规则（那样只能照顾到材料
图标，别的地方用同一种写法插的图标——比如干员属性图标、职业图标——还是会
丢名字），而是照MediaWiki本身"文件链接"这个通用写法来修：在通用标签清洗
之前，先识别"一个`<a>`标签，带`title`属性，内部除了一个`<img>`没有任何其它
可见内容"这种模式（`_ICON_LINK_RE`），把整个`<a>...</a>`替换成方括号包住的
`title`文字（比如`[技巧概要·卷3]`），再交给后面的通用标签清洗——这样任何
地方出现的"纯图标链接"都能保住名字，不是只修材料这一种。跟`_LOW_VALUE_
SECTIONS`那条设计选择是同一个原则："看的是HTML本身的结构模式，不看问题
内容/不看具体是哪张表"，泛化性更好，也不会陷入"发现一个新反例就加一条新
规则"的坏循环（见模块开头坑C/D那次教训）。

**如实记录的局限**：这仍然是一条正则规则，不是真正的HTML解析器（这个文件
从一开始就是"只做最基础的标签剥离，不追求完美"这个定位，没有引入
BeautifulSoup这类解析库，这次也没有改变这个选择）。它能覆盖的是"纯图标
链接、`title`在`<a>`标签上"这一种具体的、已经用真实页面验证过的模式；如果
以后遇到"名字既不在`<a title=...>`也不在`<img alt=...>`、而是藏在别的什么
属性或者更深层嵌套结构里"的新反例，这条规则不会自动覆盖到，需要再看到
具体案例才能补——规则只覆盖已经验证过的具体模式，是一类需要如实说明的
局限。

【坑A遗留的一个更根本的问题：模糊搜索兜底只取第一条，"关键词短语压根没有
对应任何真实页面标题"这类问题答不上来——已修复为候选列表+消歧】
坑A修的是"精确标题存在时，不该被模糊搜索的相关度排序抢走"（比如"能天使"
应该直接命中干员本人页面，不该被《能天使的中坚信物》这个道具页面抢先）；
但`search_page_title()`（旧版模糊搜索包装，2026-09架构复盘确认全仓库零
调用点后已删除，见HANDOFF.md㉗）本身留了一个明确写在文档字符串里的简化：不管精确
标题存不存在，模糊搜索兜底这条路径永远"只取排第一的结果"，完全不处理
"关键词有歧义、可能对应多个页面"，更没处理"关键词根本不是任何页面标题的
一部分，模糊搜索返回的候选里其实没有一条是真正想要的答案"这种情况——真机
测试里"明日方舟主线第五章的最终boss是谁"就是这后一种的真实案例：路由抽出
的关键词是"主线第五章"这样的短语，wiki上剧情章节的页面命名方式（具体关卡
代号/别的规则）跟这个短语大概率对不上，模糊搜索排第一的结果不一定是想要
的那个页面，甚至可能所有候选都不是（详见`arkseek/README.md`
"④索引/聚合/动态更新"一节和"接下来"第15条）。

**修复思路：把"模糊搜索只取第一条"换成"取前N条候选，交给上一层判断该选
哪个（或者都不像，诚实说没找到）"**，但这一层判断需要读懂用户原问题的语义，
这个文件从第一行代码开始就是纯wiki I/O、不依赖任何LLM（LLM相关的判断全部
在`wiki_router.py`/`conversation_memory.py`那一层）——不想为了这一个改动
打破这条一直坚持的边界。所以选择用一个可选的回调函数（`disambiguator`）
做这层解耦：`_locate_page()`/`fetch_page_for_answer()`
都新增一个`disambiguator`可选参数，类型是
`Callable[[str, List[Dict[str, str]]], Optional[str]]`（输入关键词+候选
列表，输出选中的标题或者None表示都不确定）——这个文件本身完全不关心
`disambiguator`内部是不是调用了LLM，只负责"候选不止一条时，把候选交给
它、按它的返回值决定用哪个"这件事，真正的判断逻辑（`disambiguate_wiki_
candidates()`）写在`wiki_router.py`里，那边本来就是这个项目里做LLM
判断的地方。

**不传`disambiguator`时的行为完全不变**（默认值`None`）：候选不止一条也
还是取第一条——这保证了独立脚本（`run_prts_wiki_demo.py`等，这些场景要么
没有LLM在手边，要么就是想要旧的"尽量猜一个"行为）不会因为这次改动而报错
或者行为突变。只有`conversation_
memory.py`这条真正有LLM可用的路径会主动传入`disambiguator`，走新的候选
判断逻辑。

**候选数量为1时不调用`disambiguator`**，直接当成"精确匹配"处理——多数
"精确标题不存在但模糊搜索只搜到一个像样结果"的场景（比如简称/别名类
关键词）不该多花一次LLM调用去问"这一个候选选不选"，唯一的候选摆在那里，
选它不需要判断。这跟`prts_cargo_client.py`"能明确判断的地方不额外找LLM
兜底"是同一个成本意识。

**如实记录的边界**：`disambiguator`的判断质量完全取决于MediaWiki全文搜索
本身有没有把正确答案排进前N条候选里——如果连候选列表里都没有正确页面
（比如这个概念在wiki上压根没有对应的独立页面，纯属"没有这个页面"），
`disambiguator`诚实地在候选里挑不出来、返回None，这里就诚实地报告"没有
找到对应页面"，不会因为加了这层消歧就变成万能的搜索引擎——这跟
`prts_cargo_client.py`"不追求跟干员一览页面筛选结果完全一致"是同一种
诚实记录局限的态度，具体真机验证效果待你在自己电脑上确认。
"""

import html as html_module
import json
import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Callable, Dict, List, Optional, Tuple

# 关键词+候选列表（每条候选是{"title": ..., "snippet": ...}）-> 选中的标题，
# 或者None表示候选里没有一条是确定想要的答案——见模块开头"坑A遗留的一个更
# 根本的问题"一节。这个文件本身不实现任何一种具体的Disambiguator，只接受
# 调用方注入一个，保持这个模块LLM无关。
Disambiguator = Callable[[str, List[Dict[str, str]]], Optional[str]]

BASE_URL = "https://prts.wiki/api.php"

# 带上有辨识度的User-Agent，是对被访问方最基本的礼貌——让对方看日志时知道
# 这是什么来源的请求，而不是伪装成普通浏览器混在流量里。
_USER_AGENT = (
    "memory-system-prototype/0.1 "
    "(personal portfolio project by qikunrui; small-scale live query prototype)"
)


def _call_api(params: Dict[str, str]) -> Optional[dict]:
    """所有对 api.php 的请求走这一个函数——方便以后统一加超时/重试/限流逻辑。
    请求失败（网络不通/超时/返回非JSON）统一返回None，不抛异常，跟项目里
    llm_client.chat()同样的降级哲学：调用方看到None就知道这次查不到，
    自己决定接下来怎么办，而不是被一个没处理的异常打断。
    """
    query_params = dict(params)
    query_params.setdefault("format", "json")
    url = BASE_URL + "?" + urllib.parse.urlencode(query_params)
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        from timing import span
        with span("wiki请求", query_params.get("action", "")):
            with urllib.request.urlopen(request, timeout=10) as response:
                raw = response.read().decode("utf-8")
        return json.loads(raw)
    except Exception as e:
        print(f"[prts_wiki_client] 请求prts.wiki失败（{type(e).__name__}: {e}），本次查询不可用。")
        return None


_SEARCH_SNIPPET_TAG_RE = re.compile(r"<[^>]+>")


def _clean_search_snippet(snippet: str) -> str:
    """MediaWiki搜索接口返回的`snippet`字段是带HTML的（命中的关键词会被包在
    `<span class="searchmatch">...</span>`里高亮），这里只是简单剥掉标签、
    转回正常字符——跟这个文件一贯"只做最基础的标签剥离"的定位一致，这段
    摘要只是给消歧判断当参考用的辅助信息，不需要完美还原格式。
    """
    text = _SEARCH_SNIPPET_TAG_RE.sub("", snippet or "")
    return html_module.unescape(text).strip()


def search_page_candidates(keyword: str, limit: int = 5) -> Optional[List[Dict[str, str]]]:
    """用MediaWiki标准全文搜索接口（action=query&list=search）找跟关键词
    最相关的最多`limit`条候选页面，每条返回`{"title": ..., "snippet": ...}`
    （`snippet`是搜索接口自带的、命中关键词高亮的正文摘要，已去除HTML）。

    只作为`fetch_page_html()`精确标题匹配失败之后的兜底（见模块开头
    "坑A"）——全文搜索按相关度排序，不保证真正想要的页面排第一，调用方
    （`_locate_page()`）应该把整份候选列表交给消歧逻辑判断，而不是无条件
    取第一条。返回空列表表示搜索本身成功但没有任何匹配结果（诚实的"没有"，
    不是失败）；返回None表示请求本身失败或者返回的数据形状不对。
    """
    data = _call_api({
        "action": "query", "list": "search", "srsearch": keyword,
        "srlimit": str(limit), "srprop": "snippet",
    })
    if data is None:
        return None
    try:
        results = data["query"]["search"]
    except (KeyError, TypeError):
        print(f"[prts_wiki_client] 搜索接口返回的数据形状跟预期不一样，原始返回：{data}")
        return None
    return [
        {"title": r.get("title", ""), "snippet": _clean_search_snippet(r.get("snippet", ""))}
        for r in results
        if r.get("title")
    ]


# 【HTML清洗：只做最基础的标签剥离，不追求完美】
# 直接把 MediaWiki 渲染好的HTML丢给LLM会混着大量标签和排版噪音——这里用正则
# 做一遍简单清洗，目标是"读起来像正常中文段落/表格"，不是"完美还原网页视觉
# 结构"，能支撑LLM读懂内容即可。
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
_BLOCK_BREAK_RE = re.compile(r"</(p|div|tr|li|h[1-6]|table)>", re.IGNORECASE)
_LINE_BREAK_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_EDIT_LINK_RE = re.compile(r"\[(编辑|edit)\]", re.IGNORECASE)
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")

# 【坑E：纯图标链接，名字挂在<a title=...>上，不是<img alt=...>】见模块开头
# "坑E"一节的完整排查过程。MediaWiki渲染文件/图标链接的标准写法是
# `<a href="..." title="材料名"><img .../></a>`——img本身没有alt，名字在外层
# <a>的title属性上，而且<a>内部除了这一个img、没有任何其它可见文字。必须在
# _HTML_TAG_RE这条"整个标签连属性一起删"的通用清洗**之前**处理，不然title
# 这个属性会跟标签一起被删掉，没有第二个地方能找回这个名字。
# 用"非贪婪+排除嵌套</a>"这个写法（`(?:(?!</a>).)*?`）而不是简单的`.*?`，
# 是为了防止一页里连续出现多个图标链接时，非贪婪匹配跨过中间的</a>误吞下一个
# 链接——这个写法保证只匹配到最近的一个</a>为止。
_ICON_LINK_RE = re.compile(
    r'<a\b[^>]*\btitle="([^"]*)"[^>]*>(?:(?!</a>).)*?<img\b[^>]*>(?:(?!</a>).)*?</a>',
    re.DOTALL | re.IGNORECASE,
)


def _inline_icon_titles(raw_html: str) -> str:
    """把"纯图标链接"（<a title="材料名"><img .../></a>这种、内部没有其它
    可见文字的链接）替换成方括号包住的title文字，再交给后面的通用标签清洗——
    见模块开头"坑E"一节。不是只对材料表格生效，页面上任何符合这个HTML结构
    的图标链接（属性图标、职业图标等）都会被保住名字，是一条通用规则，不是
    针对某一种具体表格的特判。
    """
    def repl(match: "re.Match") -> str:
        title = html_module.unescape(match.group(1))
        return f"[{title}]"

    return _ICON_LINK_RE.sub(repl, raw_html)


def _clean_html(raw_html: str) -> str:
    text = _HTML_COMMENT_RE.sub("", raw_html)
    text = _SCRIPT_STYLE_RE.sub("", text)
    text = _inline_icon_titles(text)  # 必须在_HTML_TAG_RE之前，见"坑E"
    # 块级标签换成换行，避免整页文字被剥完标签之后挤成一长条不好读。
    text = _BLOCK_BREAK_RE.sub("\n", text)
    text = _LINE_BREAK_RE.sub("\n", text)
    text = _HTML_TAG_RE.sub("", text)
    text = html_module.unescape(text)  # &amp;/&nbsp; 这类HTML实体转回正常字符
    text = _EDIT_LINK_RE.sub("", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()


# 【章节感知清洗：跟 _clean_html() 走同一套清洗步骤，唯一区别是先把标题标签
# 标记出来、清洗完之后标记还留在文本里，供切出"小节名 -> 小节正文"的对应
# 关系。用途已经从"挑出最相关的几条候选"改成"筛掉几个已知没用的大块"——
# 见模块开头的架构反思。用 \x01 这个正常网页文本里不会出现的控制字符做
# 标记，不用担心跟真实内容混淆，也不受后面空格/换行清洗步骤影响。】
_HEADING_RE = re.compile(r"<h([1-6])\b[^>]*>(.*?)</h\1>", re.DOTALL | re.IGNORECASE)
_SECTION_MARK = "\x01"


def _mark_headings(raw_html: str) -> str:
    def repl(match: "re.Match") -> str:
        inner = _HTML_TAG_RE.sub("", match.group(2))
        inner = html_module.unescape(inner)
        inner = _EDIT_LINK_RE.sub("", inner).strip()
        return f"\n\n{_SECTION_MARK}{inner}{_SECTION_MARK}\n\n"

    return _HEADING_RE.sub(repl, raw_html)


def _clean_html_keep_sections(raw_html: str) -> str:
    text = _HTML_COMMENT_RE.sub("", raw_html)
    text = _SCRIPT_STYLE_RE.sub("", text)
    text = _inline_icon_titles(text)  # 必须在_HTML_TAG_RE之前，见"坑E"
    text = _mark_headings(text)  # 必须在通用标签清洗之前做，这时候 <h1>~<h6> 标签还完整
    text = _BLOCK_BREAK_RE.sub("\n", text)
    text = _LINE_BREAK_RE.sub("\n", text)
    text = _HTML_TAG_RE.sub("", text)
    text = html_module.unescape(text)
    text = _EDIT_LINK_RE.sub("", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()


def _split_into_sections(marked_text: str) -> List[Tuple[str, str]]:
    """把带 _SECTION_MARK 标记的文本切成 [(小节名, 小节正文), ...]。

    `_mark_headings()`把每个标题替换成"标记+标题文字+标记"，按标记切开后，
    标题文字和小节正文会交替出现在奇数/偶数位置。第一段（第一个标题之前的
    内容，比如页面最上面的名称+简介）归到"开头"这个小节名下。
    """
    parts = marked_text.split(_SECTION_MARK)
    sections: List[Tuple[str, str]] = []

    intro = parts[0].strip()
    if intro:
        sections.append(("开头", intro))

    i = 1
    while i + 1 < len(parts):
        name = parts[i].strip() or "未命名小节"
        body = parts[i + 1].strip()
        if body:
            sections.append((name, body))
        i += 2

    return sections


# 这几个小节体量往往很大（尤其是热门干员的语音记录，可能有几十条台词），
# 但内容是台词/花絮/立绘展示，几乎不会被用来回答"职业/属性/天赋/技能"这类
# 事实性问题——不管问题具体问的是什么，这几个小节的信息量都很低，所以直接
# 按小节名固定排除，控制一下prompt长度。这跟坑C/D里"针对某一类问题做规则
# 消歧"是两回事：这里排除的标准是"小节类型本身信息密度低"，不看问题内容，
# 不会出现"顾了这个问题又漏了那个问题"的情况。
_LOW_VALUE_SECTIONS = {"语音记录", "干员密录", "悖论模拟", "干员模型"}


def fetch_page_html(title: str) -> Optional[Dict[str, str]]:
    """按标题直接取页面渲染后的HTML（action=parse&prop=text&redirects=1）。

    渲染后的HTML是模板已经被MediaWiki服务器端展开、替换成最终显示内容的
    版本——不管页面用了多少层模板拼数据，拿到的都是真实显示的文字/表格内容，
    不会像直接解析wikitext原文那样把模板一起当噪音删掉、连数据本身也删没了
    （见模块开头"坑B"的真实教训）。

    `redirects=1` 顺带处理"关键词是简称/别名，实际重定向到正式页面"的情况。

    返回 {"title": 实际标题（跟着redirects解析过跳转，可能跟传入的title不同），
    "html": 渲染后的原始HTML}；标题不存在（比如关键词本身不是一个精确页面
    标题）或请求失败，返回None——调用方应该退回 search_page_candidates 模糊搜索，
    这不算异常情况。
    """
    data = _call_api({"action": "parse", "page": title, "prop": "text", "redirects": "1"})
    if data is None:
        return None
    if data and "error" in data:
        return None
    try:
        actual_title = data["parse"]["title"]
        raw_html = data["parse"]["text"]["*"]
    except (KeyError, TypeError):
        print(f"[prts_wiki_client] 页面接口返回的数据形状跟预期不一样，原始返回：{data}")
        return None
    return {"title": actual_title, "html": raw_html}


class _DisambiguationLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.depth = 0
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "li":
            self.depth += 1
        if tag != "a" or not self.depth:
            return
        attrs = dict(attrs)
        url = urllib.parse.urlsplit(attrs.get("href", ""))
        if url.netloc and url.netloc != "prts.wiki":
            return
        if "new" in attrs.get("class", "").split():
            return
        title = ""
        if url.path.startswith("/w/"):
            title = urllib.parse.unquote(url.path[3:]).replace("_", " ")
        elif url.path.endswith("index.php"):
            parameters = urllib.parse.parse_qs(url.query)
            if "action" not in parameters:
                title = parameters.get("title", [""])[0].replace("_", " ")
        if title and ":" not in title and title not in self.links:
            self.links.append(title)

    def handle_endtag(self, tag):
        if tag == "li":
            self.depth = max(0, self.depth - 1)


def _disambiguation_candidates(page):
    if not re.search(r"本页面为消歧义页|这是一个消歧义页|id=[\"']disambigbox[\"']", page["html"]):
        return None
    parser = _DisambiguationLinks()
    parser.feed(page["html"])
    titles = [title for title in parser.links if title != page["title"] and page["title"] in title]
    return [{"title": title, "snippet": title} for title in titles[:10]]


def _follow_disambiguation(page, keyword, disambiguator):
    if page is None:
        return None
    candidates = _disambiguation_candidates(page)
    if candidates is None:
        return page
    chosen = candidates[0]["title"] if len(candidates) == 1 else (
        disambiguator(keyword, candidates) if candidates and disambiguator else None)
    if chosen not in {candidate["title"] for candidate in candidates}:
        print("[prts_wiki_client] 命中消歧义页，但无法确定具体条目，不把索引当作正文。")
        return None
    print(f"[prts_wiki_client] 从消歧义页继续读取《{chosen}》。")
    target = fetch_page_html(chosen)
    return target if target is not None and _disambiguation_candidates(target) is None else None


def _locate_page(keyword: str, disambiguator: Optional[Disambiguator] = None) -> Optional[Dict[str, str]]:
    """关键词 -> {"title": ..., "html": ...}（渲染后的原始HTML，未清洗）。

    优先把关键词当成精确页面标题直接取（干员名、道具名这类专有名词大概率
    本身就是一个真实存在的标题），精确标题不存在时，才退回模糊全文搜索找
    最接近的页面——这个顺序是从真实踩坑改出来的，见模块开头"坑A"。

    `disambiguator`可选：模糊搜索候选不止一条时，交给这个回调判断该用
    哪一条（或者返回None表示都不确定）——不传时保持旧行为，直接取第一条
    候选，见模块开头"坑A遗留的一个更根本的问题"一节。`disambiguator`返回的
    标题会校验是否真的在候选列表里，不盲目信任（防御性编程，`disambiguator`
    内部的判断逻辑再怎么谨慎，这里也不该假设它一定守规矩）。

    `fetch_page_for_answer()`基于这个函数，"怎么找到页面"这段逻辑只写一份。
    （早期还有一个`query_prts_wiki()`同样基于这个函数，2026-09架构复盘确认
    零调用点后已删除，见HANDOFF.md㉗。）
    """
    exact = fetch_page_html(keyword)
    if exact is not None:
        print(f"[prts_wiki_client] 关键词“{keyword}”直接命中精确页面标题《{exact['title']}》。")
        return _follow_disambiguation(exact, keyword, disambiguator)

    candidates = search_page_candidates(keyword, limit=5)
    if not candidates:
        print(f"[prts_wiki_client] 没搜到跟“{keyword}”匹配的页面。")
        return None

    if len(candidates) == 1 or disambiguator is None:
        title = candidates[0]["title"]
        print(f"[prts_wiki_client] 精确标题未命中，模糊搜索“{keyword}”匹配到页面《{title}》，正在获取正文……")
        return _follow_disambiguation(fetch_page_html(title), keyword, disambiguator)

    chosen = disambiguator(keyword, candidates)
    valid_titles = {c["title"] for c in candidates}
    if chosen is None or chosen not in valid_titles:
        titles_preview = "、".join(c["title"] for c in candidates)
        print(f"[prts_wiki_client] “{keyword}”模糊搜索有{len(candidates)}个候选（{titles_preview}），"
              f"消歧判断不出确定的一个，诚实报告没有找到对应页面。")
        return None
    print(f"[prts_wiki_client] “{keyword}”模糊搜索有{len(candidates)}个候选，消歧判断选中《{chosen}》，正在获取正文……")
    return _follow_disambiguation(fetch_page_html(chosen), keyword, disambiguator)


def fetch_page_for_answer(keyword: str, disambiguator: Optional[Disambiguator] = None) -> Optional[Dict[str, str]]:
    """一站式函数：关键词 -> 页面正文（已经去掉`_LOW_VALUE_SECTIONS`里那几个
    体量大但几乎不含事实性游戏机制信息的固定小节，其余部分保持完整、连续，
    每个小节的标题也保留在文本里）——这是给`llm_generate_answer()`用的
    正文，直接整段交给LLM，不做切分/精排/候选打分。见模块开头的架构反思：
    "已经定位到单一页面"这个场景，让LLM自己阅读连续上下文、结合标题结构
    做综合判断，比切分成小候选片段再挑最相关的几条效果更好、代码也更简单。

    返回 {"title": ..., "text": ...}；任何一步失败都返回None。`disambiguator`
    透传给`_locate_page()`——见该函数说明和模块开头"坑A遗留的一个更根本的
    问题"一节；`conversation_memory.py`会传入一个基于LLM的实现，独立脚本
    不传时保持旧的"模糊搜索取第一条"行为。
    """
    page = _locate_page(keyword, disambiguator=disambiguator)
    if page is None:
        return None

    marked = _clean_html_keep_sections(page["html"])
    sections = _split_into_sections(marked)
    kept = [(name, text) for name, text in sections if name not in _LOW_VALUE_SECTIONS]
    full_text = "\n\n".join(f"【{name}】\n{text}" for name, text in kept)
    return {"title": page["title"], "text": full_text}


if __name__ == "__main__":
    print("【注意】这个自测必须在能直接访问 prts.wiki 的网络环境下运行"
          "（云端沙盒/远程设备通道连不上，需要在你自己电脑上跑）。\n")

    result = fetch_page_for_answer("能天使")
    if result is None:
        print("没有拿到结果——如果是网络问题，检查一下当前设备能不能直接打开"
              "https://prts.wiki ；如果是解析报错，把上面的报错信息发给Claude，"
              "照着实际返回的JSON结构改 fetch_page_html()。")
    else:
        print(f"页面标题：{result['title']}")
        print(f"正文长度：{len(result['text'])} 字符（已排除{_LOW_VALUE_SECTIONS}这几个固定小节）")
        print(f"正文前300字预览：\n{result['text'][:300]}")
