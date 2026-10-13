# -*- coding: utf-8 -*-
"""
翻聊天记录：隔不同轮数再问，能不能答对（2026-10-01~02）

背景：真机"我不喜欢游泳，只喜欢骑行"两轮后问"我不喜欢什么"答成"没翻到"——检索把
那句排第一，但"问句对陈述句"精排只有0.38，没过0.5门槛。这次加了：画像记"不喜欢"、
最近窗口的发言直接当证据、没过门槛时改写成陈述句重查（改写路径门槛0.75）、命中提问
时带上紧跟的那条回复。这个脚本验证后几项在不同距离下的效果，并给"要不要调门槛"提供数据。

做法：每道题一个新会话（:memory:，不碰data/memory.db）。先写入一条事实（用户说的，
或者助手说过的），再填若干轮对话，然后提问，走完整流程（路由->翻聊天记录->生成）。
距离：隔1轮（在最近窗口内）、隔5轮（刚出窗口）、隔20轮（很久以前）。窗口外的部分用一份
占位摘要模拟"已经并进摘要"的稳定状态（摘要不含这条事实，不让摘要替检索答题）。

第二版（2026-10-02，用户要求）加了**同话题干扰**：第一版填充的闲聊跟事实完全不相关，
干扰分几乎都是0，比真实聊天记录容易。现在每条事实配两轮同话题、内容不同的对话（比如
事实"买了捷安特公路车"，干扰"朋友买了电动车""骑行要注意什么"），穿插在中间；另加
两道贴得很近的诱导题（说过买自行车、问买了什么电脑；推荐过电影/综艺/小说、问推荐的
纪录片）。

除了最终回答，还记下精排分（衰减前）：目标（事实那一问一答）、同话题干扰、其它闲聊
里的最高分，原问法一次、改写成陈述句后一次——用来判断0.5/0.75这两个门槛合不合适。

用法：cd src && python run_l2_recall_eval.py [题号 ...]
报告写到 ../l2_recall_eval_results.md
"""

import builtins
import os
import sys
import time
from datetime import datetime

import conversation_memory as cm
from conversation_memory import ConversationSession
from llm_client import build_llm_client
from memory_store import MemoryStore
from pipeline import MemorySystemPipeline
from query_planner import rewrite_for_history

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "..", "l2_recall_eval_results.md")

# (编号, 说话人, 事实原话, 问题, 判对关键词, 同话题干扰两轮)
# 说话人assistant的那条：前面配一句用户提问 F7_ASK
FACTS = [
    ("F1", "user", "我们组决定用Chroma做向量数据库，主要是因为轻量、Python里直接能用",
     "我们组之前选的向量数据库是哪个来着", ["Chroma"],
     [("我们组的服务器准备换成A100显卡", "好的，算力会提升不少"),
      ("向量检索的召回率怎么提高", "可以试试混合检索，加上BM25关键词召回")]),
    ("F2", "user", "下周三我要去杭州开个学术会议", "我下周要去哪里出差来着", ["杭州"],
     [("下个月我想去成都吃火锅", "成都火锅确实值得专程去一趟"),
      ("出差住酒店有什么省钱技巧", "可以提前订，用会员价或者企业协议价")]),
    ("F3", "user", "导师让我把开题报告的第三章重写一遍", "导师之前让我改哪部分来着", ["第三章"],
     [("导师说下周组会要我汇报进展", "好的，可以提前准备下PPT"),
      ("开题报告的参考文献格式怎么统一", "可以用Zotero统一导出格式")]),
    ("F4", "user", "我昨天把实验用的小鼠从12只加到了20只", "实验小鼠我之前说加到多少只了", ["20"],
     [("实验室新到了一批细胞培养箱", "设备升级了，做实验更方便"),
      ("小鼠一般养到多大做实验", "看实验设计，常见是8周龄左右")]),
    ("F5", "user", "我上周买了一辆二手的捷安特公路车", "我之前说买了什么车来着", ["捷安特"],
     [("我朋友最近买了一辆电动车", "电动车通勤挺方便的"),
      ("骑行要注意什么", "记得戴头盔，长途注意补水")]),
    ("F6", "user", "国庆我打算去青岛玩三天", "我国庆打算去哪玩来着", ["青岛"],
     [("元旦我可能回老家", "回家团聚挺好的"),
      ("去海边旅游带什么好", "防晒霜和拖鞋必备")]),
    ("F7", "assistant", "可以看看《星际穿越》，节奏紧凑，配乐也很棒", "你上次推荐我看的电影是哪部", ["星际穿越"],
     [("推荐本科幻小说", "可以读读《三体》，设定很宏大"),
      ("推荐个下饭的综艺", "可以看看《向往的生活》，轻松解压")]),
]
F7_ASK = "周末想看部电影，推荐一个呗"
# 诱导题：(编号, 问题, 背景事实编号或None)。答"没翻到/没说过"之类算对，编出具体内容算错
PROBES = [
    ("N1", "我之前说我女朋友叫什么来着", None),
    ("N2", "我上次说要去哪个城市读博来着", None),
    ("N3", "我们组之前决定用什么做数据可视化来着", "F1"),  # 说过向量数据库，问数据可视化
    ("N4", "我之前说我买了什么电脑来着", "F5"),          # 说过买自行车，问电脑
    ("N5", "你上次推荐我看的纪录片是哪部", "F7"),        # 推荐过电影/综艺/小说，问纪录片
]
DISTANCES = [1, 5, 20]  # 事实和提问之间隔几轮

FILLER = [
    ("今天天气不错", "是呀，适合出去走走～"),
    ("你觉得明日方舟哪个干员最帅", "这个看个人口味啦，我挺喜欢银灰那种气质的"),
    ("中午吃什么好", "可以试试番茄炒蛋盖饭，简单又下饭"),
    ("最近有点累", "注意休息呀，别太拼了"),
    ("周末去爬山怎么样", "不错呀，记得带够水"),
    ("你会讲笑话吗", "会一点～为什么数学书总是很忧郁？因为它有太多问题"),
    ("晚上想喝点什么", "热牛奶助眠，或者来杯菊花茶"),
    ("我今天跑了五公里", "厉害！坚持下去会越来越轻松的"),
    ("你喜欢猫还是狗", "都挺可爱的，猫更独立一点"),
    ("怎么提高专注力", "可以试试番茄工作法，25分钟专注加5分钟休息"),
    ("最近在追什么剧", "我没法追剧啦，不过可以帮你找找口碑好的"),
    ("明天好像要降温", "记得多穿点～"),
]
NO_ANSWER_MARKS = ("没翻到", "没有提到", "没提到", "没说过", "没有说过", "没有找到", "不记得", "没有记录",
                   "没提过", "没有相关", "没有推荐过", "没推荐过", "没有给你推荐", "没有看到", "没看到", "记混")
FACT_BY_ID = {f[0]: f for f in FACTS}


def _quiet(fn, *a, **k):
    p = builtins.print
    builtins.print = lambda *x, **y: None
    try:
        return fn(*a, **k)
    finally:
        builtins.print = p


def build_session(kb, llm, fact, distance, uid):
    """返回(session, 目标turn_index集合, 同话题干扰turn_index集合)。
    同话题干扰两轮放在事实之后第1轮和第3轮（距离不够就放得下几轮放几轮）。"""
    s = ConversationSession(kb, MemoryStore(":memory:"), user_id=uid, llm_client=llm)
    s.web_search = None
    turns, target, same = [], set(), set()
    if fact is not None:
        _, who, text, _, _, distractors = fact
        turns += [("user", F7_ASK), ("assistant", text)] if who == "assistant" else [("user", text), ("assistant", "好的，了解～")]
        target = {0, 1}  # 事实那一问一答：助手那条存进检索时带着它回应的那句话
    else:
        distractors = []
    slots = {0: 0, 2: 1}  # 间隔里的第几轮放第几条干扰
    filler_i = 0
    for i in range(distance):
        if i in slots and slots[i] < len(distractors):
            q, a = distractors[slots[i]]
            same |= {len(turns), len(turns) + 1}
        else:
            q, a = FILLER[filler_i % len(FILLER)]
            filler_i += 1
        turns += [("user", q), ("assistant", a)]
    for role, text in turns:
        s.store.append_turn(uid, role, text)
    # 模拟稳定状态：窗口（window_size条）之外的都已并进摘要；摘要里不写这条事实
    cutoff = len(turns) - s.window_size - 1
    if cutoff >= 0:
        s.store.set_l0_summary(uid, "（测试用占位摘要）闲聊过天气、吃饭、运动、电影等日常话题。", cutoff)
    return s, target, same


def scores_of(s, query, target, same):
    """(目标, 同话题干扰, 其它) 三组里的最高精排分（衰减前；门槛临时设0，看原始分数）。"""
    old = cm.L2_RERANK_FLOOR
    cm.L2_RERANK_FLOOR = 0.0
    try:
        res = _quiet(s._search_history, query, top_n=60)
    finally:
        cm.L2_RERANK_FLOOR = old
    pick = lambda ids, inside: max((p.relevance_score for p in res if (p.chunk.chunk_id in ids) == inside
                                    and (inside or p.chunk.chunk_id not in target)), default=None)
    return (pick(target, True) if target else None, pick(same, True) if same else None,
            max((p.relevance_score for p in res if p.chunk.chunk_id not in target | same), default=None))


def main():
    llm = build_llm_client()
    if not llm.is_real:
        print("需要真实LLM才能跑。")
        return
    only = set(sys.argv[1:])
    kb = MemorySystemPipeline(os.path.join(HERE, "..", "data", "knowledge_base"))
    cases = [(f, d, None) for f in FACTS for d in DISTANCES] + \
            [((pid, None, None, q, None, None), 5, bg) for pid, q, bg in PROBES]
    rows, details = [], []
    n_ok = n_run = 0
    fmt = lambda x: "-" if x is None else f"{x:.2f}"
    for fact, dist, bg in cases:
        cid = f"{fact[0]}-{dist}"
        if only and cid not in only and fact[0] not in only:
            continue
        n_run += 1
        is_probe = fact[1] is None
        uid = f"l2eval_{cid}"
        s, target, same = build_session(kb, llm, FACT_BY_ID.get(bg) if is_probe else fact, dist, uid)
        if is_probe:
            target = set()  # 诱导题没有正确答案，背景事实也算干扰
        question = fact[3]
        rewrites = rewrite_for_history(question, llm)
        o = scores_of(s, question, target, same)
        rws = [scores_of(s, rw, target, same) for rw in rewrites]
        r = tuple(max((x[i] for x in rws if x[i] is not None), default=None) for i in range(3))
        logs = []
        p = builtins.print
        builtins.print = lambda *x, **y: logs.append(" ".join(str(v) for v in x))
        t0 = time.perf_counter()
        try:
            reply = s.user_says(question, defer_summary=True)
        finally:
            builtins.print = p
        dt = time.perf_counter() - t0
        if is_probe:
            ok = any(m in reply for m in NO_ANSWER_MARKS) or reply.strip() == cm._HISTORY_MISS_REPLY
        else:
            ok = all(k in reply for k in fact[4])
        n_ok += ok
        path = "改写重查" if any("改写成陈述句重查" in l for l in logs) else ("翻聊天记录" if "【L2情景记忆" in reply else "其它")
        mark = "✅" if ok else "❌"
        rows.append(f"| {cid} | {question} | {dist} | {mark} | {path} | {fmt(o[0])} | {fmt(o[1])} | {fmt(o[2])} | "
                    f"{fmt(r[0])} | {fmt(r[1])} | {fmt(r[2])} | {dt:.1f}s |")
        details += [f"### {cid} {mark} {question}（隔{dist}轮{'，背景' + bg if bg else ''}）", "",
                    f"- 改写：{rewrites}", f"- 回复：{reply.replace(chr(10), ' ')}", ""]
        print(rows[-1], flush=True)
    lines = ["# 翻聊天记录：隔不同轮数再问（含同话题干扰）", "", f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "",
             f"**{n_ok}/{n_run} 答对**（诱导题答\"没翻到/没说过\"之类算对）。"
             f"门槛：原问法{cm.L2_RERANK_FLOOR}，改写重查{cm.L2_REWRITE_RERANK_FLOOR}；窗口6条。", "",
             "分数是精排相关性（衰减前）里各组的最高分：目标=事实那一问一答，同话题=同话题干扰，其它=其它闲聊；"
             "诱导题没有目标，背景事实算进\"其它\"。改写列取两句改写里的最高。", "",
             "| 题 | 问题 | 隔几轮 | 结果 | 走的路 | 原问法·目标 | 原问法·同话题 | 原问法·其它 | 改写·目标 | 改写·同话题 | 改写·其它 | 用时 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"] + rows + ["", "## 每题详情", ""] + details
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{n_ok}/{n_run} 答对。报告：{os.path.abspath(REPORT)}")


if __name__ == "__main__":
    main()
