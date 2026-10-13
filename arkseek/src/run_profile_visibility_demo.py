# -*- coding: utf-8 -*-
"""画像可见性修复的端到端人工验证脚本——复现你两次真机测试里失败的那个
场景（"我导师推荐我读的是哪篇文献，我读完了吗"/"导师推荐我读的ABC我
读完了吗"），确认现在能不能正确从画像里把答案兜出来。

【这次要验证什么】
`conversation_memory.py`新增`_build_planner_context()`（顺手把画像打上
"（用户画像，代表当前已确认的最新状态）"标记注入到传给编排层的背景里）
+`_try_answer_from_profile_fallback()`（`_try_answer_from_history()`两处
"没查到可信答案"的give-up分支之前的最后一道画像兜底，调用`llm_l1.py`
新增的`find_profile_answer()`）；`llm_generate.py`的`_SYSTEM_PROMPT`也
新增了一条例外，允许生成模型把打了"用户画像"标记的背景行当成回答依据
直接引用。完整设计动机见`conversation_memory.py`模块开头"画像可见性"
一节+`llm_l1.py`模块开头对应一节+`llm_generate.py`模块开头对应一节。

【数据库：2026-09-29起改用内存数据库+预置画像】
原来故意用真实的`data/memory.db`（`default_user`，跟chat_cli共用），想验证"你
之前真的告诉过系统的画像能不能被用上"。代价是：每跑一次就往你的真实聊天记录里
加几轮测试对话，而且会把你真实画像里的内容（包括早期记下的真名）打印进报告——
2026-09-29重跑时就发生了，事后删掉了那6条记录、报告里的真名换成了"长安"。
用户决定改成跟其它测试一样：每次新建内存数据库（不碰data/memory.db），预置一份
画像（内容跟当初真机测试时画像里的那条事实一样），要验证的"画像能不能被用上"
这个逻辑不变。

【这次要验证的具体行为】
1. 打印预置的画像。
2. 问"我导师推荐我读的是哪篇文献，我读完了吗"——重点看终端有没有出现
   画像兜底相关的行为：如果走到了`_try_answer_from_history()`且候选
   检索本身很弱/为空，应该能看到最终答案正确提到文献信息，而不是像
   之前两次真机测试那样一次"没有找到相关的历史记录"、一次答非所问。
3. 换一种问法"导师推荐我读的ABC我读完了吗"，同样检查。
4. "能天使是什么职业？"——回归检查，这条问题走的是本地知识库路径，
   完全不涉及画像兜底，行为应该跟改动前完全一致，确认这次改动没有
   引入意外的副作用。

【必须在你自己电脑上跑】
需要真实LLM（MiniMax）——`find_profile_answer()`本身是新增的一次LLM
判断调用，LLM不可用的话这条兜底路径不会生效，`llm_generate.py`的新
例外也只在真实生成时才用得上。

用法：
    cd src
    python run_profile_visibility_demo.py
"""

import os

from conversation_memory import ConversationSession
from llm_client import build_llm_client
from memory_store import MemoryStore
from pipeline import MemorySystemPipeline

_log_lines = []


def log(msg: str = ""):
    print(msg)
    _log_lines.append(msg)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    kb_dir = os.path.join(here, "..", "data", "knowledge_base")
    out_path = os.path.join(here, "..", "profile_visibility_demo_results.md")
    user_id = "profile_visibility_demo"
    # 预置画像：跟当初真机测试时画像里的那条事实一样（文献只有简称ABC）
    SEED_PROFILE = {
        "称呼": "长安",
        "研究方向": "CRISPR条形码谱系追踪，导师推荐先读一篇相关经典文献（简称ABC），目前已读完",
        "偏好": "喜欢游泳，尤其是蛙泳",
    }

    llm_client = build_llm_client()
    log(f"当前LLM后端：{llm_client.name}（is_real={llm_client.is_real}）")
    if not llm_client.is_real:
        log("警告：LLM不可用——画像兜底(find_profile_answer())和生成模型引用"
            "画像的例外，都要在真实LLM可用的时候才会生效，先确认.env里的"
            "MINIMAX_API_KEY配好了。")
    log("")

    kb_pipeline = MemorySystemPipeline(kb_dir)
    store = MemoryStore(":memory:")  # 不碰 data/memory.db，见模块开头"数据库"一节
    for k, v in SEED_PROFILE.items():
        store.upsert_profile_fact(user_id, k, v)
    session = ConversationSession(kb_pipeline, store, user_id=user_id, llm_client=llm_client)

    profile = store.get_profile(user_id)
    log("=" * 70)
    log("预置的画像：")
    if profile:
        for k, v in profile.items():
            log(f"  {k}：{v}")
    else:
        log("  （空）")
    log("=" * 70)
    log("")

    # 每一项：(问题, 这一句想验证什么, 自动判定)
    script = [
        ("我导师推荐我读的是哪篇文献，我读完了吗",
         "第一次真机测试失败的原问法。重点看最终回答有没有正确提到画像里"
         "的文献信息，而不是像之前那样说'没有找到相关的历史记录'",
         lambda r: "ABC" in r and "读完" in r),
        ("导师推荐我读的ABC我读完了吗",
         "第二次真机测试的换一种问法。同样检查最终回答是否正确，而不是"
         "像之前那样答非所问（比如生成一个跟问题字面相关但实际没有回答"
         "具体问题的段落）",
         lambda r: "读完" in r),
        ("能天使是什么职业？",
         "回归检查：这条问题走本地知识库路径，完全不涉及画像兜底，"
         "行为应该跟改动前完全一致",
         lambda r: "狙击" in r),
    ]

    n_ok = 0
    for question, checkpoint, check in script:
        log("=" * 70)
        log(f"你说：{question}")
        log(f"【这句话要验证】{checkpoint}")
        log("-" * 70)
        reply = session.user_says(question)
        ok = check(reply)
        n_ok += ok
        log(reply)
        log(f"自动判定：{'✅' if ok else '❌'}")
        log("")

    log("=" * 70)
    log(f"自动判定 {n_ok}/{len(script)} 通过（只查关键词，回答内容仍建议人工看一眼）")
    log("整体重点检查：")
    log("1. 前两条问法不同的问题，最终回答是不是都正确用上了画像里的信息")
    log("   （不要求两次回答的具体来源路径/措辞完全一致，只要求内容正确、")
    log("   不再出现'没有找到相关的历史记录'或者答非所问）。")
    log("2. '能天使是什么职业'这条回归检查，回答是否跟改动前一致（本地")
    log("   知识库路径完全没有被这次改动影响到）。")
    log("3. 把这几条回答和你觉得有问题的地方发回来，一起判断这次要不要")
    log("   往前继续投入（比如`find_profile_answer()`判断质量不够、或者")
    log("   生成模型没有正确引用画像内容），还是可以确认这次修复生效了。")
    log("=" * 70)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# 画像可见性修复——端到端验证结果（人工核对）\n\n```\n")
        f.write("\n".join(_log_lines))
        f.write("\n```\n")
    print(f"\n完整输出已写到：{os.path.abspath(out_path)}")
