"""拒绝无信息的画像值，以及从泛泛问句推断出来的个人健康事实。"""

import re


def supported_profile_fact(key, value, source_text=""):
    normalized = re.sub(r"[\s。！.!（）()]", "", value).lower()
    if normalized in {"", "未提供", "未透露", "不知道", "未知", "不详", "暂无", "无", "none", "null", "unknown", "n/a"}:
        return False
    if "健康" in key or "病史" in key or "疾病" in key:
        question = re.search(r"[?？]|是不是|会不会|是否|为什么|怎么办|怎么|如何", source_text)
        personal = re.search(r"我(?:的|自己|本人|今年|最近|有|患|得|睡|血压|血糖|身体|一直|经常|这)|本人", source_text)
        if source_text and question and not personal:
            return False
    return True
