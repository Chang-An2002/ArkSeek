# -*- coding: utf-8 -*-
"""
prts.wiki Cargo结构化查询客户端 —— ④"索引/聚合/动态更新"类问题里"跨干员
多字段筛选"这一种子类型的正式实现（比如"六星狙击干员有哪些""有哪些五星
医疗干员"）。

【为什么不是在prts_wiki_client.py里加一个函数，而是新建一个文件】
prts_wiki_client.py整个文件的定位是"给定一个关键词，定位到wiki上的某一个
具体页面，把渲染后的HTML清洗成一段干净的正文"——不管是干员页面还是这次
新加的首页，走的都是`action=parse`这一个接口，产出都是"一段可以整段交给
LLM读的文本"。这次要做的事情从根上不是同一件事：不是"清洗一个页面"，是
"直接问wiki自己维护的结构化数据表要一份筛选结果"，用的是完全不同的接口
（`action=cargoquery`），产出也是结构化的行数据，不是一段文本——两者的
输入输出形状都不一样，硬塞进同一个文件、同一套函数签名里反而会让"这个
文件到底在做什么"变得含糊。只有最底层的HTTP请求函数`_call_api()`是两边
真正共用的部分，直接从prts_wiki_client.py导入复用，不重复实现一份。

【怎么确认这条路径可行的：四轮真机探测，不是拍脑袋设计】
1. 用户截图确认prts.wiki的"干员一览"页面本身是一个带17列属性、支持职业/
   稀有度/位置/性别/获取途径/词缀等多字段组合筛选的表格，这类"跨干员批量
   筛选"的问题真实存在、且用户自己平时就是这么用的。
2. 抓这个页面的原始HTML验证过——`action=parse`拿到的服务器渲染HTML里
   `<table>`标签出现次数是**0**，证明这张表格是页面加载完之后前端另外
   查出来再填进去的，"抓这个页面的HTML再解析表格"这条路从根上就走不通，
   不是清洗规则不够好的问题。
3. 转而探测prts.wiki是否启用Cargo扩展：`action=cargotables`确认启用，
   有`chara`（干员基础属性：`profession`职业/`rarity`稀有度/`cn`中文名等）/
   `chara_data`（战斗数值）/`chara_extra_info`（性别/出生地等档案字段，
   不少为空）等13张表。
4. 用`action=cargofields`摸清`chara`表的真实字段名和取值格式：
   `profession`存的是中文值（"先锋""狙击""医疗"……，共8个职业，见下面
   `VALID_PROFESSIONS`），`rarity`存的是**从0开始编号的整数**（1星=0、
   2星=1……6星=5，不是游戏里说的"六星"字面对应6，用真实数据验证过：
   `rarity=5`查到18个狙击干员，`rarity=6`查到0个）；另外Cargo的API有
   一条自己的语法限制——查询结果里想要页面标题，字段名`_pageName`不能
   直接用，必须显式起一个不带下划线开头的别名（写成`_pageName=PageName`），
   直接裸用会报错。用真实字段名+真实取值格式查`profession='狙击' AND
   rarity=5`，查到的18个干员名单跟用户截图里"职业=狙击+稀有度=★6"筛选
   结果（17个）几乎完全吻合，唯一差的1条是`Stormeye(卫戍协议)`——对照
   "干员一览"页面上"专属干员另见干员一览/专属干员页面"的提示，这类仅限
   特殊模式使用的干员被那个页面单独排除了，不是查询查错了。

【一个如实记录的局限：不追求跟"干员一览"页面的筛选结果完全一致】
上面第4点的差异（18 vs 17）来自"专属干员"这一类特殊标记，`chara`表的
`cargofields`列出的字段里没有一个明显对应"是否为专属干员"的布尔字段，
名字模式（页面标题带不带括号）也不是可靠的判断依据——`阿米娅(近卫)`这种
带括号的标题是正常可玩的职业分支，不是专属干员，无法只靠名字模式区分。
所以这里选择不做这层过滤，如实告诉用户查询结果可能跟"干员一览"页面的
默认筛选视图有1-2个的出入、以及具体原因，而不是假装完全精确或者用一条
基于名字模式的脆弱规则强行对齐——这跟`prts_wiki_client.py`一路秉持的
"如实记录局限，不用一条脆弱规则换取虚假的精确"是同一个原则。

【安全性：为什么职业/稀有度要做白名单校验，而不是直接把LLM给的值拼进
WHERE子句】
`_call_api()`把参数直接做URL编码发给Cargo API，`where`子句是用字符串
拼接构造的，不是参数化查询——如果直接把route_query()那次LLM调用解析出的
职业/稀有度原样拼进去，理论上存在被查询语法字符污染/搞出格式错误query的
风险（虽然prts.wiki是只读查询、后果有限，但没有验证过的输入不应该直接拼
进任何查询语句，这是基本的防御性编程习惯）。所以这里对职业做精确白名单
校验（必须是8个真实职业名之一）、对稀有度做范围校验（必须是1-6的整数），
校验不通过直接拒绝查询并返回None，不做任何"尽量猜一下修正"的宽容处理——
校验失败大概率意味着route_query()那次解析出了问题，让调用方看到None、
退回本地知识库，比带着一个不确定合法的值继续查询更诚实。
"""

from typing import Dict, List, Optional

from prts_wiki_client import _call_api

# prts.wiki干员职业的8个真实取值，跟Cargo `chara`表`profession`字段的
# 真实取值一一对应（真机验证过，不是猜的）——LLM解析出的职业名如果不在
# 这个集合里，说明解析有问题，直接拒绝查询。
VALID_PROFESSIONS = {"先锋", "近卫", "重装", "狙击", "术师", "医疗", "辅助", "特种"}


def _to_internal_rarity(rarity) -> Optional[int]:
    """人类说法的"N星"（1-6）转成Cargo `chara`表`rarity`字段真实存储的
    从0开始编号的整数（六星对应rarity=5，不是6）——见模块开头"真机验证"
    一节。传入不是1-6范围内的整数（包括类型不对/超出范围）一律返回None，
    调用方应该把这当成"稀有度参数不合法"处理，不要尝试做任何修正/取整。
    """
    try:
        n = int(rarity)
    except (TypeError, ValueError):
        return None
    if not 1 <= n <= 6:
        return None
    return n - 1


def query_operators(profession: Optional[str] = None, rarity: Optional[int] = None,
                     limit: int = 100) -> Optional[List[Dict[str, str]]]:
    """按职业/稀有度组合筛选查询干员，返回结构化的干员列表（不是一段文本，
    调用方自己决定怎么拼成给用户看的话）。

    profession/rarity 至少要给一个——不支持不带任何筛选条件的"查询全部
    干员"（既没有实际使用场景，也可能一次拉回几百条不必要的数据）。

    返回 None 表示查询本身没有成功（参数不合法/网络请求失败/Cargo返回
    了非预期的数据形状），调用方应该退回本地知识库或者诚实告知查询失败，
    不要当成"没有符合条件的干员"处理——这是两种不同的情况，前者是"这次
    没查成"，后者是"查成了、确实没有"，`query_operators()`返回空列表
    `[]`才对应后者。
    """
    if profession is None and rarity is None:
        print("[prts_cargo_client] query_operators()至少需要职业或稀有度其中一个"
              "筛选条件，拒绝执行不带任何筛选条件的整表查询。")
        return None

    conditions = []

    if profession is not None:
        if profession not in VALID_PROFESSIONS:
            print(f"[prts_cargo_client] 职业“{profession}”不是合法的职业名称"
                  f"（应该是{sorted(VALID_PROFESSIONS)}之一），拒绝查询。")
            return None
        conditions.append(f"profession='{profession}'")

    if rarity is not None:
        internal_rarity = _to_internal_rarity(rarity)
        if internal_rarity is None:
            print(f"[prts_cargo_client] 稀有度“{rarity}”不是合法的1-6星，拒绝查询。")
            return None
        conditions.append(f"rarity={internal_rarity}")

    where = " AND ".join(conditions)
    params = {
        "action": "cargoquery",
        "tables": "chara",
        # _pageName不能直接当字段名/隐式别名用（Cargo自己的语法限制，别名
        # 不能以下划线开头），必须显式起一个别名——见模块开头"真机验证"一节。
        "fields": "_pageName=PageName,cn,profession,rarity",
        "where": where,
        "limit": str(limit),
    }
    data = _call_api(params)
    if data is None:
        return None
    if "error" in data:
        print(f"[prts_cargo_client] Cargo查询返回错误：{data['error']}")
        return None

    try:
        rows = data["cargoquery"]
    except (KeyError, TypeError):
        print(f"[prts_cargo_client] Cargo查询返回的数据形状跟预期不一样：{data}")
        return None

    results: List[Dict[str, str]] = []
    for row in rows:
        # 真机验证过的实际返回形状是[{"title": {字段名: 值, ...}}, ...]——
        # 每一行的字段都包在"title"这个键下面，不是直接摊平在row本身。
        fields = row.get("title") if isinstance(row, dict) else None
        if not isinstance(fields, dict):
            continue
        name = (fields.get("PageName") or "").strip()
        if not name:
            continue
        rarity_raw = fields.get("rarity")
        try:
            rarity_display = str(int(rarity_raw) + 1)
        except (TypeError, ValueError):
            rarity_display = "?"
        results.append({
            "name": name,
            "profession": fields.get("profession", ""),
            "rarity_display": rarity_display,
        })
    return results


if __name__ == "__main__":
    print("【注意】这个自测必须在能直接访问 prts.wiki 的网络环境下运行"
          "（云端沙盒/远程设备通道连不上，需要在你自己电脑上跑）。\n")

    cases = [
        ("狙击", 6),
        ("医疗", 5),
        (None, 6),
        ("侦察", 6),  # 故意给一个不存在的职业名，验证白名单校验能拒绝
        ("狙击", 9),  # 故意给一个不合法的稀有度，验证范围校验能拒绝
    ]
    for profession, rarity in cases:
        print(f"查询 profession={profession!r}, rarity={rarity!r} ……")
        result = query_operators(profession=profession, rarity=rarity)
        if result is None:
            print("  查询失败/参数不合法（见上面的提示）。\n")
        else:
            print(f"  查到{len(result)}个干员：{[r['name'] for r in result]}\n")
