# -*- coding: utf-8 -*-
"""
联网搜索客户端（QQ群压力测试准备（二））

【为什么要联网搜索】
用户计划把机器人放进QQ群给朋友测试，朋友会问很多wiki/本地知识库查不到的
问题（"哪个干员最强""推荐个攻略up主"）。用户的决策：查不到时不要一句"不知道"
把天聊死，而是先联网搜索、再退到大模型自身知识，每一级都明确标注来源。
"推荐up主"这类时效性内容只靠模型训练数据会过时，所以要真的联网。

【后端：MiniMax Token Plan 自带的搜索接口】
用户的MiniMax是Token Plan月度会员，Token Plan的key可以直接调用
POST https://api.minimaxi.com/v1/coding_plan/search（请求体 {"q": 问题}，
返回 organic 列表：title/link/snippet/date），2026-09-24实测可用，不用另外
申请搜索服务。两点如实记录、需要用户留意：
- 搜索是否/如何占用套餐额度，官方文档没写——每次调用都追加一条记录到
  data/web_search_log.jsonl（时间+问题+结果条数），方便用户跟控制台用量对账。
  这个文件含群友的提问，已加入.gitignore。
- 接口路径叫 coding_plan/search，本意是给编程工具用的，个人非商业的群聊测试
  用它是否符合条款，需要用户自己看一眼Token Plan服务条款。

跟 llm_client.py 同一个"可插拔后端"思路：build_web_search() 按环境变量
WEB_SEARCH_PROVIDER 选后端（默认minimax，设成none关闭）；以后要换博查等
其它服务，只加一个实现 .search() 的类。没配好/调用失败时 .search() 返回
None，调用方退到"大模型自身知识"那一级——跟其它组件一样，降级时打印醒目
提示、记进 /status（chroma_index.announce_degrade）。
"""

import html
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
SEARCH_LOG = os.path.join(HERE, "..", "data", "web_search_log.jsonl")

try:
    from dotenv import load_dotenv  # 跟 llm_client.py 一样，装了就从 .env 读
    load_dotenv(os.path.join(HERE, "..", ".env"))
except ImportError:
    pass



_TAG_RE = re.compile(r"<[^>]+>")


def _clean(text: str) -> str:
    """搜索接口偶尔把网页标题原样带着HTML标签返回（2026-10-01真机：萌娘百科的
    `<span class="mw-page-title-main">玛恩纳</span>`），去掉标签、还原实体。"""
    return html.unescape(_TAG_RE.sub("", text or "")).strip()

class NoWebSearch:
    is_real = False
    name = "未启用联网搜索"

    def search(self, query: str, max_results: int = 5) -> Optional[List[Dict]]:
        return None


class MiniMaxWebSearch:
    """MiniMax Token Plan 的搜索接口。请求格式参照 MiniMax 官方
    MiniMax-Coding-Plan-MCP 的实现（POST /v1/coding_plan/search，body {"q":...}，
    Bearer鉴权，响应里 base_resp.status_code==0 表示成功）。"""

    is_real = True
    URL = "https://api.minimaxi.com/v1/coding_plan/search"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.name = "MiniMax Token Plan 联网搜索"
        self.calls = 0  # 本进程累计调用次数，/status 里展示

    def search(self, query: str, max_results: int = 5) -> Optional[List[Dict]]:
        import requests

        self.calls += 1
        results, error = None, ""
        try:
            from timing import span
            with span("联网搜索"):
                resp = requests.post(self.URL, json={"q": query}, timeout=30,
                                     headers={"Authorization": f"Bearer {self.api_key}",
                                              "Content-Type": "application/json"})
            data = resp.json()
            status = (data.get("base_resp") or {}).get("status_code")
            if resp.status_code != 200 or status != 0:
                error = f"HTTP {resp.status_code}, base_resp={data.get('base_resp')}"
            else:
                results = [
                    {"title": _clean(r.get("title", "")), "link": r.get("link", ""),
                     "snippet": _clean(r.get("snippet", "")), "date": r.get("date", "")}
                    for r in (data.get("organic") or [])[:max_results]
                    if r.get("snippet") or r.get("title")
                ]
        except Exception as e:  # 网络/JSON问题一律当这次搜索失败
            error = f"{type(e).__name__}: {e}"

        self._log(query, results, error)
        if error:
            print(f"[web_search] 联网搜索失败（{error}），这一次退到大模型自身知识回答。")
            return None
        print(f"[web_search] 第{self.calls}次联网搜索（本次运行累计）：{query!r} -> {len(results)}条结果")
        return results

    @staticmethod
    def _log(query: str, results, error: str) -> None:
        try:
            os.makedirs(os.path.dirname(SEARCH_LOG), exist_ok=True)
            with open(SEARCH_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), "query": query,
                                    "results": len(results) if results is not None else None,
                                    "error": error}, ensure_ascii=False) + "\n")
        except OSError:
            pass  # 记日志失败不影响搜索本身


def build_web_search():
    from chroma_index import announce_degrade

    provider = os.environ.get("WEB_SEARCH_PROVIDER", "minimax").lower()
    if provider == "none":
        print("[web_search] WEB_SEARCH_PROVIDER=none，已关闭联网搜索。")
        return NoWebSearch()
    if provider == "minimax":
        key = os.environ.get("MINIMAX_API_KEY", "").strip()
        if key:
            client = MiniMaxWebSearch(key)
            print(f"[web_search] 已配置：{client.name}")
            return client
        announce_degrade("联网搜索", "没有配置MINIMAX_API_KEY",
                         "查不到时直接退到大模型自身知识回答（照样会标注来源）")
        return NoWebSearch()
    announce_degrade("联网搜索", f"不认识的WEB_SEARCH_PROVIDER={provider!r}",
                     "查不到时直接退到大模型自身知识回答（照样会标注来源）")
    return NoWebSearch()
