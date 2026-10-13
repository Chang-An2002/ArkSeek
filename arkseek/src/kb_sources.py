# -*- coding: utf-8 -*-
"""
知识库来源：Zotero文献目录 + JC材料 + 本地txt，统一转成带元数据的片段

【背景：知识库扩充（QQ群压力测试准备（三））】
用户要把JC讲过的内容和做课题参考过的文献加进知识库，并且能回答"作者读过哪些文献、
都讲了什么"。讨论时用户指出"文献目录"本质上就是Zotero/EndNote做的事——所以文献
目录以Zotero为唯一数据源（跟"SQLite是原件、Chroma是索引"同一个思路）：
- Zotero分类"本地知识库"：哪些文献、标题/作者/年份/期刊/DOI/摘要/标签/PDF附件。
  通过Zotero 7官方的本地接口读取（http://localhost:23119/api，需要在Zotero设置→
  高级里勾选"允许此计算机上的其他应用程序与Zotero通讯"，用户已勾选）。
- JC文件夹（D:\\Ruan Lab\\JC，每个日期子文件夹一次JC）：讲稿（Word/Markdown）、笔记
  脉络、PPT、补充图PDF。Zotero里没有的信息（"哪天讲过、讲稿/PPT是哪几份"）从这里补。
- data/knowledge_base/*.txt：原来的两份课题报告，照旧。

JC文件夹跟Zotero条目的对应关系自动匹配：优先用文件夹里PDF前两页里的DOI对Zotero
的DOI（精确），没有PDF的（25.11.29只有PPT）用PPT首页标题跟Zotero标题做词重合度。
匹配结果写进目录，build_catalog()会打印出来供核对。

【Zotero没开着怎么办】
机器人以后在QQ群里跑，不能要求Zotero一直开着。所以每次成功读到Zotero，就把目录
（含每篇PDF附件的本地路径）存一份到 data/kb_catalog.json；读不到时用这份缓存，并
打印醒目的降级提示（跟其它组件一样记进/status）。PDF文件本身在Zotero的storage
目录里，Zotero关着也读得到。

【每篇文献的中文简介】
"作者读过哪些文献、都讲了什么"是"列清单"类问题，不适合相似度检索（没法保证列全），
直接查目录回答。每篇的中文简介在建目录时用LLM根据标题+摘要生成一次，按DOI缓存在
kb_catalog.json里，以后不重复生成。
"""

import json
import os
import re
from typing import Dict, List, Optional
from urllib.parse import unquote, urlparse

from bot_profile import OWNER_NAME
from chunking import Chunk, chunk_by_sections, chunk_document
from doc_loaders import read_any

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
CONFIG_PATH = os.path.join(DATA_DIR, "kb_sources.json")
CATALOG_PATH = os.path.join(DATA_DIR, "kb_catalog.json")

DEFAULT_CONFIG = {
    "zotero_api": "http://localhost:23119/api/users/0",
    "zotero_collection": "本地知识库",
    "jc_root": r"D:\Ruan Lab\JC",
}


def load_config() -> Dict:
    cfg = dict(DEFAULT_CONFIG)
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg.update(json.load(f))
    return cfg


# ============================================================
# Zotero
# ============================================================

def _strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s or "").strip()


def fetch_zotero_papers(cfg: Dict) -> Optional[List[Dict]]:
    """读Zotero分类里的全部文献。读不到（Zotero没开/没勾选本地接口）返回None。"""
    import requests
    base = cfg["zotero_api"]
    try:
        cols = requests.get(f"{base}/collections", params={"limit": 100}, timeout=5).json()
    except Exception:
        return None
    if not isinstance(cols, list):
        return None
    col = next((c for c in cols if c["data"]["name"] == cfg["zotero_collection"]), None)
    if col is None:
        print(f"[kb_sources] Zotero里没有找到名为「{cfg['zotero_collection']}」的分类。")
        return []
    items = requests.get(f"{base}/collections/{col['key']}/items/top",
                         params={"limit": 100, "format": "json"}, timeout=10).json()
    papers = []
    for it in items:
        d = it["data"]
        if d.get("itemType") in ("attachment", "note"):
            continue
        pdf_path = None
        children = requests.get(f"{base}/items/{it['key']}/children", params={"format": "json"}, timeout=10).json()
        for ch in children:
            href = ch.get("links", {}).get("enclosure", {}).get("href", "")
            if ch["data"].get("contentType") == "application/pdf" and href.startswith("file:"):
                pdf_path = unquote(urlparse(href).path).lstrip("/")
                break
        creators = [c for c in d.get("creators", []) if c.get("creatorType") == "author"]
        year = (re.search(r"(19|20)\d{2}", d.get("date", "")) or [None])[0]
        papers.append({
            "key": it["key"],
            "title": _strip_html(d.get("title", "")),
            "authors": [f"{c.get('firstName', '')} {c.get('lastName', '')}".strip() or c.get("name", "")
                        for c in creators],
            "first_author": (creators[0].get("lastName") or creators[0].get("name", "")) if creators else "",
            "year": year,
            "journal": d.get("publicationTitle", ""),
            "doi": (d.get("DOI") or "").lower(),
            "abstract": _strip_html(d.get("abstractNote", "")),
            "tags": [t["tag"] for t in d.get("tags", [])],
            "pdf_path": pdf_path,
        })
    return papers


# ============================================================
# JC文件夹
# ============================================================

def _jc_date(folder: str) -> Optional[str]:
    """"25.7.5 JC" -> "2025-07-05"。"""
    m = re.match(r"(\d{2})\.(\d{1,2})\.(\d{1,2})", folder)
    if not m:
        return None
    return f"20{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"


def _classify(fname: str) -> Optional[str]:
    low = fname.lower()
    ext = low.rsplit(".", 1)[-1]
    if ext == "pptx":
        return "PPT"
    if ext in ("docx", "md"):
        if "讲稿" in fname:
            return "讲稿"
        if any(k in fname for k in ("笔记", "脉络", "解析")):
            return "笔记"
        return "笔记"
    if ext == "pdf":
        return "补充材料" if "supplement" in low else "文献PDF"
    return None  # 图片等暂不收录（用户确认过）


def scan_jc_folder(root: str) -> List[Dict]:
    sessions = []
    if not os.path.isdir(root):
        return sessions
    for folder in sorted(os.listdir(root)):
        path = os.path.join(root, folder)
        date = _jc_date(folder)
        if not os.path.isdir(path) or date is None:
            continue
        files = []
        for fname in sorted(os.listdir(path)):
            kind = _classify(fname)
            if kind:
                files.append({"kind": kind, "path": os.path.join(path, fname), "name": fname})
        sessions.append({"folder": folder, "date": date, "files": files})
    return sessions


_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"<>]+", re.I)


def _norm_words(s: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 2}


def match_session(session: Dict, papers: List[Dict]) -> Dict:
    """给一次JC找对应的Zotero文献。返回 {"key":..., "method":..., "detail":...}。"""
    by_doi = {p["doi"]: p for p in papers if p["doi"]}
    for f in session["files"]:
        if f["kind"] != "文献PDF":
            continue
        import pymupdf
        doc = pymupdf.open(f["path"])
        head = " ".join(doc[i].get_text() for i in range(min(2, doc.page_count)))
        for doi in _DOI_RE.findall(head):
            doi = doi.rstrip(".,;)").lower()
            if doi in by_doi:
                return {"key": by_doi[doi]["key"], "method": "DOI", "detail": doi}
    # 没有PDF/PDF里没找到DOI：用PPT首页标题跟Zotero标题比词重合度
    ppt = next((f for f in session["files"] if f["kind"] == "PPT"), None)
    if ppt:
        first_page = read_any(ppt["path"]).split("\n\n")[0]
        words = _norm_words(first_page)
        best, score = None, 0.0
        for p in papers:
            tw = _norm_words(p["title"])
            s = len(words & tw) / max(len(tw), 1)
            if s > score:
                best, score = p, s
        if best and score >= 0.6:
            return {"key": best["key"], "method": "PPT标题", "detail": f"重合度{score:.2f}"}
    return {"key": None, "method": "未匹配", "detail": ""}


# ============================================================
# 目录
# ============================================================

def _cn_date(iso: str) -> str:
    y, m, d = iso.split("-")
    return f"{y}年{int(m)}月{int(d)}日"


def _short_title(title: str, n: int = 40) -> str:
    return title if len(title) <= n else title[:n].rstrip() + "…"


def build_catalog(llm_client=None, verbose: bool = True) -> Optional[Dict]:
    """读Zotero+JC文件夹，建目录并存到kb_catalog.json；读不到Zotero时返回缓存。"""
    from chroma_index import announce_degrade

    cfg = load_config()
    cached = None
    if os.path.exists(CATALOG_PATH):
        with open(CATALOG_PATH, encoding="utf-8") as f:
            cached = json.load(f)

    papers = fetch_zotero_papers(cfg)
    if papers is None:
        if cached is not None:
            announce_degrade("文献目录", "读不到Zotero本地接口（Zotero没开着，或者没勾选允许其它应用通讯）",
                             f"沿用上次保存的目录（{cached.get('updated', '时间未知')}），Zotero里后来的改动不会反映出来")
            return cached
        announce_degrade("文献目录", "读不到Zotero本地接口，也没有缓存的目录", "知识库里只有本地txt，不含文献和JC材料")
        return None

    sessions = scan_jc_folder(cfg["jc_root"])
    for s in sessions:
        s["match"] = match_session(s, papers)

    old_summaries = {p["doi"]: p.get("summary", "") for p in (cached or {}).get("papers", [])}
    for p in papers:
        p["jc"] = [s["date"] for s in sessions if s["match"]["key"] == p["key"]]
        p["summary"] = old_summaries.get(p["doi"], "")
        if not p["summary"] and llm_client is not None and getattr(llm_client, "is_real", False):
            p["summary"] = _summarize(p, llm_client) or ""

    from datetime import datetime
    # 只记目录来源相关的配置；真名/学号这类脱敏清单不复制进目录缓存
    source_cfg = {k: cfg[k] for k in ("zotero_api", "zotero_collection", "jc_root") if k in cfg}
    catalog = {"updated": datetime.now().isoformat(timespec="seconds"), "config": source_cfg,
               "papers": papers, "jc_sessions": sessions}
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CATALOG_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=1)

    if verbose:
        key2p = {p["key"]: p for p in papers}
        print(f"[kb_sources] 文献目录已更新：Zotero「{cfg['zotero_collection']}」{len(papers)}篇，JC {len(sessions)}次。")
        for s in sessions:
            m = s["match"]
            t = _short_title(key2p[m["key"]]["title"]) if m["key"] else "（没匹配上）"
            print(f"    JC {s['date']} -> {t}  [{m['method']} {m['detail']}]")
    return catalog


_SUMMARY_PROMPT = """\
根据下面这篇文献的标题和摘要，用中文写2~3句简介：它解决什么问题、用了什么方法、
主要发现是什么。只根据给出的内容写，不要补充摘要里没有的信息。只输出简介本身。"""


def _summarize(p: Dict, client) -> Optional[str]:
    source = p["abstract"]
    if not source and p.get("pdf_path") and os.path.exists(p["pdf_path"]):
        # Zotero条目没抓到摘要（实测用户新加的两篇就是这样）：退而用PDF开头部分，
        # 论文的摘要/Summary就在第一页
        source = read_any(p["pdf_path"])[:3000]
    if not source:
        return None
    reply = client.chat([{"role": "system", "content": _SUMMARY_PROMPT},
                         {"role": "user", "content": f"标题：{p['title']}\n摘要（或论文开头部分）：{source}"}],
                        caller="kb_summarize()")
    return reply.strip() if reply else None


# ============================================================
# 目录 + 所有材料 -> 带元数据的片段
# ============================================================

def paper_label(p: Dict) -> str:
    return f"文献《{_short_title(p['title'])}》（{p['first_author']} {p['year']}，{p['journal']}）"


def _txt_label(fname: str) -> str:
    if "开题" in fname:
        return f"{OWNER_NAME}的课题开题报告"
    if "中期" in fname:
        return f"{OWNER_NAME}的课题中期进展报告"
    return fname


def iter_documents(kb_dir: str, catalog: Optional[Dict]):
    """产出 (doc_id, 展示标签, 文件路径或None, 文本或None, meta)。文本为None时由调用方读文件。"""
    import glob
    for path in sorted(glob.glob(os.path.join(kb_dir, "*.txt"))):
        fname = os.path.basename(path)
        yield (f"file:{fname}", _txt_label(fname), path, None,
               {"doc_type": "课题报告", "title": _txt_label(fname)})
    if not catalog:
        return
    papers = {p["key"]: p for p in catalog["papers"]}
    for p in catalog["papers"]:
        if p.get("pdf_path") and os.path.exists(p["pdf_path"]):
            yield (f"paper:{p['key']}", paper_label(p), p["pdf_path"], None,
                   {"doc_type": "文献原文", "paper_key": p["key"], "title": p["title"]})
        # 目录条目本身也作为一个小片段进索引：让"作者讲过哪篇关于DNA甲基化的文献"
        # 这类问题在相似度检索里也能命中
        jc = (f"；{OWNER_NAME}在" + "、".join(_cn_date(d) for d in p["jc"]) + "的JC上讲过这篇"
              if p.get("jc") else f"；{OWNER_NAME}读过但没有在JC上讲过")
        # "文献作者："显式标出来，跟后面的"长安在…讲过这篇"区分开（原来两处都叫"作者"）
        entry = (f"文献目录条目：《{p['title']}》，文献作者：{', '.join(p['authors'][:3])}{' 等' if len(p['authors']) > 3 else ''}，"
                 f"{p['journal']} {p['year']}{jc}。简介：{p.get('summary') or p['abstract'][:300]}")
        yield (f"catalog:{p['key']}", paper_label(p), None, entry,
               {"doc_type": "文献目录条目", "paper_key": p["key"], "title": p["title"]})
    for s in catalog["jc_sessions"]:
        p = papers.get(s["match"]["key"]) if s["match"]["key"] else None
        topic = _short_title(p["title"], 30) if p else s["folder"]
        for f in s["files"]:
            if f["kind"] == "文献PDF":
                continue  # 文献原文统一用Zotero里的那份，不重复收录
            kind_label = {"讲稿": "JC讲稿", "笔记": "JC笔记", "PPT": "JC PPT", "补充材料": "文献补充材料"}[f["kind"]]
            # 标签里同时写"2025年7月5日"和文件夹名"25.7.5"，两种说法问都能匹配上
            when = f"{_cn_date(s['date'])}（{s['folder'].split()[0]}）"
            label = f"{OWNER_NAME}{when} {kind_label}：{topic}"
            if f["kind"] == "笔记":
                label = f"{OWNER_NAME}{when} JC笔记《{os.path.splitext(f['name'])[0]}》：{topic}"
            yield (f"jc:{s['date']}:{f['name']}", label, f["path"], None,
                   {"doc_type": kind_label, "jc_date": s["date"], "paper_key": p["key"] if p else "",
                    "title": p["title"] if p else s["folder"]})


# 解析结果缓存：PDF按文本块清洗一遍要几秒，13篇文献每次启动都重来一遍是启动慢的
# 主要原因之一。按"路径+大小+修改时间+解析器版本"记一份解析好的纯文本，文件没变
# 就直接用；doc_loaders.PARSER_VERSION变了（清洗规则改了）则全部重新解析。
# 跟Chroma索引一样是派生数据，删掉只会让下次启动慢一点。
TEXT_CACHE_PATH = os.path.join(DATA_DIR, "kb_text_cache.json")


def _load_text_cache() -> Dict:
    try:
        with open(TEXT_CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _read_with_cache(path: str, cache: Dict) -> str:
    from doc_loaders import PARSER_VERSION
    st = os.stat(path)
    stamp = [st.st_size, int(st.st_mtime), PARSER_VERSION]
    hit = cache.get(path)
    if hit and hit.get("stamp") == stamp:
        return hit["text"]
    text = read_any(path)
    cache[path] = {"stamp": stamp, "text": text}
    return text


def redact_owner_names(text: str, real_names: List[str], remove_terms: List[str] = ()) -> str:
    """把资料里主人的真名换成昵称（入库时脱敏）。PPT首页的汇报人、报告页眉里都有
    真名，不换的话回答里会出现"由（拼音真名）主讲"——光靠prompt让LLM改称昵称不可靠，
    它不知道这个名字就是主人。真名清单放在不提交的 data/kb_sources.json 的
    "owner_real_names"里（不写进代码），不区分大小写；中文名直接替换。
    remove_terms：直接删掉的个人信息（比如报告页眉里的学号），同样放在
    data/kb_sources.json 的"redact_terms"里。"""
    for name in sorted(real_names, key=len, reverse=True):
        if name:
            text = re.sub(re.escape(name), OWNER_NAME, text, flags=re.IGNORECASE)
    for term in remove_terms:
        if term:
            text = text.replace(term, "")
    return text


def load_knowledge_sources(kb_dir: str, catalog: Optional[Dict], attach_questions: bool = True) -> List[Chunk]:
    """attach_questions：给课题报告片段写上缓存里"这段能回答的问题"（见kb_questions.py）；
    build_kb_questions.py 生成问题时要拿补问题之前的片段，传False。"""
    chunks: List[Chunk] = []
    cfg = load_config()
    real_names, remove_terms = cfg.get("owner_real_names", []), cfg.get("redact_terms", [])
    cache = _load_text_cache()
    before = json.dumps(cache, sort_keys=True)
    seen = set()
    for doc_id, label, path, text, meta in iter_documents(kb_dir, catalog):
        if text is None:
            seen.add(path)
            try:
                text = _read_with_cache(path, cache)
            except Exception as e:
                print(f"[kb_sources] 读取失败，跳过：{path}（{type(e).__name__}: {e}）")
                continue
        text = redact_owner_names(text, real_names, remove_terms)
        # 每个片段开头加上文档标签（contextual chunk header，Anthropic的contextual
        # retrieval是它的完整版）：日期、"讲稿/PPT"、文献标题这些信息只在文件名/
        # 目录里，不加的话"7月5日那次JC讲了什么"这种问题在片段正文里根本匹配不到
        if meta.get("doc_type") == "课题报告":
            # 课题报告有逐级编号：按章节切、片段不跨章节，标题写到章节路径（"长安的课题
            # 开题报告 › 2. 研究方案 › … › 2.2.3 试验方案"），见chunking.chunk_by_sections()
            chunks.extend(chunk_by_sections(text, label, {**meta, "doc_id": doc_id}, start_chunk_id=len(chunks)))
            continue
        for c in chunk_document(text, label, {**meta, "doc_id": doc_id}, start_chunk_id=len(chunks)):
            c.text = f"[{label}]\n{c.text}"
            chunks.append(c)
    cache = {k: v for k, v in cache.items() if k in seen}  # 已经不在知识库里的文件不留
    if json.dumps(cache, sort_keys=True) != before:
        try:
            with open(TEXT_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False)
        except OSError as e:
            print(f"[kb_sources] 解析缓存写入失败（不影响使用，只是下次启动还要重新解析）：{e}")
    if attach_questions:
        from kb_questions import attach_questions as _attach
        _attach(chunks)
    return chunks


def local_kb_description(catalog: Optional[Dict]) -> str:
    """给开场白/自我介绍用的一句话描述，按目录实际内容生成。"""
    base = f"{OWNER_NAME}的专业方向——单细胞谱系追踪"
    if not catalog:
        return f"{base}课题（开题报告、中期进展报告）"
    n_jc = len(catalog["jc_sessions"])
    n_p = len(catalog["papers"])
    return f"{base}：{OWNER_NAME}的课题开题/中期报告、{n_jc}次文献分享会（JC）的讲稿和PPT、读过的{n_p}篇相关文献"
