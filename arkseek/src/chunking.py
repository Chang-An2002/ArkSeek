# -*- coding: utf-8 -*-
"""
文本切分模块（Chunking）

对应方案文档 2.2 节提到的"文本切分"环节：把长文档切成适合检索的小片段。
这里用一个简单、透明的规则：按空行分段，再对过长的段落按字符数二次切分，
并保留少量重叠（overlap），避免关键信息被切在两个片段的边界上而检索不到。

真实项目里可以换成更讲究的切分策略（按句子边界、按标题层级等），
但对于验证"混合召回"这套核心逻辑来说，这个简单版本已经足够。
"""

import os
import glob
import re
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Chunk:
    chunk_id: int
    source_file: str   # 展示给用户看的来源标签（引用来源里显示的就是它）
    text: str
    # 知识库扩充后新增的元数据（doc_id/doc_type/paper_key/jc_date/lang/title…），
    # 见kb_sources.py。doc_id是一份文档的唯一标识，Chroma按它做增量更新；老代码
    # 创建的Chunk没有meta，一切照旧（Chroma退回用source_file当文档标识）。
    meta: Dict = field(default_factory=dict)


def _split_long_paragraph(paragraph: str, max_len: int = 200, overlap: int = 40) -> List[str]:
    """对超长段落做二次切分，保留一定重叠。"""
    paragraph = paragraph.strip()
    if len(paragraph) <= max_len:
        return [paragraph]

    pieces = []
    start = 0
    while start < len(paragraph):
        end = min(start + max_len, len(paragraph))
        pieces.append(paragraph[start:end])
        if end == len(paragraph):
            break
        start = end - overlap
    return pieces


def chunk_text(text: str, source_file: str, start_chunk_id: int = 0,
               max_len: int = 200, overlap: int = 40) -> List[Chunk]:
    """把任意一段文本切成 Chunk 列表——按空行分段，过长段落再按字符数二次
    切分保留重叠。这是 load_knowledge_base() 对每个本地文件做的事情，抽成
    公共函数只是为了内部不重复代码（`load_knowledge_base()`对每个文件调用
    一次），以后改切分策略也只用改一个地方。

    【曾经尝试过、后来放弃的一个用途】`prts_wiki_client.py`原本也想复用这个
    函数去切分实时查到的wiki页面正文，但真机验证发现"按字符数切分"这套
    假设连续散文的规则不适合按小标题组织的结构化wiki内容（同一个词在不同
    小节含义不同，切分后会丢失小节归属这个关键的消歧信息），最终改成不切分、
    整页交给LLM——完整过程见`prts_wiki_client.py`模块开头的"架构反思"。
    这个函数目前只被`load_knowledge_base()`使用，但作为一次真实的复用尝试
    和它没有成功的原因都值得记录，不代表这个函数本身设计得不好，只是
    "按字符数切分"这个具体策略本来就是针对连续散文调的，用错了场景。

    source_file 只是个标签，用来标注这些片段来自哪里，不要求真的是磁盘上
    的文件路径。start_chunk_id 让调用方可以把多段文本的chunk_id接续编号，
    不冲突。
    """
    chunks: List[Chunk] = []
    chunk_id = start_chunk_id

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    for para in paragraphs:
        for piece in _split_long_paragraph(para, max_len=max_len, overlap=overlap):
            if not piece:
                continue
            chunks.append(Chunk(chunk_id=chunk_id, source_file=source_file, text=piece))
            chunk_id += 1

    return chunks


def load_knowledge_base(kb_dir: str, max_len: int = 200, overlap: int = 40) -> List[Chunk]:
    """
    读取知识库目录下所有 .txt 文件，切分成 Chunk 列表。

    参数：
        kb_dir: 知识库文件夹路径（比如 data/knowledge_base）
        max_len: 单个片段的最大字符数
        overlap: 二次切分时相邻片段的重叠字符数
    """
    chunks: List[Chunk] = []

    txt_files = sorted(glob.glob(os.path.join(kb_dir, "*.txt")))
    for filepath in txt_files:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        chunks.extend(chunk_text(
            content, source_file=os.path.basename(filepath),
            start_chunk_id=len(chunks), max_len=max_len, overlap=overlap,
        ))

    return chunks


if __name__ == "__main__":
    # 简单自测：直接跑这个文件可以看到切分结果
    here = os.path.dirname(os.path.abspath(__file__))
    kb_dir = os.path.join(here, "..", "data", "knowledge_base")
    chunks = load_knowledge_base(kb_dir)
    print(f"共切出 {len(chunks)} 个片段，来自 {len(set(c.source_file for c in chunks))} 个文件：\n")
    for c in chunks:
        preview = c.text[:40].replace("\n", " ")
        print(f"[{c.chunk_id:02d}] ({c.source_file}) {preview}...")


# ============================================================
# 按token数切片（知识库扩充：加入英文文献之后）
# ============================================================
# 原来的chunk_text()按"字符数"切（200字符）——中文里200个字符信息量还行，英文里
# 200个字符只有三十来个词，一篇论文会被切成几千段碎片，每段都缺上下文。业界通行
# 做法是按token数切，这里用一个不依赖具体分词器的近似：一个中文字≈1个token，一个
# 英文单词≈1.3个token（bge-m3这类多语言模型的分词器大致如此）。在段落边界处切，
# 段落太长再按句子切，相邻片段保留一两句重叠，避免一句话被切成两半后两边都读不懂。

_CJK = re.compile(r"[\u4e00-\u9fff]")
_EN_WORD = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*")
_SENT_SPLIT = re.compile(r"(?<=[。！？；!?;])|(?<=\.)\s+(?=[A-Z(\[])")


def approx_tokens(text: str) -> int:
    return len(_CJK.findall(text)) + int(1.3 * len(_EN_WORD.findall(text))) + len(re.findall(r"\d+", text))


def detect_lang(text: str) -> str:
    cjk = len(_CJK.findall(text))
    words = len(_EN_WORD.findall(text))
    return "zh" if cjk >= words else "en"


def _sentences(paragraph: str) -> List[str]:
    parts = [x.strip() for x in _SENT_SPLIT.split(paragraph) if x and x.strip()]
    out = []
    for sent in parts:  # 极长的"句子"（表格、没有标点的列表）按字数硬切
        while approx_tokens(sent) > 400:
            cut = max(len(sent) // 2, 1)
            out.append(sent[:cut])
            sent = sent[cut:]
        out.append(sent)
    return out


def chunk_document(text: str, source_file: str, meta: Dict = None, start_chunk_id: int = 0,
                   max_tokens: int = 320, overlap_tokens: int = 50) -> List[Chunk]:
    """按段落累加到max_tokens左右切一段；单个段落超长就按句子切。相邻片段重叠
    overlap_tokens左右（按整句）。每个片段带上meta（外加检测出的语言lang）。"""
    meta = dict(meta or {})
    units: List[str] = []
    for para in (p.strip() for p in re.split(r"\n\s*\n", text)):
        if not para:
            continue
        units.extend([para] if approx_tokens(para) <= max_tokens else _sentences(para))

    chunks: List[Chunk] = []
    current: List[str] = []
    cur_tokens = 0

    def flush():
        body = "\n\n".join(current).strip()
        if body:
            chunks.append(Chunk(chunk_id=start_chunk_id + len(chunks), source_file=source_file, text=body,
                                meta={**meta, "lang": detect_lang(body)}))

    for u in units:
        t = approx_tokens(u)
        if current and cur_tokens + t > max_tokens:
            flush()
            # 重叠：把上一段末尾的几句带进下一段
            tail, tail_tokens = [], 0
            for sent in reversed(_sentences(current[-1])):
                if tail_tokens + approx_tokens(sent) > overlap_tokens:
                    break
                tail.insert(0, sent)
                tail_tokens += approx_tokens(sent)
            current, cur_tokens = ([" ".join(tail)] if tail else []), tail_tokens
        current.append(u)
        cur_tokens += t
    flush()
    return chunks


# ============================================================
# 按章节切片 + 章节路径标题（2026-09-29，同义说法检索改进）
# ============================================================
# 问"这个课题的实验设计"查不到：课题报告里写的是"技术路线/试验方案/研究方法"。而且
# 课题报告几乎没有空行，chunk_document() 按段落切时，片段会跨章节、中间的片段也不带
# 章节标题——"2.2.3 试验方案"下面的正文片段里一个"方案""方法"都没有，问法再对也匹配
# 不上（检索只看片段正文）。业界做法之一是"上下文片段标题"（contextual chunk headers）：
# 按文档结构先分章节、片段不跨章节，每个片段前面写上"文档 › 各级章节标题"，BM25、
# 向量和精排都能用上标题里的词。这里完全用代码按编号识别章节，不调LLM，结果固定。
# 只用于课题报告（有清楚的逐级编号）；JC讲稿、文献原文照旧用 chunk_document()。

_HEADING_PATTERNS = (
    (re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,3})\.?\s+([一-鿿（(].{0,58})$"), None),   # 1. / 2.1 / 2.2.3
    (re.compile(r"^([一二三四五六七八九十]{1,3})、\s*(\S.{0,40})$"), 1),                       # 一、
    (re.compile(r"^（([一二三四五六七八九十]{1,3})）\s*(\S.{0,40})$"), 2),                     # （一）
    (re.compile(r"^（(\d{1,2})）\s*([^。；，,]{1,30})$"), 4),                                  # （1）短标题
)


def _heading(line: str):
    """是章节标题就返回(层级, 标题原文)，否则None。排除正文句子（带句号）和很长的行。"""
    s = line.strip()
    if not s or len(s) > 62 or "。" in s:
        return None
    for rx, level in _HEADING_PATTERNS:
        m = rx.match(s)
        if m:
            return (m.group(1).count(".") + 1 if level is None else level), s
    return None


def chunk_by_sections(text: str, source_file: str, meta: Dict = None, start_chunk_id: int = 0,
                      max_tokens: int = 320, overlap_tokens: int = 50) -> List[Chunk]:
    """先按章节标题把全文分成若干节，每节各自用 chunk_document() 切，片段不跨章节；
    每个片段前面加一行"【文档 › 一级标题 › 二级标题 …】"。"""
    sections, stack, buf = [], [], []

    def close():
        body = "\n".join(buf).strip()
        if body:
            sections.append((" › ".join(t for _, t in stack), body))

    for line in text.split("\n"):
        h = _heading(line)
        if h:
            close()
            buf = []
            level, title = h
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            buf.append(line)
    close()

    chunks: List[Chunk] = []
    for path, body in sections:
        header = f"【{source_file}{' › ' + path if path else ''}】"
        for c in chunk_document(body, source_file, meta, start_chunk_id=start_chunk_id + len(chunks),
                                max_tokens=max_tokens, overlap_tokens=overlap_tokens):
            c.text = f"{header}\n{c.text}"
            chunks.append(c)
    return chunks
