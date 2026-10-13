# -*- coding: utf-8 -*-
"""
精排换 OpenVINO / int8 后的速度和准确率对比（2026-09-30，回复慢）

题目、判定跟 run_rerank_speed_eval.py 完全一样（60题，只测检索），检索做法固定用
现在的正式配置（同义说法合并精排、候选池20条），只换精排模型的运行方式：

  PyTorch      ：现在的做法（基准）
  OV-fp32-CPU  ：OpenVINO 引擎，不量化，跑在CPU
  OV-int8-CPU  ：OpenVINO + int8 静态量化，CPU
  OV-fp32-GPU / OV-int8-GPU：同上，跑在 Intel Arc 核显
  ORT-fp32-CPU / ORT-int8-CPU：ONNX Runtime 对照组（2026-09-30补做，int8是动态量化）

模型先用 build_openvino_reranker.py / build_onnx_reranker.py 导出。用项目的.venv运行：
  cd src && ..\\.venv\\Scripts\\python.exe run_openvino_rerank_eval.py [变体名 ...] [--tag 名字]
只列变体名时只跑这几个（PyTorch 基准总会跑）。报告写到 ../openvino_rerank_eval_results[_名字].md
"""

import os
import sys
import time
from datetime import datetime

from concurrency import LockedCalls
from pipeline import MemorySystemPipeline
from build_openvino_reranker import MODEL, ov_dir
import run_rerank_speed_eval as ev

HERE = os.path.dirname(os.path.abspath(__file__))
TAG = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else ""
REPORT = os.path.join(HERE, "..", f"openvino_rerank_eval_results{'_' + TAG if TAG else ''}.md")
FP32, INT8 = ov_dir()
VARIANTS = [  # (名称, 加载参数；"ort"键表示ONNX Runtime模型文件)
    ("PyTorch", dict(model_name_or_path=MODEL)),
    ("OV-fp32-CPU", dict(model_name_or_path=FP32, backend="openvino", model_kwargs={"device": "CPU"})),
    ("OV-int8-CPU", dict(model_name_or_path=INT8, backend="openvino", model_kwargs={"device": "CPU"})),
    ("OV-fp32-GPU", dict(model_name_or_path=FP32, backend="openvino", model_kwargs={"device": "GPU"})),
    ("OV-int8-GPU", dict(model_name_or_path=INT8, backend="openvino", model_kwargs={"device": "GPU"})),
    ("ORT-fp32-CPU", dict(ort="fp32")),
    ("ORT-int8-CPU", dict(ort="int8")),
]


def _load(kwargs):
    if "ort" in kwargs:
        import build_onnx_reranker as ob
        return ob.ORTReranker(ob.FP32 if kwargs["ort"] == "fp32" else ob.INT8)
    from sentence_transformers import CrossEncoder
    return CrossEncoder(**kwargs)


def main():
    args = [a for a in sys.argv[1:] if a != TAG and a != "--tag"]
    variants = [v for v in VARIANTS if not args or v[0] in args or v[0] == "PyTorch"]
    kb = MemorySystemPipeline(os.path.join(HERE, "..", "data", "knowledge_base"))
    results, load_s = {}, {}
    for name, kwargs in variants:
        print(f"== {name} ==")
        t0 = time.perf_counter()
        kb.reranker.model = LockedCalls(_load(kwargs), {"predict"})
        kb.retrieve("预热", top_n=3)  # 核显第一次要编译，预热不计时
        load_s[name] = time.perf_counter() - t0
        results[name] = ev.run_config(kb, True, 10, 20)
    base = results["PyTorch"]
    Q = ev.QUESTIONS
    expect_idx = [i for i, x in enumerate(Q) if x[4] is not None]
    hit_idx = [i for i, x in enumerate(Q) if x[3] is not None]
    lines = [f"# 精排推理方式对比：速度和准确率{'（' + TAG + '）' if TAG else ''}", "", f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "",
             f"{len(Q)}道题（同 run_rerank_speed_eval.py），只测检索；检索做法固定（同义说法合并精排、候选池20条），"
             f"只换精排模型的运行方式。门槛：精排分≥{kb.RERANK_FLOOR}。", "",
             "## 汇总", "",
             "| 做法 | 过门槛 | 符合期望 | 前3条找对 | 过门槛判定跟PyTorch一样 | 前3条跟PyTorch一样 | 最高分最大偏差 | 精排总耗时 | 加载+预热 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, _ in variants:
        r = results[name]
        dev = max(abs(r[i]["top"] - base[i]["top"]) for i in range(len(r)))
        lines.append(
            f"| {name} | {sum(x['passed'] for x in r)}/{len(r)} | "
            f"{sum(r[i]['passed'] == Q[i][4] for i in expect_idx)}/{len(expect_idx)} | "
            f"{sum(bool(r[i]['hit']) for i in hit_idx)}/{len(hit_idx)} | "
            f"{sum(r[i]['passed'] == base[i]['passed'] for i in range(len(r)))}/{len(r)} | "
            f"{sum(set(r[i]['ids']) == set(base[i]['ids']) for i in range(len(r)))}/{len(r)} | "
            f"{dev:.3f} | {sum(x['rr'] for x in r):.0f}s | {load_s[name]:.0f}s |")
    lines += ["", "## 逐题（过门槛 最高分 找对 / 前3条同PyTorch / 精排秒数）", "",
              "| 组 | 问题 | " + " | ".join(n for n, _ in variants) + " |", "|---|---|" + "---|" * len(variants)]
    for i, (grp, q, *_rest) in enumerate(Q):
        cells = []
        for name, _ in variants:
            x = results[name][i]
            hit = "" if x["hit"] is None else ("找对" if x["hit"] else "没找对")
            same = "" if name == "PyTorch" else ("同" if set(x["ids"]) == set(base[i]["ids"]) else "变")
            cells.append(f"{'✅' if x['passed'] else '❌'} {x['top']:.3f} {hit} {same} {x['rr']:.1f}s".strip())
        lines.append(f"| {grp} | {q} | " + " | ".join(cells) + " |")
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[8:10 + len(variants)]))
    print(f"报告：{os.path.abspath(REPORT)}")


if __name__ == "__main__":
    main()
