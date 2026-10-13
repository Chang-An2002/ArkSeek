# -*- coding: utf-8 -*-
"""
把精排模型导出成 OpenVINO 格式，并做 int8 静态量化（2026-09-30，回复慢）

背景：精排（bge-reranker-v2-m3，5.6亿参数）在CPU上每条候选约0.5秒，是本地检索最慢的
一步（见run_latency_profile.py / run_rerank_speed_eval.py）。CPU上部署模型的常见做法是
换专门的推理引擎（本机是Intel Core Ultra，用Intel的OpenVINO）+ int8量化。

产出（放在本机目录，不在坚果云里，跟Chroma索引一样）：
  <ARKSEEK_DIR>/ov_reranker/fp32/   只换引擎、不量化
  <ARKSEEK_DIR>/ov_reranker/int8/   再做int8静态量化

量化需要一批"校准样本"来确定每一层数值的范围。sentence-transformers 自带的函数用的是
英文影评数据集（glue/sst2），跟我们"中文问题+中英文资料片段"的分布差得远，所以这里
自己造：用离线生成的"可能被问的问题"（data/kb_question_cache.json，跟所有测试题
都不重合）去检索知识库，取召回到的候选组成"问题-片段"对——跟真实精排看到的输入同分布，
既有相关的也有不相关的。

依赖：openvino / optimum-intel / nncf / datasets（在requirements.txt里），用项目的
.venv运行：cd src && ..\\.venv\\Scripts\\python.exe build_openvino_reranker.py
模型存到项目的 models/ov_reranker/（.gitignore排除，丢了重新运行本脚本即可）。
"""

import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "BAAI/bge-reranker-v2-m3"
N_CALIBRATION = 300


def ov_dir():
    base = os.path.join(HERE, "..", "models", "ov_reranker")
    return os.path.join(base, "fp32"), os.path.join(base, "int8")


def calibration_pairs(n=N_CALIBRATION, seed=0):
    from pipeline import MemorySystemPipeline
    cache = json.load(open(os.path.join(HERE, "..", "data", "kb_question_cache.json"), encoding="utf-8"))
    questions = [q for qs in cache.values() for q in qs]
    # 再加一些文献/JC类的问题：用每个非课题报告片段的开头当"问题"，覆盖英文和文献内容
    kb = MemorySystemPipeline(os.path.join(HERE, "..", "data", "knowledge_base"))
    rng = random.Random(seed)
    others = [c for c in kb.chunks if c.meta.get("doc_type") != "课题报告"]
    questions += [c.text.split("\n")[-1][:60] for c in rng.sample(others, min(80, len(others)))]
    rng.shuffle(questions)
    pairs = []
    for q in questions:
        cands = kb.retrieve(q, top_n=20)  # 只要召回的候选，分数不用
        for p in rng.sample(cands, min(3, len(cands))):
            pairs.append((q, p.chunk.text))
        if len(pairs) >= n:
            break
    return pairs[:n]


def main():
    from sentence_transformers import CrossEncoder
    from datasets import Dataset
    from optimum.intel.openvino import OVConfig, OVQuantizationConfig, OVQuantizer

    fp32_dir, int8_dir = ov_dir()
    if not os.path.exists(os.path.join(fp32_dir, "openvino", "openvino_model.xml")) or "--force" in sys.argv:
        print("导出 OpenVINO fp32 模型……")
        model = CrossEncoder(MODEL, backend="openvino")
        model.save_pretrained(fp32_dir)
    model = CrossEncoder(fp32_dir, backend="openvino")

    print("准备校准样本……")
    pairs = calibration_pairs()
    enc = model.tokenizer([q for q, _ in pairs], [t for _, t in pairs], padding="max_length",
                          max_length=512, truncation=True)
    calib = Dataset.from_dict(dict(enc))

    print(f"int8 静态量化（{len(pairs)} 个校准样本）……")
    quantizer = OVQuantizer.from_pretrained(model.transformers_model)
    # quantize()自己会在save_directory下建openvino/子目录，CrossEncoder加载时也从那里找
    quantizer.quantize(calib, save_directory=int8_dir,
                       ov_config=OVConfig(quantization_config=OVQuantizationConfig(num_samples=len(pairs))))
    model.tokenizer.save_pretrained(int8_dir)
    # CrossEncoder加载需要的配置文件跟fp32版一样，复制过去
    import shutil
    for name in os.listdir(fp32_dir):
        src = os.path.join(fp32_dir, name)
        if os.path.isfile(src) and not os.path.exists(os.path.join(int8_dir, name)):
            shutil.copy(src, int8_dir)
    print(f"完成：\n  fp32 {fp32_dir}\n  int8 {int8_dir}")


if __name__ == "__main__":
    main()
