# -*- coding: utf-8 -*-
"""
ONNX Runtime 对照组：把精排模型导出成 ONNX，并做 int8 动态量化（2026-09-30，回复慢）

选 OpenVINO 时说过拿 ONNX Runtime 做对照（它是最通用的推理引擎，跨平台/跨芯片），这里补上。
没用 sentence-transformers 的 onnx 扩展：它会把 transformers 降到 4.x，跟 OpenVINO 那套版本
冲突。改用标准做法：torch.onnx.export 导出 → onnxruntime 直接跑；int8 用 onnxruntime 自带
的动态量化（quantize_dynamic，只量化权重、激活在运行时量化，不需要校准样本，是 Transformer
类模型在 CPU 上最常用的量化方式）。只多装 onnx / onnxscript 两个小包。

产出：models/ort_reranker/model.onnx（fp32）、model_int8.onnx（不提交，丢了重新运行即可）。
ORTReranker 实现了跟 CrossEncoder 一样的 predict(pairs, activation_fn=...) 接口，可以直接
替换 CrossEncoderReranker.model，给 run_openvino_rerank_eval.py 做对比用。

用法：cd src && ..\\.venv\\Scripts\\python.exe build_onnx_reranker.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "BAAI/bge-reranker-v2-m3"
OUT = os.path.join(HERE, "..", "models", "ort_reranker")
FP32, INT8 = os.path.join(OUT, "model.onnx"), os.path.join(OUT, "model_int8.onnx")


class ORTReranker:
    """onnxruntime 版精排：predict() 返回原始 logit（跟 CrossEncoder 传 Identity 激活时一样）。"""

    def __init__(self, onnx_path, max_length=8192, batch_size=32):
        import onnxruntime as ort
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.max_length, self.batch_size = max_length, batch_size
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.session.get_inputs()}

    def predict(self, pairs, **_ignored):
        out = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            enc = self.tokenizer([q for q, _ in batch], [t for _, t in batch], padding=True,
                                 truncation=True, max_length=self.max_length, return_tensors="np")
            feed = {k: v.astype(np.int64) for k, v in enc.items() if k in self.input_names}
            out.append(self.session.run(None, feed)[0].reshape(-1))
        return np.concatenate(out) if out else np.array([])


def main():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    os.makedirs(OUT, exist_ok=True)
    if not os.path.exists(FP32) or "--force" in sys.argv:
        print("导出 ONNX fp32……")
        tok = AutoTokenizer.from_pretrained(MODEL)
        # 用torch新的dynamo导出器（torch官方现在推荐的路径，需要onnxscript）：旧的TorchScript
        # 导出器跟transformers 5.x生成注意力掩码的代码不兼容（IndexError: tuple index out of
        # range，sdpa和eager实现都一样）。注意力用eager实现，数学上跟默认的sdpa等价
        model = AutoModelForSequenceClassification.from_pretrained(MODEL, attn_implementation="eager").eval()
        # 示例输入用两条、长度不同，导出器才不会把批次/长度当成常数
        enc = tok(["示例问题", "另一个稍微长一点的示例问题"], ["示例片段", "另一个片段"],
                  padding=True, return_tensors="pt")
        batch, seq = torch.export.Dim("batch"), torch.export.Dim("seq", max=8192)
        with torch.no_grad():
            torch.onnx.export(model, (enc["input_ids"], enc["attention_mask"]), FP32,
                              input_names=["input_ids", "attention_mask"], output_names=["logits"],
                              dynamic_shapes=({0: batch, 1: seq}, {0: batch, 1: seq}),
                              opset_version=18, dynamo=True, external_data=True)
    print("int8 动态量化……")
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic
    # 去掉导出时附带的中间张量形状标注（value_info，可选信息）：量化前的形状推断跟这些标注
    # 对不上会报 "Inferred shape and existing shape differ"
    clean = os.path.join(OUT, "model_for_quant.onnx")
    m = onnx.load(FP32)
    del m.graph.value_info[:]
    onnx.save(m, clean, save_as_external_data=True, location="model_for_quant.onnx.data")
    # 模型超过2GB，ONNX的权重存在外部数据文件里，量化时也要按外部数据格式读写
    quantize_dynamic(clean, INT8, weight_type=QuantType.QInt8, use_external_data_format=True)
    for f in (clean, clean + ".data"):
        os.remove(f)
    print(f"完成：\n  fp32 {os.path.abspath(FP32)}\n  int8 {os.path.abspath(INT8)}")


if __name__ == "__main__":
    main()
