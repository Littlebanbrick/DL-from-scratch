# DL From Scratch

从零手写神经网络到 GPT，再接到推理优化：手写 autograd、GPT、KV cache、量化、Triton kernel。

计划与自测见 [DL-from-scratch.md](DL-from-scratch.md)。

所有实测数字的唯一出处：[RESULTS.md](RESULTS.md)。测量一律走 [perf/bench.py](perf/bench.py)。

## 进度

| 周 | 主题 | 状态 | 目录 |
|---|---|---|---|
| 1 | PyTorch 变工具 | 完成 | [week1-mlp/](week1-mlp/) |
| 2 | 反向传播 | 完成 | [week2/micrograd/](week2/micrograd/) |
| 3 | 语言模型 | 完成 | [week3/makemore/](week3/makemore/) |
| 4 | attention 与 GPT | 完成（bf16 待 GPU） | [week4/gpt/](week4/gpt/) |
| 5 | 量化自己的 GPT | 完成（速度待 GPU） | [week5/quant/](week5/quant/) |
| 6 | Triton 与 roofline | 需 GPU | |
| 7 | 测量研究 | 未开始 | |
| 8 | 收口 | 未开始 | |

## 文档来源声明

**本仓库的 Markdown 文档由 AI 生成。**

具体来说，下列文件全部或绝大部分是 AI 撰写的，不是本人手写：

- 本文件 `README.md`
- `RESULTS.md` 及其所有表格
- `week1-mlp/README.md`
- `week2/micrograd/README.md`
- `week3/makemore/README.md`
- `week5/quant/README.md`
- `DL-from-scratch.md`（学习计划）

## 环境

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

CPU：`torch 2.14.1+cpu` · `torchvision 0.29.2+cpu` · Python 3.13.13。

Week 1–5 全部在 CPU 上完成。Week 6（Triton kernel）必须有 GPU。
