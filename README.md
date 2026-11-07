# DL From Scratch

从零手写神经网络到 GPT，再接到推理优化：手写 autograd、GPT、KV cache、量化、Triton kernel。

浙大计算机本科阶段自学计划，8 周 × 20h。计划与自测见 [DL-from-scratch.md](DL-from-scratch.md)。

所有实测数字的唯一出处：[RESULTS.md](RESULTS.md)。测量一律走 [perf/bench.py](perf/bench.py)。

## 进度

| 周 | 主题 | 状态 | 目录 |
|---|---|---|---|
| 1 | PyTorch 变工具 | 完成 | [week1-mlp/](week1-mlp/) |
| 2 | 反向传播 | 完成 | [week2/micrograd/](week2/micrograd/) |
| 3 | 语言模型 | 未开始 | |
| 4 | attention 与 GPT | 未开始 | |
| 5 | 量化自己的 GPT | 未开始 | |
| 6 | Triton 与 roofline | 未开始 | |
| 7 | 测量研究 | 未开始 | |
| 8 | 收口 | 未开始 | |

## 文档来源声明

**本仓库的 Markdown 文档由 AI 生成。**

具体来说，下列文件全部或绝大部分是 AI 撰写的，不是本人手写：

- 本文件 `README.md`
- `RESULTS.md` 及其所有表格
- `week1-mlp/README.md`
- `week2/micrograd/README.md`
- `DL-from-scratch.md`（学习计划）

## 环境

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

CPU：`torch 2.14.1+cpu` · `torchvision 0.29.2+cpu` · Python 3.13.13。Week 1–3 不需要 GPU。
