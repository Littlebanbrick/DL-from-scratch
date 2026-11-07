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

AI 在这些文档里做的事：整理实测数字、写表格、组织叙述、检查数字自洽、指出测量方法中的问题。**文档里的每一个数字都来自本人在本机跑出来的输出，没有一个是编造的**；每个数字的原始来源和测量条件记录在 `RESULTS.md`。

**代码的情况不同**，逐文件说明：

| 文件 | 来源 |
|---|---|
| `perf/bench.py` | AI 撰写 |
| `week1-mlp/train.py` | 本人写骨架，AI 增补 loss 曲线、计时、profiler、bench 模式 |
| `week1-mlp/bench_demo.py` | AI 撰写 |
| `week2/micrograd/value.py` | **本人撰写**（AI 只指出过一个 bug 并解释原理） |
| `week2/micrograd/nn.py` | 参照 Karpathy micrograd 的结构 |
| `week2/micrograd/train.py` | AI 撰写 |
| `week2/micrograd/test.py` | 本人抄录 Karpathy 的两个对拍测试，AI 增补有限差分 gradcheck |
| `week2/micrograd/compete.py` | 本人撰写，AI 修正 GC 与公平性问题 |

逐条数字、每处改动都经过本人 review。如果某处描述与本人的理解不符，以本人为准。

## 环境

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

CPU：`torch 2.14.1+cpu` · `torchvision 0.29.2+cpu` · Python 3.13.13。Week 1–3 不需要 GPU。
