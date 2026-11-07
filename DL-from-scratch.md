# DL From Scratch — ML × System

> 本文件由 AI 生成。声明详见 `README.md`。

浙大计算机 大二 · 目标：ML × System（高效推理 / 模型压缩）· 进组 → 美博 · 8 周 × 20h。有期末的周降级，宁拉长不中断。

**已有**：HPC101 全套 —— W8A8 MoE 向量化（87.36×）、TileLang GDN prefill（H800）、INT8 张量核（3461 GFLOPS）、Gemma4-12B GPTQ（ΔNLL=0.1018）、鲲鹏算子 58–115×。**缺**：ML 侧。
**目标**：从零写并训练 GPT → 加 KV cache → 量化到 4bit → 写 Triton kernel → 回答一个自己提的测量问题。

## 0. 方向

**默认主向：高效推理 + 模型压缩。** AI for Systems 前置 1–2 年（编译 / OS / DB），会场是 OSDI / SOSP / ATC，与主向的 MLSys / ICLR / NeurIPS / ASPLOS 不是同一批导师。

**现在不读论文定方向。** 看不懂的论文读三遍也不产生判断。三篇的位置本来就排好了：FlashAttention 在 Week 6、GPTQ 在 Week 5、Ansor 在「只读不做」。等认识 attention、跑过量化，再回头确认。

**复核点：Week 4 结束、Week 6 结束，各花 1 小时**，把结论写进 `notes/00-direction.md`。中间不重开这个问题。

## 1. 开工前（约 1 小时）

Week 1–3 纯 CPU。**买卡不急，但 Week 3 结束前必须定下来** —— Week 4 第一件事就是训练 GPT。

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # 前 3 周 CPU 版
pip install numpy matplotlib
pip freeze > requirements.txt
git init && printf '.venv/\n__pycache__/\n*.pt\n*.pth\ndata/\n.ipynb_checkpoints/\n*.log\n' > .gitignore
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

前 3 周输出 `False` 正常。triton 到 Week 6 再装，CPU 阶段没用。

**Week 3 结束前问清六件事**，写进 `notes/hardware.md`，每份报告引用：

- 型号与显存 → Week 6–7 序列长度上限
- 是否 MIG → 1g.10gb 卡住 Week 4 训练、Week 7 长序列
- 能否跑 `ncu` / `nsys` → 不能就全程 `torch.profiler`，Week 6 改方案
- 容器内是否 root → 装 Triton、编 CUDA 扩展
- CUDA 与驱动版本 → Triton 匹配
- 计费与预算 → Week 4 训练时长

结构：`README.md`（一张表，每实验一行数字）· `RESULTS.md`（实测数字唯一出处）· `perf/bench.py`（全局共用）· `notes/`（论文批判 + hardware.md）· `week1-mlp/` … `week8-report/` · `archive/`（过程产物，不放根目录）。

先写 `perf/bench.py` 再写模型：warmup ≥5 次、`torch.cuda.synchronize()` 后计时、min-of-N（N≥20）、中位数判优劣、最小值报性能、输出带 GPU 型号与 CUDA 版本与显存峰值。沿用 HPC101 铁律：per-run 配对计时、单变量演进、正确性门禁、Fast-but-wrong 无价值。

## 2. 八周

### Week 1 — PyTorch 变工具（10h，硬上限 15h）

**「PyTorch 学到手」的出口条件就是下面四条自测。** 全过才往下走，别把这一周拖成三周教程漫游。

材料：D2L 第 3 章；PyTorch Learn the Basics 按需查。
交付：`week1-mlp/` MNIST MLP，训练循环自己写，禁 Lightning / `Trainer`。
数字：`torch.profiler` 跑自己的循环，列出瓶颈段、matmul 占比、DataLoader 是否饿 GPU。
自测（出口条件）：四件套默写 / `zero_grad()` 为何必须 / `batch_size` 改动后显存与时间走向 / 哪段 compute-bound、哪段 memory-bound。

### Week 2 — 反向传播（20h）

材料：micrograd 视频 https://youtu.be/VMj-3S1tku0
交付：`week2-micrograd/` 合上视频重写 mini autograd，支持 `+ - * / tanh exp`。
数字 1：finite-difference gradcheck，每算子相对误差 < 1e-6，写成断言进 `test_engine.py`。
数字 2：同一两层 MLP，你的引擎 vs PyTorch autograd 前向反向耗时比值。
自测：手推两层 MLP 梯度与引擎对上 / 拓扑排序为何必须 / `exp` 的 backward 为何溢出 / 慢多少倍、慢在哪（Python 开销、无向量化、无融合）。
本周禁用 PyTorch autograd，只用纯 Python + numpy。

### Week 3 — 语言模型（20h）

材料：makemore 第 1 集 https://youtu.be/PaCmpygFfXo · 第 2 集 https://youtu.be/TCH_1BHY58I
交付：`week3-makemore/` 字符级名字生成器，bigram → MLP，含 train/val/test 与 loss 曲线。
数字：学习率扫描跨 3 个量级，画 val loss vs lr。
自测：cross-entropy 与 NLL 的关系 / embedding table 的 shape 与每维含义 / 过拟合拐点 / 最优点与直觉差多少。
**本周发第一封邮件**（第 5 节），带 `week2-micrograd/` 与 loss 曲线。
**本周结束前锁定 GPU**（第 1 节六问），别拖到 Week 4 第一天。

### Week 4 — attention 与 GPT（24h，最重）

材料：Let's build GPT https://www.youtube.com/watch?v=kCc8FmEb1nY · 参考 https://github.com/karpathy/nanoGPT。卡住才看：Illustrated Transformer、3Blue1Brown 两集。
数据：`mkdir -p week4-gpt/data && cd week4-gpt/data && curl -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`
交付：`week4-gpt/` multi-head、layer norm、residual、positional embedding 全手写，tiny shakespeare 训练到能生成像样文本。
数字 1：**自己写 KV cache**（缓存 K/V、增量 decode、位置对齐）。测 prefill 延迟、decode 每 token 延迟、显存 vs `block_size` / 层数 / head 数、与无 cache 的 O(n²) 重算的交叉点。
数字 2：**bf16 训练**，记录第一次溢出或发散在第几步，对比 fp32 的显存与 step 时间。
长训练：`sudo apt install -y tmux && tmux new -s train`（Ctrl+B 再按 D 脱离，`tmux attach -t train` 回来）。
自测：默写每个 tensor 的 shape / 为何除以 `sqrt(d_k)` / causal mask 为何上三角 / 改 `block_size` 哪些 tensor 变 / KV cache 把 decode 复杂度从什么降到什么、实测数字 / bf16 省多少显存快多少、误差从哪来。

### Week 5 — 量化自己的 GPT（20h）

材料：GPTQ 精读第一遍；AWQ、SmoothQuant、LLM.int8() 读摘要。
交付：`week5-quant/` W8A16 / W4A16 weight-only，自己写量化、反量化、group-wise 反量化后的 matmul。
数字：三点曲线 —— perplexity / decode 延迟 / 显存 vs bit-width（fp16、8bit、4bit）。
失败记录：哪一层对 4bit 最敏感。对照 HPC101 Lab5 的 ΔNLL=0.1018，差在哪、为什么。
自测：对称与非对称量化的公式与误差来源 / per-channel vs per-tensor 差多少 / group size 取舍 / weight-only 在 decode 阶段收益为何有限、瓶颈在哪。
禁用 `bitsandbytes`、`torchao`、`auto-gptq`、`awq`、`quanto`。

### Week 6 — Triton 与 roofline（20h）

材料：Triton 官方 tutorial（fused softmax、matmul）· FlashAttention 第二遍。
交付：`week6-kernel/` fused softmax 与 fused attention 各一版，扫序列长度对 naive PyTorch 测加速比，并与 `torch.compile` 的产出对打。
数字：每个 kernel 的 arithmetic intensity（FLOPs/bytes），画 roofline、标落点、判定带宽还是算力瓶颈；实测带宽占峰值百分比。
自测：为何 FlashAttention 是 IO-aware 而非 FLOPs 更少 / 你的 attention kernel 在短序列为何可能输给 naive / tiling block 怎么选、选错会怎样。

### Week 7 — 测量研究（20h）

题目：线性注意力与 softmax attention 的 prefill 交叉序列长度在哪，是否与 roofline 预测一致。
交付：`week7-study/` 纯 PyTorch 实现 softmax attention 与一个线性注意力（gated linear / RetNet-lite），扫 seq len 512 → 32k。**跑之前先算预测**：FLOPs 比值、KV / 状态内存比值、带宽比值 → 预测交叉点；再跑实测，画预测 vs 实测，解释偏差。关联 HPC101 Lab3 的 FlashQLA 数据。
自测：白板讲清假设、测量、结论、哪里可能不成立 / 偏差的第一解释是什么、怎么证伪它。

### Week 8 — 收口（20h）

- 每实验一个目录一个 README（第 4 节格式）
- `RESULTS.md`：每实验一行，数字 + 硬件 + 测量条件
- 两篇长文发 littlebanbrick.cn：①「写了个 GPT，量化到 4bit」三点曲线 + 失败记录 ②「交叉点在哪」Week 7 完整测量
- GitHub 整理（第 5 节）
- 第二封邮件，带数字
- 一页 research statement 草稿

自测：陌生人只看 README 能复现你的数字 / 教授随便挑一个数字问怎么测的，你能答。

## 3. 规则

1. **视频必须跟写。看完 24 小时内关视频重写一遍。** 比其余所有加起来重要。
2. **卡点分两类，处理相反。** 环境、依赖、报错、配置 → 卡 40 分钟就跳，记 TODO。概念、数学、shape 对不上 → 不跳，降维到最小可验证子问题（2 层、2 个 token、手算）。
3. 每个交付至少产出一个数字，进 `RESULTS.md`。
4. 每天 commit，信息写清改了什么、数字怎么变。
5. 主线只有 Karpathy 一条。系统侧不加教程主线，只查工具文档。
6. 测量一律走 `perf/bench.py`。导师信不信你的数字就看这条。
7. 前六周禁用 `nn.MultiheadAttention`、`nn.Transformer`、`transformers` 的 `DynamicCache` / `past_key_values`、`peft`、`bitsandbytes`、`torchao`。
8. 每周 2 小时保持 kernel 手感。

## 4. 模板

论文笔记（`notes/`，文件名用论文标题）：

```
标题：
问题：它解决什么问题
方法：核心做法，三句话以内
结果：关键数字
我的评价：它哪里可能不成立
可做的实验：我要接着做，第一个实验是什么   ← 最重要
```

实验 README：

```
复现了什么：
和原文差在哪：
我推测的原因：          ← 最重要
我额外试了什么：
硬件与测量条件：        ← 引用 notes/hardware.md
数字：                  ← 指向 RESULTS.md 对应行
```

## 5. 邮件与 GitHub

**Week 3 发第一封。** HPC101 仓库已够撑冷邮件；等八周的风险是名额被占、方向被改、白学三周。

第一封：我是谁（浙大计算机大二）+ 一句话现状（HPC101 + H800 TileLang kernel + Gemma-12B GPTQ 端到端）→ 读了你某篇工作的某一节，具体到一个问题，不泛夸 → 接下来六周做什么，产出公开在仓库 → 一个廉价请求（旁听组会 / 每周几小时的本科任务）→ 附两个**具体文件**直链，不只给首页，结尾不要求回复。
第二封（Week 8）：三点量化曲线 + 交叉点结论 + 「下一个想做的问题是 X」。

GitHub：pin `hpc101`、`hpc101-final-project`、`dl-from-scratch`（现在首页第一眼是 rEFInd 引导主题）。`hpc101` 根目录的 `_new_moe_opt.cpp`、`_probe_s4_src.cpp`、`_row4_baseline_revert.cpp`、`_rv5_full.sh`、`ab_t2d_novt_rerun.sh`、`_splice.py` 移进 `archive/`，每 Lab 顶层只留 `README.md` + `final/`。bio 加方向。`CLAUDE.md` 与 `memory/` 保留，但每个数字都要能白板讲解。

## 6. 不碰

强化学习、扩散模型、RAG、Agent 框架；分布式训练 / ZeRO / FSDP（Week 8 后）；CUDA C++ 手写 kernel（Triton 优先，非主线）；2026 新论文（Week 8 后）；同时看两个教程；全量读 PyTorch 源码（但 Week 4 后读一两个 ATen CUDA kernel 是推荐的）。

## 7. 论文（按序，每篇一节 notes）

1. Attention Is All You Need — 起点
2. FlashAttention / FlashAttention-2 — Week 6 核心
3. Roofline（Williams et al. 2009）— Week 6–7 判据
4. GPTQ — Week 5 精读
5. SmoothQuant · AWQ · LLM.int8() — Week 5 摘要
6. PagedAttention / vLLM — Week 4 后
7. 投机解码（Leviathan et al.）— 延迟优化第二条路
8. GDN / 线性注意力 — Week 7 对照

只读不做（每周 ≤1h）：TVM、Ansor、The Case for Learned Index Structures。

## 8. 资源

主线课程 https://karpathy.ai/zero-to-hero.html · micrograd https://github.com/karpathy/micrograd · nanoGPT https://github.com/karpathy/nanoGPT · 中文教材 https://zh.d2l.ai/ · Triton 教程 https://triton-lang.org/main/getting-started/tutorials/ · ML × 系统桥接 https://github.com/stanford-cs336/assignment2-systems · 高效 AI 计算（Song Han）https://hanlab.mit.edu/courses/2026-fall-65940 · Transformer 可视化 https://jalammar.github.io/illustrated-transformer/

Karpathy 视频是英文，开字幕跟；B 站有中文字幕搬运版。

## 9. Week 8 之后

分布式训练（CS336 assignment 2 的 parallelism 部分）；读 2026 新论文更新 `notes/`；若 Week 7 结论有意思，做到能投 MLSys 或 workshop 的程度。
