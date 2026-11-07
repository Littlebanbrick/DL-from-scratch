"""M6: 引擎的正确性门禁。

两部分，互相独立：

  1. 跟 PyTorch 对拍   —— 强判据，但只在 torch 能装的地方跑
  2. 有限差分 gradcheck —— 不依赖 torch，逐个算子验证导数公式

跑法：
    python test.py
全部通过返回 0，有失败返回 1。
"""

import torch

from value import Value


# ============================================================
# 第一部分：跟 PyTorch 对拍
# ============================================================

def test_sanity_check():
    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    assert ymg.data == ypt.data.item(), "forward mismatch"
    assert xmg.grad == xpt.grad.item(), "backward mismatch"


def test_more_ops():
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    amg, bmg, gmg = a, b, g

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    assert abs(gmg.data - gpt.data.item()) < tol, "forward mismatch"
    assert abs(amg.grad - apt.grad.item()) < tol, "grad a mismatch"
    assert abs(bmg.grad - bpt.grad.item()) < tol, "grad b mismatch"


# ============================================================
# 第二部分：有限差分 gradcheck
#
# 数值梯度 = (f(x+eps) - f(x-eps)) / (2*eps)   中心差分，比前向差分误差小一个量级
# 解析梯度 = backward() 给出的值
# 相对误差 <= tol 才算过
# ============================================================

def _numeric_grad(fn, inputs, i, eps):
    """只扰动第 i 个输入，且每次都新建一张图。

    复用同一批 Value 会让梯度累加，测出来的东西是假的。
    """
    hi = list(inputs)
    hi[i] += eps
    lo = list(inputs)
    lo[i] -= eps
    f_hi = fn(*[Value(t) for t in hi]).data
    f_lo = fn(*[Value(t) for t in lo]).data
    return (f_hi - f_lo) / (2 * eps)


def check(name, fn, *inputs, eps=1e-6, tol=1e-6):
    """对 fn 的每个输入，比对解析梯度与数值梯度。"""
    clean = [float(t) for t in inputs]

    vals = [Value(t) for t in clean]
    out = fn(*vals)
    out.backward()

    for i, v in enumerate(vals):
        num = _numeric_grad(fn, clean, i, eps)
        ana = v.grad
        denom = max(1.0, abs(num), abs(ana))
        rel = abs(ana - num) / denom
        assert rel <= tol, (
            f"{name}: input[{i}] analytic {ana:.9f} vs numeric {num:.9f} (rel {rel:.2e})"
        )


# ---------- 单算子 ----------

def test_grad_add():
    check("add", lambda a, b: a + b, 1.5, -2.5)


def test_grad_sub():
    check("sub", lambda a, b: a - b, 1.5, -2.5)


def test_grad_mul():
    check("mul", lambda a, b: a * b, 1.5, -2.5)


def test_grad_div():
    check("div", lambda a, b: a / b, 3.0, 2.5)


def test_grad_pow():
    check("pow", lambda a: a**3, 1.7)
    check("pow-neg", lambda a: a**-2, 1.3)


def test_grad_exp():
    check("exp", lambda a: a.exp(), 0.7)


def test_grad_tanh():
    check("tanh", lambda a: a.tanh(), -0.8)


def test_grad_relu():
    # relu 在 0 点不可导：中心差分给 0.5，解析给 0。不是 bug，避开 0
    check("relu+", lambda a: a.relu(), 1.5)
    check("relu-", lambda a: a.relu(), -1.5)


# ---------- 组合：这三个才是真正抓 bug 的 ----------

def test_grad_shared_node():
    """同一个节点用两次，梯度必须翻倍。"""
    check("shared a*a", lambda a: a * a, 3.0)

    a = Value(3.0)
    y = a * a + a
    y.backward()
    assert a.grad == 7.0, f"expected 7.0 (=2*3+1), got {a.grad}"


def test_grad_multipath():
    """多条路径汇合到同一个输入。d/da (a*b + a) = b + 1。"""
    check("multipath", lambda a, b: a * b + a, 2.0, 5.0)


def test_grad_chain():
    """混合链式，跨多层。"""
    check("chain", lambda a, b: (a + b).tanh() * a, 1.0, 2.0)


def test_grad_deep_chain():
    """四层嵌套，每层都非线性。"""
    check("deep", lambda a: ((a * 2).tanh() * 3 + a).exp().tanh() * a, 0.4)


def test_backward_accumulates():
    """不 zero_grad 时，两次 backward 梯度翻倍。与 PyTorch 行为一致。"""
    a = Value(3.0)
    y = a * a
    y.backward()
    assert a.grad == 6.0, f"expected 6.0, got {a.grad}"
    y.backward()
    assert a.grad == 12.0, f"grad should accumulate to 12.0, got {a.grad}"


def test_zero_grad():
    """清零之后重新计算，结果要和第一次一致。"""
    a = Value(3.0)
    b = Value(2.0)
    y = a * b
    y.backward()
    assert a.grad == 2.0 and b.grad == 3.0

    a.grad = 0.0
    b.grad = 0.0
    y = a * b
    y.backward()
    assert a.grad == 2.0 and b.grad == 3.0, "after reset should be identical"


# ============================================================
# 跑法
# ============================================================

if __name__ == "__main__":
    tests = [
        (name, fn)
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failed.append(name)
            print(f"FAIL  {name}\n      {e}")
        except Exception as e:
            failed.append(name)
            print(f"ERR   {name}\n      {type(e).__name__}: {e}")

    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
