'''
当对张量执行运算时，autograd 会在后台构建一张计算图（Computational Graph），记录下数据是怎么一步步算出来的。
要让 PyTorch 追踪某个张量（通常是模型参数），需要设置 `requires_grad=True`
'''

'''
完整的参数更新流程：

设置 requires_grad=True
        ↓
前向传播算出 loss（同时构建计算图）
        ↓
loss.backward()  → 梯度自动填入 .grad （反向传播不会直接修改参数，而是把算好的梯度存到每个张量的 .grad 属性里）
        ↓
optimizer.step()  → 根据 .grad 更新参数
        ↓
optimizer.zero_grad()  → 梯度清零，准备下一轮
'''

import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

loss.backward()
print(w.grad)
print(b.grad)
