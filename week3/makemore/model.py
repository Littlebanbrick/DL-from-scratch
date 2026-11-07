import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, V=27, emb=2, hidden=100, block=3):
        super().__init__()
        self.block = block
        self.C = nn.Embedding(V, emb)              # 字符 -> emb 维向量
        self.fc1 = nn.Linear(emb * block, hidden)  # 3 个字符拼起来
        self.fc2 = nn.Linear(hidden, V)

    def forward(self, x):                          # x: [N, block]
        e = self.C(x)                              # [N, 3, emb]
        h = e.view(e.shape[0], -1)                 # [N, 3*emb]
        h = torch.tanh(self.fc1(h))                # [N, hidden]
        return self.fc2(h)                         # [N, V] logits
