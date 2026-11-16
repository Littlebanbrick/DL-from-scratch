import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    def __init__(self, n_embd, block_size, head_size):
        super().__init__()
        self.head_size = head_size
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x, kv_cache=None):        # x: (B, T, n_embd)
        B, T, C = x.shape
        k = self.key(x)              # (B, T, hs)
        q = self.query(x)            # (B, T, hs)
        v = self.value(x)            # (B, T, hs)

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=1)     # (B, T_past+T, hs)
            v = torch.cat([kv_cache[1], v], dim=1)
        new_cache = (k, v)

        T_total = k.shape[1]
        T_past = T_total - T

        # query 只有当前 T 个，key 是全部 T_total 个；
        # q 的第 i 行对应绝对位置 T_past+i，所以要从 tril 里切出对应的行
        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5   # (B, T, T_total)
        mask = self.tril[T_past:T_past + T, :T_total]
        wei = wei.masked_fill(mask == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)

        return wei @ v, new_cache                    # (B, T, hs), cache
