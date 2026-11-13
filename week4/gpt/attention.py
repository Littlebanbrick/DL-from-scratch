import torch
import torch.nn as nn

from head import Head


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd 必须能被 n_head 整除"
        head_size = n_embd // n_head
        ''' 
        造出 n_head 个 Head
        必须用 ModuleList 而不是 Sequential, 因为多头不是串联而是并联
        '''
        self.heads = nn.ModuleList([Head(n_embd, block_size, head_size) for _ in range(n_head)])
        self.proj = nn.Linear(head_size * n_head, n_embd)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.proj(out)

