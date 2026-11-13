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

    def forward(self, x):                       # x: (B, T, n_embd)
        B, T, C = x.shape
        k = self.key(x)              # (B, T, hs)
        q = self.query(x)            # (B, T, hs)
        v = self.value(x)            # (B, T, hs)

        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5   # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)                      # (B, T, T)
        return wei @ v                                    # (B, T, hs)

