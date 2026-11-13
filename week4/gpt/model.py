import torch
import torch.nn as nn
import torch.nn.functional as F

from block import Block


class GPT(nn.Module):

    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        ''' 
        `*`把列表摊平成位置参数: 
            nn.Sequential(*[b0, b1, b2, b3])     # 等价于
            nn.Sequential(b0, b1, b2, b3)
        '''
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        # 最后一层 Block 的输出没被归一化过, 这里再来一次
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, inputs, targets=None):
        B, T = inputs.shape
        pos = torch.arange(T, device=inputs.device)
        x = self.token_embedding(inputs) + self.position_embedding(pos)   # (B,T,C)
        x = self.blocks(x)
        logits = self.lm_head(self.ln_f(x))                              # (B,T,vocab_size)

        if targets is None:
            return logits, None

        loss = F.cross_entropy(
            logits.view(B * T, -1), 
            targets.view(B * T)
        )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, temperature=1.0):
        ''' idx: (B, T) 的整数张量，返回 (B, T + max_new_tokens) '''
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]        # 超出 block_size 必须截断
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature     # (B, C)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=-1)
        return idx
