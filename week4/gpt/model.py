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
        这里要用 ModuleList 而不是 Sequential: 每层都要收/发自己的 kv_cache,
        Sequential 没法把 cache 传下去。
        '''
        self.blocks = nn.ModuleList([Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        # 最后一层 Block 的输出没被归一化过, 这里再来一次
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, inputs, targets=None, kv_caches=None, use_cache=False):
        B, T = inputs.shape

        # 有 cache 时, 当前位置不是从 0 开始, 位置编码要用绝对偏移
        T_past = 0
        if kv_caches is not None and kv_caches[0] is not None:
            T_past = kv_caches[0][0][0].shape[1]
        pos = torch.arange(T_past, T_past + T, device=inputs.device)

        x = self.token_embedding(inputs) + self.position_embedding(pos)   # (B,T,C)

        new_caches = []
        for i, block in enumerate(self.blocks):
            cache = None if kv_caches is None else kv_caches[i]
            x, c = block(x, cache)
            new_caches.append(c)

        logits = self.lm_head(self.ln_f(x))                              # (B,T,vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(B * T, -1),
                targets.view(B * T)
            )

        if use_cache:
            return logits, loss, new_caches
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, temperature=1.0, use_cache=False):
        ''' idx: (B, T) 的整数张量，返回 (B, T + max_new_tokens) '''
        if not use_cache:
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.block_size:]        # 超出 block_size 必须截断
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature     # (B, C)
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=-1)
            return idx

        # 有 cache: 先 prefill 整个 prompt, 之后每步只喂 1 个 token
        n_new = min(max_new_tokens, self.block_size - idx.shape[1])
        logits, _, caches = self(idx, use_cache=True)
        for _ in range(n_new):
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=-1)
            logits, _, caches = self(idx_next, kv_caches=caches, use_cache=True)
        return idx
