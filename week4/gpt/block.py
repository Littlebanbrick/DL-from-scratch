import torch
import torch.nn as nn

from attention import MultiHeadAttention


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_head, block_size)
        self.ffwd = FeedForward(n_embd)
        '''
        LayerNorm: 沿特征维（最后一维）归一化成均值 0、方差 1, 每个 token 独立处理。
        作用是把激活值的尺度拉回稳定区间，否则深网络里数值会逐层漂移。
        '''
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        ''' pre-norm: 先归一化再进子层, 残差那条路完全干净, 梯度可以直通 '''
        x = x + self.sa(self.ln1(x))        # 1. attention 跨位置交流
        x = x + self.ffwd(self.ln2(x))      # 2. FFN 单位置加工
        return x
