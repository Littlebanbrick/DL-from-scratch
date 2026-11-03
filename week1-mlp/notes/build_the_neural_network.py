import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
# print(f"Using {device} device")

'''
We define our neural network by subclassing `nn.Module`, 
    and initialize the neural network layers in `__init__`. 
Every `nn.Module` subclass implements the operations on input data in the `forward` method.
'''
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        nn.Flatten() 默认不是把整个张量展平成一维，而是保留第 0 维，只把第 1 维到最后一维展平。
        nn.Flatten(start_dim=1, end_dim=-1)
        如此设计是因为 PyTorch 的惯例是 第 0 维 = batch 维
        '''
        self.flatten = nn.Flatten()
        '''
        nn.Sequential 是 PyTorch 里的一个容器模块，
            用来把多个层按顺序“串”起来，形成一个前向传播时依次执行的模型。
        '''
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        ''' logits 这个命名强调它是 softmax 之前的原始预测分数 '''
        logits = self.linear_relu_stack(x)
        return logits

''' Create an object and move it to device '''
model = NeuralNetwork().to(device)
# print(model)

''' 随机生成一个噪声图 '''
X = torch.rand(1, 28, 28, device=device)
''' 调用 model(X) 实际执行的是 NeuralNetwork.forward(X) '''
logits = model(X)
'''
    下面这行的写法相当于：
    softmax = nn.Softmax(dim=1)
    pred_probab = softmax(logits)
'''
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
# print(f"Predicted class: {y_pred}")
