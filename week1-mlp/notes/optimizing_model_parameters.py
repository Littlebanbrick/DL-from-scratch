''' 
    TRIVIAL 
'''

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()

'''
We define the following hyperparameters for training:
    Number of Epochs - the number of times to iterate over the dataset
    Batch Size - the number of data samples propagated through the network 
                    before the parameters are updated
    Learning Rate - how much to update models parameters at each batch/epoch. 
'''
learning_rate = 1e-3
batch_size = 64
epochs = 5

''' Initialize the loss function '''
loss_fn = nn.CrossEntropyLoss()

'''
Optimizer（优化器）是负责“根据梯度更新模型参数”的对象。
autograd 只负责算出梯度（存进 .grad），优化器负责拿这些梯度去修改参数，让损失变小。

传入什么？
    model.parameters()：告诉优化器“要更新哪些张量”。优化器内部保存的是这些参数的引用，所以 step() 能原地修改它们。
    lr（学习率）：每次更新的步长。

此处使用 SGD 优化器：最基础，`w -= lr * grad`

Inside the training loop, optimization happens in three steps:
    Call optimizer.zero_grad() to reset the gradients of model parameters. 
    Backpropagate the prediction loss with a call to loss.backward(). 
        PyTorch deposits the gradients of the loss w.r.t. each parameter.
    Once we have our gradients, we call optimizer.step() to adjust the parameters 
        by the gradients collected in the backward pass.

'''
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

''' FULL IMPLEMENTATION '''

def train_loop(dataloader, model, loss_fn, optimizer):
    # 训练集样本总数，用于打印进度
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)             # 1. 前向传播
        loss = loss_fn(pred, y)     # 2. 计算损失

        # Backpropagation
        loss.backward()             # 3.
        optimizer.step()            # 4.
        optimizer.zero_grad()       # 5.

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")


