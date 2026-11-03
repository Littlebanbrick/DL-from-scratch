import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import v2

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([
        # 以下两者相当于图像预处理：
        # 把 PIL Image 转成 torch.Tensor 图像
        v2.ToImage(),
        # 把 dtype 转为 float32，并且 scale=True 表示把 [0, 255] 的整数线性缩放到 [0.0, 1.0] 的浮点数
        v2.ToDtype(torch.float32, scale=True),
    ]),
    target_transform=v2.Lambda(
        # FashionMNIST 默认返回的 label 是一个 Python 整数（0~9 的类别索引）。
        # 这里把它转成 one-hot 向量
        lambda y: F.one_hot(torch.tensor(y), num_classes=10).float()
    ),
)



