import torch
import numpy as np

data = [[1,2], [3,4]]
x_data = torch.tensor(data)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

''' retains the properties (shape, datatype) of x_data '''
x_ones = torch.ones_like(x_data)
# print(f"Ones Tensor: \n {x_ones} \n")

''' overrides the datatype of x_data '''
x_rand = torch.rand_like(x_data, dtype=torch.float)
# print(f"Random Tensor: \n {x_rand} \n")

''' `shape` is a tuple of tensor dimensions '''
shape = (2,3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

# print(f"Random Tensor: \n {rand_tensor} \n")
# print(f"Ones Tensor: \n {ones_tensor} \n")
# print(f"Zeros Tensor: \n {zeros_tensor}")

tensor = torch.rand(3,4)

# print(f"Shape of tensor: {tensor.shape}")
# print(f"Datatype of tensor: {tensor.dtype}")
# print(f"Device tensor is stored on: {tensor.device}")

'''
OPERATIONS ON TENSORS

By default, tensors are created on the CPU. 
We need to explicitly move tensors to the accelerator using `.to` method 
    (after checking for accelerator availability).
'''

if torch.accelerator.is_available():
    tensor = tensor.to(torch.accelerator.current_accelerator())

tensor = torch.ones(4, 4)
# print(f"First row: {tensor[0]}")
# print(f"First column: {tensor[:, 0]}")
# print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
# print(tensor)

''' Joining the tensors together '''
t1 = torch.cat([tensor, tensor, tensor], dim=1)
# print(t1)

''' Arithmetic operations '''
# ``tensor.T`` returns the transpose of a tensor
# y1, y2 and y3 will have the same value
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)

# This computes the element-wise product. 
# z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

''' One-element tensor '''
agg = tensor.sum()
agg_item = agg.item()
# print(agg_item, type(agg_item))

'''
In-place operations
Operations that store the result into the operand are called in-place. 
They are denoted by a _ suffix. 
For example: x.copy_(y), x.t_(), will change x.
'''

# print(f"{tensor} \n")
tensor.add_(5)
# print(tensor)
