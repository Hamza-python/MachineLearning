import torch
a1 = torch.tensor([7, 4, 3, 2, 6])

# To find the datatype of tensor.
print(a1.dtype)

# To find Method Type.
print(a1.type)

a2 = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4])
print(a2.dtype)
print(a2.type)

# a3 = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4], dtype=torch.int32)
# print(a3.dtype)

a4 = torch.FloatTensor([0, 1, 2, 3, 4, 5])
print(a4.dtype)
# To find the size of tensor.
print(a4.size())
# To find the dimensions of the tensor.
print(a4.ndimension())

a5 = torch.tensor([0, 1, 2, 3, 4, 5])
a6 = a5.view(6,1)
print(a6)
print(a6.ndimension())