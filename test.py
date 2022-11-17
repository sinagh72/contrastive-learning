import torch

x = torch.zeros(2, 3)
x1 = torch.ones(2, 3)
print(torch.cat([x,x1], dim=0))