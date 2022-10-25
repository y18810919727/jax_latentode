#!/usr/bin/python
# -*- coding:utf8 -*-
from common import Interpolation
from matplotlib import pyplot as plt
import torch
X = torch.randn((10, 5, 2))
t = torch.randn((10, 1))
t = t + abs(t.min()) + 0.1
t = torch.cumsum(t, dim=0)
# t = torch.arange(10).float().unsqueeze(dim=-1) / 10
len, batch_size, dim = X.shape

device = X.device
# inter = KernelInterpolation(sfs, ells, t, X, eps=1e-5, kernel='exp')

inter = Interpolation(t, X)

# for b in range(batch_size):
#     ts.append(torch.linspace(t[b].min(), t[b].max(), 20).unsqueeze(-1))
# ts = torch.stack(ts, dim=0)

ts = torch.linspace(t.min(), t.max(), 30)

pred_X = torch.stack([inter(t) for t in ts], dim=0)
plt.figure()
for b in range(batch_size):
    for j in range(dim):
        print(b, j)
        plt.subplot(batch_size, dim, b*dim+j+1)
        plt.plot(ts, pred_X[:, b, j], 'r-', label='pred')
        plt.scatter(t.squeeze(), X[:, b, j], label='true')
# plt.show()
plt.savefig('test.png')
# plt.show()
plt.close()
print('end')