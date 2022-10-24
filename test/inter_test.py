#!/usr/bin/python
# -*- coding:utf8 -*-
from test.interpolation import KernelInterpolation
from matplotlib import pyplot as plt
import torch
X = torch.randn((5, 10, 2))
t = torch.randn((5, 10, 1))
t = t + t.min()
t = torch.cumsum(t, dim=1)
batch_size, len, dim = X.shape

device = X.device
sfs = 1.0 * torch.ones([batch_size, 1, 1], device=device, dtype=torch.float32)
ells = 0.5 * torch.ones([batch_size, 1, 1], device=device, dtype=torch.float32)

inter = KernelInterpolation(sfs, ells, t, X, eps=1e-5, kernel='exp')

ts = []
for b in range(batch_size):
    ts.append(torch.linspace(t[b].min(), t[b].max(), 20).unsqueeze(-1))
ts = torch.stack(ts, dim=0)

pred_X = inter(ts)


plt.figure()
for b in range(batch_size):
    for j in range(dim):
        print(b, j)
        plt.subplot(batch_size, dim, b*dim+j+1)
        plt.plot(ts[b], pred_X[b, :, j], 'r-')
        plt.plot(t[b], X[b, :, j], 'b.')
# plt.show()
plt.savefig('test.png')
plt.close()
print('end')