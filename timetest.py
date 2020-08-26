# -*- coding: utf-8 -*-
#%%
import time
import torch
import numpy as np

device = torch.device("cuda")

mat1 = torch.randn((100,100))
mat2 = torch.randn((100,100))

start = time.time()
rst = torch.matmul(mat1, mat2)
end = time.time()

print("cpu time: " + str(end-start))

mat1_g = mat1.to(device)
mat2_g = mat2.to(device)
start_g = time.time()
rst = torch.matmul(mat1_g, mat2_g)
end_g = time.time()

print("gpu time:" + str(end_g-start_g))

tmp = np.random.rand(100)
st = time.time()
torch.from_numpy(tmp).to(device)
ed = time.time()
print("to_device: " + str(ed-st))

st = time.time()
torch.from_numpy(tmp).cuda()
ed = time.time()
print("cuda: " + str(ed-st))



