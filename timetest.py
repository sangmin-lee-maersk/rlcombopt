# -*- coding: utf-8 -*-
#%%
import time
import torch

device = torch.device("cuda")

mat1 = torch.randn((100,100))
mat2 = torch.randn((100,100))

start = time.time()
rst = torch.matmul(mat1, mat2)
end = time.time()

print("cpu time: " + str(end-start))

start_g = time.time()
mat1_g = mat1.to(device)
mat2_g = mat2.to(device)
rst = torch.matmul(mat1_g, mat2_g)
end_g = time.time()

print("gpu time:" + str(end_g-start_g))
