import torch
import time

start_time = time.time()

rand = torch.rand(size=[100000000])


print(time.time() - start_time)
