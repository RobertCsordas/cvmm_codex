
from git import List
import torch
import triton
import os
from dataclasses import dataclass

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from cvmm import cvmm_prepare_sel2, cvmm

for i in range(torch.cuda.device_count()):
    if "PG199" in torch.cuda.get_device_name(i):
        print("WARNING: PG199 has broken cuda event counters. Switching to cuda-graph based performance measurement.")
        def do_bench(fn, *args, **kwargs):
            # Do a warmup, so if autotune runs, it doesn't run in a nested cuda graph
            fn()
            return triton.testing.do_bench_cudagraph(fn, *args, **kwargs)
        triton.testing.do_bench = do_bench
        break


B, T, N = 512, 1024, 1024
E=128
K=16
N_E = 512
device = 'cuda'

test_data = torch.randn(B, T, N, dtype=torch.bfloat16, device=device, requires_grad=True)
w = torch.randn(N_E, N, E, dtype=torch.float32, device=device, requires_grad=True)
sel = torch.randn(B, T, N_E, dtype=torch.bfloat16, device=device).topk(K, dim=-1).indices

print(sel.min().item(), sel.max().item())
sel2 = cvmm_prepare_sel2(sel)

with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    r = cvmm(test_data, sel2, w)  # warmup
print(r.shape)

r.sum().backward()