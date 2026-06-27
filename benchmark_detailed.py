
from git import List
import torch
import triton
import os
from dataclasses import dataclass
import time

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

print("Selected GPU type:", torch.cuda.get_device_name(torch.cuda.current_device()))

@dataclass
class ExperimentShape:
    B: int
    T: int
    N: int
    E: int
    K: int
    N_E: int


experiments: List[ExperimentShape] = [
    ExperimentShape(512, 1024, 1024, 128, 16, 512),
    ExperimentShape(8, 16384, 1024, 128, 16, 512),
    ExperimentShape(8192, 1024, 64, 64, 4, 32),
    ExperimentShape(512, 1024, 4096, 512, 16, 1024)
]

def benchmark(s: ExperimentShape, N: int = 100):
        device = 'cuda'

        test_data = torch.randn(s.B, s.T, s.N, dtype=torch.bfloat16, device=device, requires_grad=True)
        w = torch.randn(s.N_E, s.N, s.E, dtype=torch.float32, device=device, requires_grad=True)
        sel = torch.randn(s.B, s.T, s.N_E, dtype=torch.bfloat16, device=device).argsort(dim=-1, descending=True)[..., :s.K]

        sel2 = cvmm_prepare_sel2(sel)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            r = cvmm(test_data, sel2, w)  # warmup

        r.sum().backward()

        torch.cuda.synchronize()

        tsum_sel = 0
        tsum_fw = 0
        tsum_bw = 0
        for i in range(N):
            test_data.grad = None
            w.grad = None
            sel.grad = None

            start = time.perf_counter()    
            sel2 = cvmm_prepare_sel2(sel)
            
            torch.cuda.synchronize()
            t_sel = time.perf_counter()    

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                r = cvmm(test_data, sel2, w)

            torch.cuda.synchronize()
            fw = time.perf_counter()

            r.sum().backward()
            torch.cuda.synchronize()
            bw = time.perf_counter()

            tsum_sel += t_sel - start
            tsum_fw += fw - t_sel
            tsum_bw += bw - fw

        return tsum_sel / N, tsum_fw / N, tsum_bw / N


for e in experiments:
    sel_time, fw_time, bw_time = benchmark(e)
    print(f"Shape: B={e.B}, T={e.T}, N={e.N}, E={e.E}, K={e.K}, N_E={e.N_E} | sel: {sel_time*1000:.6f}ms, fw: {fw_time*1000:.6f}ms, bw: {bw_time*1000:.6f}ms")
