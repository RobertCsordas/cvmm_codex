from dataclasses import dataclass
from typing import List
import os
import subprocess
import sys
import time

import torch
import triton

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import cvmm as cvmm_module
from cvmm import CVMMSel, cvmm, cvmm_prepare_sel2


def install_pg199_benchmark_workaround():
    for i in range(torch.cuda.device_count()):
        if "PG199" in torch.cuda.get_device_name(i):
            print("WARNING: PG199 has broken cuda event counters. Switching to cuda-graph based performance measurement.")

            def do_bench(fn, *args, **kwargs):
                fn()
                return triton.testing.do_bench_cudagraph(fn, *args, **kwargs)

            triton.testing.do_bench = do_bench
            break


@dataclass(frozen=True)
class ExperimentShape:
    name: str
    B: int
    T: int
    N: int
    E: int
    K: int
    N_E: int
    weighted: bool = False
    route_input: bool = False


experiments: List[ExperimentShape] = [
    # SigmaMoE feed-forward up/down projections from moeut.py.
    ExperimentShape("ff_up_1024", 8, 2048, 1024, 128, 8, 512),
    ExperimentShape("ff_down_1024_weighted", 8, 2048, 128, 1024, 8, 512, weighted=True, route_input=True),
    ExperimentShape("ff_up_1024_h256", 8, 2048, 1024, 256, 8, 512),
    ExperimentShape("ff_down_1024_h256_weighted", 8, 2048, 256, 1024, 8, 512, weighted=True, route_input=True),
    ExperimentShape("ff_up_1024_h512", 8, 2048, 1024, 512, 8, 512),
    ExperimentShape("ff_down_1024_h512_weighted", 8, 2048, 512, 1024, 8, 512, weighted=True, route_input=True),
    ExperimentShape("ff_up_2048", 4, 2048, 2048, 128, 8, 512),
    ExperimentShape("ff_down_2048_weighted", 4, 2048, 128, 2048, 8, 512, weighted=True, route_input=True),
    ExperimentShape("ff_up_2048_h256", 4, 2048, 2048, 256, 8, 512),
    ExperimentShape("ff_down_2048_h256_weighted", 4, 2048, 256, 2048, 8, 512, weighted=True, route_input=True),
    ExperimentShape("ff_up_2048_h512", 4, 2048, 2048, 512, 8, 512),
    ExperimentShape("ff_down_2048_h512_weighted", 4, 2048, 512, 2048, 8, 512, weighted=True, route_input=True),
    # SwitchHead v/o projections. N_E is n_heads * n_experts and K is n_heads * moe_k.
    ExperimentShape("switch_v_1024", 8, 2048, 1024, 64, 32, 512),
    ExperimentShape("switch_o_1024_weighted", 8, 2048, 64, 1024, 32, 512, weighted=True, route_input=True),
    ExperimentShape("switch_v_2048_h32", 4, 2048, 2048, 64, 64, 1024),
    ExperimentShape("switch_o_2048_h32_weighted", 4, 2048, 64, 2048, 64, 1024, weighted=True, route_input=True),
    # Smaller consumer-GPU oriented model shapes.
    ExperimentShape("ff_up_768", 4, 2048, 768, 128, 8, 256),
    ExperimentShape("switch_o_768_weighted", 4, 2048, 64, 768, 24, 384, weighted=True, route_input=True),
    # Larger-model stress probes; 1024-2048 is the main decision range.
    ExperimentShape("ff_up_4096_stress", 2, 1024, 4096, 128, 8, 512),
    ExperimentShape("ff_down_4096_weighted_stress", 2, 1024, 128, 4096, 8, 512, weighted=True, route_input=True),
    ExperimentShape("ff_up_4096_h256_stress", 2, 1024, 4096, 256, 8, 512),
    ExperimentShape("ff_down_4096_h256_weighted_stress", 2, 1024, 256, 4096, 8, 512, weighted=True, route_input=True),
    ExperimentShape("ff_up_4096_h512_stress", 2, 1024, 4096, 512, 8, 512),
    ExperimentShape("ff_down_4096_h512_weighted_stress", 2, 1024, 512, 4096, 8, 512, weighted=True, route_input=True),
]


def prepare_selection(sel: torch.Tensor, weights: torch.Tensor | None, route_input: bool) -> CVMMSel:
    return cvmm_prepare_sel2(sel, weights, route_input=route_input)


def benchmark(s: ExperimentShape, n_iters: int):
    device = "cuda"

    x_shape = (s.B, s.T, s.K, s.N) if s.route_input else (s.B, s.T, s.N)
    test_data = torch.randn(*x_shape, dtype=torch.bfloat16, device=device, requires_grad=True)
    w = torch.randn(s.N_E, s.N, s.E, dtype=torch.float32, device=device, requires_grad=True)
    sel = torch.randn(s.B, s.T, s.N_E, dtype=torch.bfloat16, device=device).topk(s.K, dim=-1).indices
    weights = None
    if s.weighted:
        weights = torch.randn(s.B, s.T, s.K, dtype=torch.float32, device=device, requires_grad=True)

    sel2 = prepare_selection(sel, weights, s.route_input)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        r = cvmm(test_data, sel2, w)
    r.sum().backward()
    torch.cuda.synchronize()

    tsum_sel = 0.0
    tsum_fw = 0.0
    tsum_bw = 0.0
    for _ in range(n_iters):
        test_data.grad = None
        w.grad = None
        if weights is not None:
            weights.grad = None

        start = time.perf_counter()
        sel2 = prepare_selection(sel, weights, s.route_input)
        torch.cuda.synchronize()
        t_sel = time.perf_counter()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            r = cvmm(test_data, sel2, w)
        torch.cuda.synchronize()
        t_fw = time.perf_counter()

        r.sum().backward()
        torch.cuda.synchronize()
        t_bw = time.perf_counter()

        tsum_sel += t_sel - start
        tsum_fw += t_fw - t_sel
        tsum_bw += t_bw - t_fw

    return tsum_sel / n_iters, tsum_fw / n_iters, tsum_bw / n_iters


def run_one(index: int, n_iters: int):
    install_pg199_benchmark_workaround()
    if os.environ.get("CVMM_FORCE_LOWMEM_BACKWARD") == "1":
        cvmm_module.cvmm_use_lowmem_grouped_backward = True
        cvmm_module.cvmm_lowmem_min_top_k = 0
        if "CVMM_LOWMEM_CHUNK_SIZE" in os.environ:
            cvmm_module.cvmm_lowmem_chunk_size = int(os.environ["CVMM_LOWMEM_CHUNK_SIZE"])
    print("Selected GPU type:", torch.cuda.get_device_name(torch.cuda.current_device()))
    e = experiments[index]
    sel_time, fw_time, bw_time = benchmark(e, n_iters)
    print(
        f"Shape: {e.name} | B={e.B}, T={e.T}, N={e.N}, E={e.E}, K={e.K}, N_E={e.N_E}, "
        f"weighted={e.weighted}, route_input={e.route_input} | "
        f"sel: {sel_time * 1000:.6f}ms, fw: {fw_time * 1000:.6f}ms, bw: {bw_time * 1000:.6f}ms"
    )


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--shape-index":
        iters = int(sys.argv[3]) if len(sys.argv) >= 4 else 20
        run_one(int(sys.argv[2]), iters)
    else:
        iters = int(sys.argv[1]) if len(sys.argv) >= 2 else 20
        for i in range(len(experiments)):
            subprocess.run([sys.executable, __file__, "--shape-index", str(i), str(iters)], check=True)
