from dataclasses import dataclass
import argparse
import os
import time
from typing import Callable

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import torch.nn.functional as F

from cvmm import cvmm, cvmm_prepare_sel2


@dataclass(frozen=True)
class FFNShape:
    name: str
    B: int
    T: int
    D: int
    H: int
    K: int
    E: int


ffn_shapes = [
    FFNShape("ffn_1024", 8, 2048, 1024, 128, 8, 512),
    FFNShape("ffn_2048", 4, 2048, 2048, 128, 8, 512),
    FFNShape("ffn_768", 4, 2048, 768, 128, 8, 256),
    FFNShape("ffn_4096_stress", 2, 1024, 4096, 128, 8, 512),
    FFNShape("ffn_1024_h256", 8, 2048, 1024, 256, 8, 512),
    FFNShape("ffn_1024_h512", 8, 2048, 1024, 512, 8, 512),
    FFNShape("ffn_2048_h256", 4, 2048, 2048, 256, 8, 512),
    FFNShape("ffn_2048_h512", 4, 2048, 2048, 512, 8, 512),
    FFNShape("ffn_4096_h256_stress", 2, 1024, 4096, 256, 8, 512),
    FFNShape("ffn_4096_h512_stress", 2, 1024, 4096, 512, 8, 512),
]


@dataclass(frozen=True)
class BenchResult:
    fw_ms: float
    bw_ms: float

    @property
    def total_ms(self) -> float:
        return self.fw_ms + self.bw_ms


def _sync() -> None:
    torch.cuda.synchronize()


def _zero_grads(params) -> None:
    for p in params:
        if p is not None:
            p.grad = None


def _bench(n_iters: int, params, forward: Callable[[], torch.Tensor]) -> BenchResult:
    out = forward()
    out.sum().backward()
    _sync()
    _zero_grads(params)

    fw_sum = 0.0
    bw_sum = 0.0
    for _ in range(n_iters):
        _zero_grads(params)
        t0 = time.perf_counter()
        out = forward()
        _sync()
        t1 = time.perf_counter()
        out.sum().backward()
        _sync()
        t2 = time.perf_counter()
        fw_sum += t1 - t0
        bw_sum += t2 - t1
    scale = 1000.0 / n_iters
    return BenchResult(fw_sum * scale, bw_sum * scale)


def benchmark_cvmm_ffn(s: FFNShape, n_iters: int) -> BenchResult:
    x = torch.randn(s.B, s.T, s.D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    gate = torch.randn(s.D, s.E, device="cuda", dtype=torch.float32, requires_grad=True)
    w_up = torch.randn(s.E, s.D, s.H, device="cuda", dtype=torch.float32, requires_grad=True)
    w_down = torch.randn(s.E, s.H, s.D, device="cuda", dtype=torch.float32, requires_grad=True)

    def forward() -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = x @ gate
            probs = torch.softmax(logits.float(), dim=-1)
            weights, sel = probs.topk(s.K, dim=-1)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
            sel_up = cvmm_prepare_sel2(sel)
            hidden = F.gelu(cvmm(x, sel_up, w_up))
            sel_down = cvmm_prepare_sel2(sel, weights, route_input=True)
            return cvmm(hidden.view(s.B, s.T, s.K, s.H), sel_down, w_down)

    return _bench(n_iters, [x, gate, w_up, w_down], forward)


def _init_tutel_once() -> None:
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29573")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("LOCAL_SIZE", "1")
    from tutel import system
    system.init_data_model_parallel(backend="nccl")


def benchmark_tutel_ffn(s: FFNShape, n_iters: int) -> BenchResult:
    _init_tutel_once()
    from tutel import moe as tutel_moe

    model = tutel_moe.moe_layer(
        gate_type={"type": "top", "k": s.K, "capacity_factor": 0.0, "fp32_gate": False},
        experts={
            "type": "ffn",
            "num_experts_per_device": s.E,
            "hidden_size_per_expert": s.H,
            "activation_fn": F.gelu,
            "has_fc1_bias": False,
            "has_fc2_bias": False,
        },
        model_dim=s.D,
        seeds=(1, 2, 3),
    ).cuda()
    x = torch.randn(s.B, s.T, s.D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    params = [x, *list(model.parameters())]

    def forward() -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return model(x)

    return _bench(n_iters, params, forward)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark full MoE FFN stacks.")
    parser.add_argument("backend", choices=["cvmm", "tutel"])
    parser.add_argument("n_iters", nargs="?", type=int, default=5)
    parser.add_argument("--shape-index", type=int, default=None)
    parser.add_argument("--pg199-workaround", action="store_true")
    args = parser.parse_args()

    if args.pg199_workaround:
        from benchmark_moe_shapes import install_pg199_benchmark_workaround
        install_pg199_benchmark_workaround()
    print("Selected GPU type:", torch.cuda.get_device_name(torch.cuda.current_device()))
    print("PG199 workaround:", "enabled" if args.pg199_workaround else "disabled")
    shape_indices = range(len(ffn_shapes)) if args.shape_index is None else [args.shape_index]
    for idx in shape_indices:
        s = ffn_shapes[idx]
        if args.backend == "cvmm":
            result = benchmark_cvmm_ffn(s, args.n_iters)
        elif args.backend == "tutel":
            result = benchmark_tutel_ffn(s, args.n_iters)
        else:
            raise AssertionError(args.backend)
        print(
            f"Backend: {args.backend} | Shape: {s.name} | B={s.B}, T={s.T}, D={s.D}, H={s.H}, K={s.K}, E={s.E} | "
            f"fw: {result.fw_ms:.6f}ms, bw: {result.bw_ms:.6f}ms, total: {result.total_ms:.6f}ms"
        )


if __name__ == "__main__":
    main()
