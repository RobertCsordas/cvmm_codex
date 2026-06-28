from dataclasses import dataclass
import argparse
import importlib
import os
import time
from typing import Callable, Optional

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch

from benchmark_moe_shapes import ExperimentShape, experiments, install_pg199_benchmark_workaround


@dataclass(frozen=True)
class BackendResult:
    prep_ms: float
    fw_ms: float
    bw_ms: float

    @property
    def total_ms(self) -> float:
        return self.prep_ms + self.fw_ms + self.bw_ms


def _sync() -> None:
    torch.cuda.synchronize()


def _clear_grad(*tensors: Optional[torch.Tensor]) -> None:
    for tensor in tensors:
        if tensor is not None:
            tensor.grad = None


def _make_common_inputs(s: ExperimentShape, weight_dtype: torch.dtype):
    device = "cuda"
    x_shape = (s.B, s.T, s.K, s.N) if s.route_input else (s.B, s.T, s.N)
    x = torch.randn(*x_shape, dtype=torch.bfloat16, device=device, requires_grad=True)
    w = torch.randn(s.N_E, s.N, s.E, dtype=weight_dtype, device=device, requires_grad=True)
    sel = torch.randn(s.B, s.T, s.N_E, dtype=torch.bfloat16, device=device).topk(s.K, dim=-1).indices
    gates = None
    if s.weighted:
        # ScatterMoE's native weighted path requires gates to match activation dtype.
        gates = torch.randn(s.B, s.T, s.K, dtype=torch.bfloat16, device=device, requires_grad=True)
    return x, w, sel, gates


def _bench_loop(n_iters: int, step: Callable[..., tuple[object, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]) -> BackendResult:
    prep_sum = 0.0
    fw_sum = 0.0
    bw_sum = 0.0
    for _ in range(n_iters):
        prep_obj, x, w, gates = step("prep")
        _clear_grad(x, w, gates)
        _sync()

        t0 = time.perf_counter()
        prep_obj, x, w, gates = step("prep")
        _sync()
        t1 = time.perf_counter()
        out, x, w, gates = step("forward", prep_obj)
        _sync()
        t2 = time.perf_counter()
        out.sum().backward()
        _sync()
        t3 = time.perf_counter()

        prep_sum += t1 - t0
        fw_sum += t2 - t1
        bw_sum += t3 - t2
    scale = 1000.0 / n_iters
    return BackendResult(prep_sum * scale, fw_sum * scale, bw_sum * scale)


def benchmark_scattermoe(s: ExperimentShape, n_iters: int) -> BackendResult:
    scatter_parallel = importlib.import_module("scattermoe.parallel_experts")
    scatter_kernels = importlib.import_module("scattermoe.kernels")
    flatten_sort_count = scatter_parallel.flatten_sort_count
    parallel_linear = scatter_parallel.parallel_linear
    group = scatter_kernels.ops.group

    x, w, sel, gates = _make_common_inputs(s, torch.bfloat16)
    flat_x = x.flatten(end_dim=-2)
    flat_sel = sel.flatten(end_dim=-2)
    flat_gates = gates.flatten(end_dim=-2) if gates is not None else None

    def step(phase: str, prep=None):
        if phase == "prep":
            return flatten_sort_count(flat_sel, s.N_E), x, w, gates
        sorted_expert_idxs, sorted_scattered_idxs, expert_offsets = prep
        if s.route_input:
            grouped_x = group(flat_x, sorted_scattered_idxs, fan_out=1)
            out = parallel_linear(
                grouped_x, w, 1,
                sorted_expert_idxs, sorted_scattered_idxs, expert_offsets,
                gates=flat_gates,
                grouped_in=True,
                grouped_out=False,
            )
        else:
            out = parallel_linear(
                flat_x, w, s.K,
                sorted_expert_idxs, sorted_scattered_idxs, expert_offsets,
                gates=flat_gates,
                grouped_in=False,
                grouped_out=False,
            )
        return out, x, w, gates

    # Warm up Python dispatch, torch.compile, Triton compilation/autotune, and backward.
    prep, _, _, _ = step("prep")
    out, _, _, _ = step("forward", prep)
    out.sum().backward()
    _sync()
    _clear_grad(x, w, gates)

    return _bench_loop(n_iters, step)


def backend_status() -> dict[str, str]:
    status = {}
    try:
        importlib.import_module("scattermoe.parallel_experts")
        status["ScatterMoE"] = "available"
    except Exception as exc:
        status["ScatterMoE"] = f"unavailable: {type(exc).__name__}: {exc}"

    try:
        importlib.import_module("megablocks")
        status["MegaBlocks"] = "importable, but FFN-level only here; projection script requires a selected-linear primitive"
    except Exception as exc:
        status["MegaBlocks"] = f"unavailable: {type(exc).__name__}: {exc}"

    try:
        importlib.import_module("tutel")
        status["Tutel"] = "importable, but FFN-level only here; projection script requires a selected-linear primitive"
    except Exception as exc:
        status["Tutel"] = f"unavailable: {type(exc).__name__}: {exc}"
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark external MoE libraries against CVMM shapes.")
    parser.add_argument("backend", nargs="?", default="scattermoe", choices=["scattermoe", "status"])
    parser.add_argument("n_iters", nargs="?", type=int, default=8)
    parser.add_argument("--shape-index", type=int, default=None)
    args = parser.parse_args()

    install_pg199_benchmark_workaround()
    print("Selected GPU type:", torch.cuda.get_device_name(torch.cuda.current_device()))

    if args.backend == "status":
        for name, msg in backend_status().items():
            print(f"{name}: {msg}")
        return

    if args.shape_index is None:
        shape_indices = range(len(experiments))
    else:
        shape_indices = [args.shape_index]

    for idx in shape_indices:
        s = experiments[idx]
        try:
            if args.backend == "scattermoe":
                result = benchmark_scattermoe(s, args.n_iters)
            else:
                raise AssertionError(args.backend)
            print(
                f"Backend: {args.backend} | Shape: {s.name} | "
                f"prep: {result.prep_ms:.6f}ms, fw: {result.fw_ms:.6f}ms, "
                f"bw: {result.bw_ms:.6f}ms, total: {result.total_ms:.6f}ms"
            )
        except Exception as exc:
            print(f"Backend: {args.backend} | Shape: {s.name} | unavailable: {type(exc).__name__}: {exc}")
            if args.shape_index is not None:
                raise


if __name__ == "__main__":
    main()
