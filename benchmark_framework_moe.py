from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import torch.nn as nn
import torch.nn.functional as F

from cvmm import cvmm, cvmm_prepare_sel2


DEFAULT_EXTERNAL_PATHS = (
    os.environ.get("CVMM_EXTERNAL_SITE", "/tmp/cvmm_ext_site"),
    os.environ.get("CVMM_MEGABLOCKS_SRC", "/tmp/megablocks_src"),
    os.environ.get("CVMM_DEEPSPEED_SRC", "/tmp/deepspeed_src"),
    os.environ.get("CVMM_MEGATRON_SRC", "/tmp/megatron_src"),
)


@dataclass(frozen=True)
class FrameworkShape:
    name: str
    B: int
    T: int
    D: int
    H: int
    K: int
    E: int


# These are intentionally smaller than the main CVMM fine-grained E=512/K=8
# shapes. DeepSpeed and Megatron's non-TE local paths materialize much heavier
# dispatch state or run per-expert Python loops, so the large CVMM shapes are not
# practical comparison points for those backends.
framework_shapes = [
    FrameworkShape("tiny_d128_h32_e16_k4", 2, 128, 128, 32, 4, 16),
    FrameworkShape("small_d256_h64_e32_k4", 2, 256, 256, 64, 4, 32),
    FrameworkShape("small_d512_h128_e64_k4", 1, 512, 512, 128, 4, 64),
]


@dataclass(frozen=True)
class BenchResult:
    fw_ms: float
    bw_ms: float

    @property
    def total_ms(self) -> float:
        return self.fw_ms + self.bw_ms


def _add_optional_paths() -> None:
    for path in reversed(DEFAULT_EXTERNAL_PATHS):
        if path and Path(path).exists() and path not in sys.path:
            sys.path.insert(0, path)


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


def _torch_dist_init(port: str = "29621") -> None:
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", port)
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=0, world_size=1)


def _destroy_torch_dist() -> None:
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


def benchmark_cvmm(s: FrameworkShape, n_iters: int) -> BenchResult:
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
            sel_down = cvmm_prepare_sel2(sel_up, weights, route_input=True)
            return cvmm(hidden.view(s.B, s.T, s.K, s.H), sel_down, w_down)

    return _bench(n_iters, [x, gate, w_up, w_down], forward)


class _DeepSpeedExpert(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


def _init_deepspeed_once() -> None:
    _add_optional_paths()
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29621")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    import deepspeed
    import deepspeed.comm as ds_dist

    if not ds_dist.is_initialized():
        deepspeed.init_distributed(dist_backend="nccl")


def benchmark_deepspeed(s: FrameworkShape, n_iters: int) -> BenchResult:
    _init_deepspeed_once()
    from deepspeed.moe.layer import MoE

    model = MoE(
        hidden_size=s.D,
        expert=_DeepSpeedExpert(s.D, s.H),
        num_experts=s.E,
        ep_size=1,
        k=s.K,
        capacity_factor=1.0,
        eval_capacity_factor=1.0,
        min_capacity=4,
        drop_tokens=False,
        use_tutel=False,
    ).cuda().bfloat16()
    model.set_deepspeed_parallelism()
    x = torch.randn(s.B, s.T, s.D, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    def forward() -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]
            return out

    return _bench(n_iters, [x, *list(model.parameters())], forward)


def _init_megatron_once() -> None:
    _add_optional_paths()
    _torch_dist_init("29622")
    from tests.unit_tests.test_utilities import Utils

    try:
        Utils.initialize_model_parallel(1, 1)
    except AssertionError:
        # Model parallel state is already initialized in this process.
        pass


def _num_heads(d_model: int) -> int:
    for heads in (16, 8, 4, 2, 1):
        if d_model % heads == 0:
            return heads
    return 1


def benchmark_megatron(s: FrameworkShape, n_iters: int) -> BenchResult:
    _init_megatron_once()
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.spec_utils import get_submodules
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.training.initialize import _set_random_seed

    _set_random_seed(seed_=123, data_parallel_random_init=False)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=s.D,
        num_attention_heads=_num_heads(s.D),
        num_moe_experts=s.E,
        use_cpu_initialization=True,
        moe_token_dispatcher_type="allgather",
        moe_router_load_balancing_type="aux_loss",
        moe_router_topk=s.K,
        moe_aux_loss_coeff=0.01,
        moe_grouped_gemm=False,
        moe_ffn_hidden_size=s.H,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
    )
    submodules = get_submodules(
        get_gpt_layer_local_submodules(num_experts=s.E, moe_grouped_gemm=False).mlp
    )
    model = MoELayer(config, submodules).cuda().bfloat16()
    x = torch.randn(s.T, s.B, s.D, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    def forward() -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out, _ = model(x)
            return out

    return _bench(n_iters, [x, *list(model.parameters())], forward)


def benchmark_megablocks(s: FrameworkShape, n_iters: int) -> BenchResult:
    _add_optional_paths()
    from megablocks import grouped_gemm_util as gg

    if not gg.grouped_gemm_is_available():
        raise RuntimeError("MegaBlocks grouped_gemm backend is not available")

    from megablocks.layers.arguments import Arguments
    from megablocks.layers.dmoe import dMoE
    from megablocks.layers.moe import batched_load_balancing_loss, clear_load_balancing_loss

    args = Arguments(
        hidden_size=s.D,
        ffn_hidden_size=s.H,
        moe_num_experts=s.E,
        moe_top_k=s.K,
        moe_capacity_factor=1,
        memory_optimized_mlp=True,
        mlp_type="mlp",
        mlp_impl="grouped",
        fp16=False,
        bf16=True,
        bias=False,
        return_bias=False,
    )
    model = dMoE(args).cuda().bfloat16()
    x = torch.randn(s.T, s.B, s.D, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    def forward() -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out, _ = model(x)
            return out + 0.0 * batched_load_balancing_loss(args)

    result = _bench(n_iters, [x, *list(model.parameters())], forward)
    clear_load_balancing_loss()
    return result


BENCHMARKS = {
    "cvmm": benchmark_cvmm,
    "deepspeed": benchmark_deepspeed,
    "megatron": benchmark_megatron,
    "megablocks": benchmark_megablocks,
}


def backend_status() -> dict[str, str]:
    _add_optional_paths()
    status = {"CVMM": "available"}

    try:
        importlib.import_module("deepspeed.moe.layer")
        status["DeepSpeed"] = "available"
    except Exception as exc:
        status["DeepSpeed"] = f"unavailable: {type(exc).__name__}: {exc}"

    try:
        importlib.import_module("megatron.core.transformer.moe.moe_layer")
        status["Megatron"] = "available; benchmark uses legacy local SequentialMLP unless TE is installed"
    except Exception as exc:
        status["Megatron"] = f"unavailable: {type(exc).__name__}: {exc}"

    try:
        importlib.import_module("megablocks")
        from megablocks import grouped_gemm_util as gg

        status["MegaBlocks"] = (
            "available" if gg.grouped_gemm_is_available()
            else "unavailable: grouped_gemm backend is not available"
        )
    except Exception as exc:
        status["MegaBlocks"] = f"unavailable: {type(exc).__name__}: {exc}"

    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CVMM against full MoE framework layers.")
    parser.add_argument("backend", nargs="?", default="status", choices=[*BENCHMARKS.keys(), "all", "status"])
    parser.add_argument("n_iters", nargs="?", type=int, default=5)
    parser.add_argument("--shape-index", type=int, default=None)
    parser.add_argument("--pg199-workaround", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Raise when a backend fails instead of reporting it.")
    args = parser.parse_args()

    if args.pg199_workaround:
        from benchmark_moe_shapes import install_pg199_benchmark_workaround

        install_pg199_benchmark_workaround()

    print("Selected GPU type:", torch.cuda.get_device_name(torch.cuda.current_device()))
    print("PG199 workaround:", "enabled" if args.pg199_workaround else "disabled")

    if args.backend == "status":
        for name, msg in backend_status().items():
            print(f"{name}: {msg}")
        return

    backends = list(BENCHMARKS) if args.backend == "all" else [args.backend]
    shape_indices = range(len(framework_shapes)) if args.shape_index is None else [args.shape_index]
    try:
        for idx in shape_indices:
            s = framework_shapes[idx]
            for backend in backends:
                try:
                    result = BENCHMARKS[backend](s, args.n_iters)
                except Exception as exc:
                    print(f"Backend: {backend} | Shape: {s.name} | unavailable: {type(exc).__name__}: {exc}")
                    if args.strict:
                        raise
                    continue
                print(
                    f"Backend: {backend} | Shape: {s.name} | B={s.B}, T={s.T}, D={s.D}, H={s.H}, "
                    f"K={s.K}, E={s.E} | fw: {result.fw_ms:.6f}ms, bw: {result.bw_ms:.6f}ms, "
                    f"total: {result.total_ms:.6f}ms"
                )
    finally:
        _destroy_torch_dist()


if __name__ == "__main__":
    main()
