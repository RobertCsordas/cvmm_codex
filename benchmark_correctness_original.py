from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Any

import torch
import triton


ORIGINAL_COMMIT = "27554dc"


@dataclass(frozen=True)
class CorrectnessCase:
    name: str
    B: int
    T: int
    N: int
    E: int
    K: int
    N_E: int
    weighted: bool = False
    route_input: bool = False


CASES = [
    CorrectnessCase("top1_unweighted_masks", 2, 7, 17, 9, 1, 13),
    CorrectnessCase("top3_unweighted", 3, 5, 19, 11, 3, 7),
    CorrectnessCase("weighted_combine", 2, 9, 23, 13, 4, 11, weighted=True),
    CorrectnessCase("weighted_route_input", 2, 8, 29, 17, 4, 9, weighted=True, route_input=True),
    CorrectnessCase("larger_top8_unweighted", 2, 16, 64, 32, 8, 16),
    CorrectnessCase("larger_top8_weighted_route_input", 2, 16, 32, 64, 8, 16, weighted=True, route_input=True),
    CorrectnessCase("many_missing_experts", 1, 5, 15, 10, 2, 32, weighted=True),
]

DTYPE_MODES = ("float32", "bf16_amp", "bf16_params")

TOLERANCES = {
    "float32": {
        "out": (2e-4, 2e-4),
        "x_grad": (5e-4, 5e-4),
        "keys_grad": (2e-3, 2e-3),
        "weights_grad": (2e-3, 2e-3),
    },
    "bf16_amp": {
        "out": (1.5e-1, 6e-2),
        "x_grad": (2.5e-1, 8e-2),
        "keys_grad": (3.5e-1, 8e-2),
        "weights_grad": (3.5e-1, 8e-2),
    },
    "bf16_params": {
        "out": (2.0e-1, 8e-2),
        "x_grad": (3.0e-1, 1e-1),
        "keys_grad": (3.5e-1, 1e-1),
        "weights_grad": (3.5e-1, 1e-1),
    },
}


def install_pg199_benchmark_workaround() -> None:
    for i in range(torch.cuda.device_count()):
        if "PG199" in torch.cuda.get_device_name(i):
            print(
                "WARNING: PG199 has broken cuda event counters. "
                "Switching to cuda-graph based performance measurement."
            )

            def do_bench(fn, *args, **kwargs):
                fn()
                return triton.testing.do_bench_cudagraph(fn, *args, **kwargs)

            triton.testing.do_bench = do_bench
            break


def _dtype_for(mode: str, tensor_name: str) -> torch.dtype:
    if mode == "float32":
        return torch.float32
    if mode == "bf16_params":
        return torch.bfloat16
    if mode == "bf16_amp":
        if tensor_name in ("x",):
            return torch.bfloat16
        return torch.float32
    raise ValueError(mode)


def _autocast_enabled(mode: str) -> bool:
    return mode != "float32"


def _case_output_shape(case: CorrectnessCase) -> tuple[int, ...]:
    if case.weighted:
        return (case.B, case.T, case.E)
    return (case.B, case.T, case.K, case.E)


def _make_tensor(shape: tuple[int, ...], dtype: torch.dtype, generator: torch.Generator, scale: float) -> torch.Tensor:
    return (torch.randn(*shape, generator=generator, dtype=torch.float32) * scale).to(dtype)


def make_case_payload(case: CorrectnessCase, dtype_mode: str, seed: int) -> dict[str, Any]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    x_shape = (case.B, case.T, case.K, case.N) if case.route_input else (case.B, case.T, case.N)
    x = _make_tensor(x_shape, _dtype_for(dtype_mode, "x"), generator, scale=0.4)
    keys = _make_tensor((case.N_E, case.N, case.E), _dtype_for(dtype_mode, "keys"), generator, scale=0.3)
    sel = torch.randint(0, case.N_E, (case.B, case.T, case.K), generator=generator, dtype=torch.int64)
    grad_out = _make_tensor(_case_output_shape(case), _dtype_for(dtype_mode, "x"), generator, scale=0.2)

    weights = None
    if case.weighted:
        weights = _make_tensor((case.B, case.T, case.K), _dtype_for(dtype_mode, "weights"), generator, scale=0.25)

    return {
        "case": case.__dict__,
        "dtype_mode": dtype_mode,
        "x": x,
        "keys": keys,
        "sel": sel,
        "weights": weights,
        "grad_out": grad_out,
    }


def materialize_original_module(target_dir: Path) -> None:
    target = target_dir / "cvmm_original.py"
    with target.open("wb") as f:
        subprocess.run(["git", "show", f"{ORIGINAL_COMMIT}:cvmm.py"], cwd=Path(__file__).parent, stdout=f, check=True)


def prepare_selection(cvmm_module, sel: torch.Tensor, weights: torch.Tensor | None, route_input: bool):
    fsel = sel.flatten().to(torch.int32)
    ssel, sel_index = fsel.sort()
    if route_input:
        return cvmm_module.CVMMSel(fsel.view_as(sel), ssel.view_as(sel), sel_index, None, weights)
    n_per_batch = sel.shape[-1]
    in_index = sel_index // n_per_batch
    return cvmm_module.CVMMSel(fsel.view_as(sel), ssel.view_as(sel), in_index, sel_index, weights)


def run_worker(args: argparse.Namespace) -> None:
    if args.pg199_workaround:
        install_pg199_benchmark_workaround()

    if args.impl == "original":
        if not args.original_dir:
            raise ValueError("--original-dir is required for --impl original")
        module_path = Path(args.original_dir) / "cvmm_original.py"
        spec = importlib.util.spec_from_file_location("cvmm_original", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load {module_path}")
        cvmm_module = importlib.util.module_from_spec(spec)
        sys.modules["cvmm_original"] = cvmm_module
        spec.loader.exec_module(cvmm_module)
    else:
        import cvmm as cvmm_module

    payload = torch.load(args.case_file, map_location="cpu")
    case = CorrectnessCase(**payload["case"])
    dtype_mode = payload["dtype_mode"]
    device = "cuda"

    x = payload["x"].to(device).detach().requires_grad_(True)
    keys = payload["keys"].to(device).detach().requires_grad_(True)
    sel = payload["sel"].to(device)
    weights = payload["weights"]
    if weights is not None:
        weights = weights.to(device).detach().requires_grad_(True)
    grad_out = payload["grad_out"].to(device)

    def step():
        x.grad = None
        keys.grad = None
        if weights is not None:
            weights.grad = None
        sel_obj = prepare_selection(cvmm_module, sel, weights, case.route_input)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=_autocast_enabled(dtype_mode)):
            out = cvmm_module.cvmm(x, sel_obj, keys)
        out.backward(grad_out.to(out.dtype))
        return out

    out = step()
    torch.cuda.synchronize()

    fw_sum = 0.0
    bw_sum = 0.0
    for _ in range(args.n_iters):
        x.grad = None
        keys.grad = None
        if weights is not None:
            weights.grad = None
        sel_obj = prepare_selection(cvmm_module, sel, weights, case.route_input)

        t0 = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=_autocast_enabled(dtype_mode)):
            out = cvmm_module.cvmm(x, sel_obj, keys)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        out.backward(grad_out.to(out.dtype))
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        fw_sum += t1 - t0
        bw_sum += t2 - t1

    result = {
        "out": out.detach().cpu(),
        "x_grad": x.grad.detach().cpu(),
        "keys_grad": keys.grad.detach().cpu(),
        "weights_grad": None if weights is None else weights.grad.detach().cpu(),
        "fw_ms": fw_sum * 1000.0 / args.n_iters,
        "bw_ms": bw_sum * 1000.0 / args.n_iters,
    }
    torch.save(result, args.out_file)


def _metrics(current: torch.Tensor, original: torch.Tensor) -> dict[str, float]:
    cur = current.float()
    orig = original.float()
    diff = cur - orig
    abs_diff = diff.abs()
    denom = orig.abs().clamp_min(1e-12)
    return {
        "max_abs": abs_diff.max().item() if abs_diff.numel() else 0.0,
        "max_rel": (abs_diff / denom).max().item() if abs_diff.numel() else 0.0,
        "rmse": math.sqrt(torch.mean(diff * diff).item()) if diff.numel() else 0.0,
    }


def _compare_tensor(name: str, dtype_mode: str, current: torch.Tensor | None, original: torch.Tensor | None) -> tuple[bool, str]:
    if current is None or original is None:
        ok = current is None and original is None
        return ok, "both None" if ok else "one side is None"

    atol, rtol = TOLERANCES[dtype_mode][name]
    close = torch.allclose(current.float(), original.float(), atol=atol, rtol=rtol)
    metrics = _metrics(current, original)
    return (
        close,
        f"max_abs={metrics['max_abs']:.4g}, max_rel={metrics['max_rel']:.4g}, "
        f"rmse={metrics['rmse']:.4g}, atol={atol:g}, rtol={rtol:g}",
    )


def run_impl(
    impl: str,
    case_file: Path,
    out_file: Path,
    original_dir: Path,
    n_iters: int,
    pg199_workaround: bool,
) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        __file__,
        "--worker",
        "--impl",
        impl,
        "--case-file",
        str(case_file),
        "--out-file",
        str(out_file),
        "--n-iters",
        str(n_iters),
    ]
    if impl == "original":
        cmd.extend(["--original-dir", str(original_dir)])
    if pg199_workaround:
        cmd.append("--pg199-workaround")
    return subprocess.run(cmd, cwd=Path(__file__).parent, text=True, capture_output=True)


def run_suite(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the original/current correctness benchmark")

    selected_cases = [case for case in CASES if args.case in (None, case.name)]
    selected_modes = [mode for mode in DTYPE_MODES if args.dtype in (None, mode)]
    if not selected_cases:
        raise SystemExit(f"No case matched {args.case!r}")
    if not selected_modes:
        raise SystemExit(f"No dtype mode matched {args.dtype!r}")

    failures = 0
    with tempfile.TemporaryDirectory(prefix="cvmm_original_correctness_") as tmp:
        tmpdir = Path(tmp)
        original_dir = tmpdir / "original"
        original_dir.mkdir()
        materialize_original_module(original_dir)

        for dtype_mode in selected_modes:
            for case_idx, case in enumerate(selected_cases):
                case_file = tmpdir / f"{case.name}_{dtype_mode}.pt"
                current_file = tmpdir / f"{case.name}_{dtype_mode}_current.pt"
                original_file = tmpdir / f"{case.name}_{dtype_mode}_original.pt"
                torch.save(make_case_payload(case, dtype_mode, args.seed + case_idx), case_file)

                current_proc = run_impl("current", case_file, current_file, original_dir, args.n_iters, args.pg199_workaround)
                original_proc = run_impl("original", case_file, original_file, original_dir, args.n_iters, args.pg199_workaround)
                if current_proc.returncode != 0 or original_proc.returncode != 0:
                    failures += 1
                    print(f"FAIL {dtype_mode:11s} {case.name}: subprocess error")
                    if current_proc.returncode != 0:
                        print("current stderr:\n", current_proc.stderr)
                        print("current stdout:\n", current_proc.stdout)
                    if original_proc.returncode != 0:
                        print("original stderr:\n", original_proc.stderr)
                        print("original stdout:\n", original_proc.stdout)
                    continue

                current = torch.load(current_file, map_location="cpu")
                original = torch.load(original_file, map_location="cpu")
                tensor_results = []
                ok = True
                for tensor_name in ("out", "x_grad", "keys_grad", "weights_grad"):
                    tensor_ok, message = _compare_tensor(tensor_name, dtype_mode, current[tensor_name], original[tensor_name])
                    tensor_results.append(f"{tensor_name}: {message}")
                    ok = ok and tensor_ok

                speed = (
                    f"current fw/bw={current['fw_ms']:.3f}/{current['bw_ms']:.3f}ms, "
                    f"original fw/bw={original['fw_ms']:.3f}/{original['bw_ms']:.3f}ms"
                )
                status = "PASS" if ok else "FAIL"
                print(f"{status} {dtype_mode:11s} {case.name}: {speed}")
                if args.verbose or not ok:
                    for message in tensor_results:
                        print(f"  {message}")
                if not ok:
                    failures += 1

    if failures:
        raise SystemExit(f"{failures} original/current correctness case(s) failed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark current CVMM correctness against the original implementation.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--impl", choices=["current", "original"], help=argparse.SUPPRESS)
    parser.add_argument("--case-file", help=argparse.SUPPRESS)
    parser.add_argument("--out-file", help=argparse.SUPPRESS)
    parser.add_argument("--original-dir", help=argparse.SUPPRESS)
    parser.add_argument("--n-iters", type=int, default=1)
    parser.add_argument("--case", choices=[case.name for case in CASES])
    parser.add_argument("--dtype", choices=DTYPE_MODES)
    parser.add_argument("--seed", type=int, default=1234)
    parser.set_defaults(pg199_workaround=True)
    parser.add_argument("--pg199-workaround", dest="pg199_workaround", action="store_true")
    parser.add_argument("--no-pg199-workaround", dest="pg199_workaround", action="store_false")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker:
        run_worker(args)
    else:
        run_suite(args)


if __name__ == "__main__":
    main()
