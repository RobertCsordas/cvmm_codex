# CVMM

Conditional vector-matrix multiplication kernels for MoE-style projections. Used by [MoEUT](https://github.com/RobertCsordas/moeut) and [σ-MoE](https://github.com/RobertCsordas/moe_layer/blob/master/triton_src/moe_layer/cvmm.py).

This is a vibe-optimized version of the kernel using Codex 5.5. Should be used with caution.

The code is optimized on NVIDIA DRIVE-PG199-PROD (in my home GPU server), which is a 32Gb A100 variant. Should work equally well on standard A100s.

## Benchmarks

Measured on an `NVIDIA DRIVE-PG199-PROD` on 2026-06-27 with the PG199
CUDA-event workaround enabled in `benchmark_moe_shapes.py`.

The raw score columns are milliseconds per iteration from
`python3 benchmark_moe_shapes.py 8`.

The original/current correctness benchmark compares this checkout with the
historical baseline commit `27554dc` in separate Python processes, using a
temporary `cvmm_original.py` module so the current `cvmm.py` is not modified or
shadowed in-repo:

```bash
python3 benchmark_correctness_original.py --n-iters 1
```

Use `--dtype float32`, `--dtype bf16_amp`, or `--dtype bf16_params` to run one
precision mode, and `--case <name>` to run one shape.


The ratio columns use kernel time only (`fw + bw`):

- `vs original`: `(original cvmm fw+bw) / (current fw+bw)`, using commit
  `27554dc` as the original baseline.
- `vs dense`: `(current fw+bw) / (dense equivalent fw+bw)`. The dense
  equivalent is a single shared dense weight with pre-materialized route-shaped
  input and the same number of route matmuls. Selection is excluded because
  the dense equivalent has no selector.

Values below `1.0x` in `vs dense` mean this benchmark run measured CVMM faster
than the dense-equivalent probe for that shape.

`scratch / base` is the estimated peak extra scratchpad as a percentage of
the already-required training tensors: input, output, grad input, grad output,
weights, grad weights, and weighted-reduction weights/grads when present. The
estimate uses the benchmark dtypes (BF16 activations and FP32 weights), includes
CVMM-prepared selector metadata and temporary route/group buffers, and excludes
CUDA allocator cache, allocator fragmentation, and the router's original
selection tensor.

`ScatterMoE / CVMM` is the total projection time ratio:
`(ScatterMoE prep + fw + bw) / (CVMM sel + fw + bw)`, so lower is faster and
`1.0x` is parity with CVMM. ScatterMoE was measured from source commit
`47b5e15` using its native BF16 `ParallelLinear` path; its backward returns
BF16 weight/gate gradients, so these numbers are a speed reference rather than
a precision-identical replacement.

| Shape | sel ms | fw ms | bw ms | total ms | vs original | vs dense | scratch / base | ScatterMoE / CVMM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ff_up_1024` | 0.427 | 1.230 | 2.666 | 4.322 | 1.93x | 2.58x | 45.5% | 1.23x |
| `ff_down_1024_weighted` | 0.370 | 2.073 | 2.601 | 5.044 | 2.50x | 1.02x | 40.2% | 2.56x |
| `ff_up_2048` | 0.406 | 1.610 | 3.039 | 5.055 | 2.13x | 2.86x | 24.4% | 1.33x |
| `ff_down_2048_weighted` | 0.347 | 2.184 | 2.913 | 5.445 | 2.53x | 1.12x | 22.9% | 2.76x |
| `switch_v_1024` | 0.352 | 1.450 | 4.637 | 6.440 | 1.80x | 1.53x | 78.5% | 1.35x |
| `switch_o_1024_weighted` | 0.292 | 4.263 | 3.879 | 8.434 | 1.93x | 0.93x | 228.3% | 4.71x |
| `switch_v_2048_h32` | 0.282 | 2.640 | 8.522 | 11.444 | 2.02x | 1.47x | 50.0% | 1.36x |
| `switch_o_2048_h32_weighted` | 0.256 | 7.074 | 7.286 | 14.616 | 1.69x | 1.03x | 168.5% | 5.12x |
| `ff_up_768` | 0.420 | 0.789 | 1.692 | 2.901 | 1.53x | 2.40x | 45.8% | 1.41x |
| `switch_o_768_weighted` | 0.377 | 1.796 | 1.803 | 3.977 | 1.92x | 1.14x | 133.8% | 3.32x |
| `ff_up_4096_stress` | 0.381 | 1.898 | 3.652 | 5.931 | 2.29x | 5.57x | 6.3% | 1.46x |
| `ff_down_4096_weighted_stress` | 0.300 | 2.192 | 3.348 | 5.840 | 2.55x | 2.12x | 6.1% | 2.90x |


### Larger FFN expert-width probes

Measured on the same GPU on 2026-06-28 with the PG199 CUDA-event workaround
enabled. These rows extend the FFN up/down projection probes to hidden widths
256 and 512; all fit with the batch sizes shown in `benchmark_moe_shapes.py`.
Raw score columns are milliseconds per iteration from 8 iterations. Original
baseline columns use kernel time (`fw + bw`) from commit `27554dc`; the temporary
selector shim only adapts that baseline to the current route-input benchmark
shape API and does not change its kernels.

| Shape | sel ms | fw ms | bw ms | total ms | orig fw+bw ms | vs original |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ff_up_1024_h256` | 0.373 | 1.702 | 3.665 | 5.740 | 12.435 | 2.32x |
| `ff_down_1024_h256_weighted` | 0.369 | 2.544 | 3.836 | 6.749 | 16.196 | 2.54x |
| `ff_up_1024_h512` | 0.342 | 2.706 | 6.425 | 9.472 | 22.114 | 2.42x |
| `ff_down_1024_h512_weighted` | 0.270 | 3.393 | 6.839 | 10.502 | 26.252 | 2.57x |
| `ff_up_2048_h256` | 0.363 | 2.568 | 4.532 | 7.463 | 17.321 | 2.44x |
| `ff_down_2048_h256_weighted` | 0.313 | 2.719 | 4.705 | 7.738 | 18.942 | 2.55x |
| `ff_up_2048_h512` | 0.298 | 4.536 | 8.902 | 13.736 | 32.856 | 2.45x |
| `ff_down_2048_h512_weighted` | 0.265 | 4.537 | 8.550 | 13.351 | 32.588 | 2.49x |
| `ff_up_4096_h256_stress` | 0.316 | 3.022 | 5.802 | 9.140 | 24.645 | 2.79x |
| `ff_down_4096_h256_weighted_stress` | 0.253 | 3.313 | 5.519 | 9.085 | 23.934 | 2.71x |
| `ff_up_4096_h512_stress` | 0.255 | 5.200 | 10.418 | 15.873 | 47.290 | 3.03x |
| `ff_down_4096_h512_weighted_stress` | 0.223 | 5.410 | 9.983 | 15.617 | 44.059 | 2.86x |

### FFN stack

Measured on the same GPU on 2026-06-28 with the PG199 workaround disabled.
Rows are full top-k MoE FFN stacks: gate matmul, softmax/top-k, up projection,
GELU, weighted down projection/combine, and backward. Raw score columns are
milliseconds per iteration from `benchmark_ffn_moe.py` with 5 iterations.
Tutel was measured from Microsoft Tutel source commit `add1bf1`. `Tutel / CVMM`
is the total-time ratio, so lower is faster and `1.0x` is parity with the CVMM
FFN stack.

| Shape | CVMM fw ms | CVMM bw ms | CVMM total ms | Tutel fw ms | Tutel bw ms | Tutel total ms | Tutel / CVMM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ffn_1024` | 3.954 | 5.232 | 9.186 | 22.870 | 17.100 | 39.970 | 4.35x |
| `ffn_1024_h256` | 5.052 | 7.647 | 12.699 | 24.253 | 20.049 | 44.303 | 3.49x |
| `ffn_1024_h512` | 7.207 | 13.594 | 20.801 | 25.919 | 25.985 | 51.904 | 2.50x |
| `ffn_2048` | 4.131 | 5.871 | 10.001 | 18.580 | 17.724 | 36.304 | 3.63x |
| `ffn_2048_h256` | 5.894 | 9.195 | 15.090 | 19.529 | 23.663 | 43.192 | 2.86x |
| `ffn_2048_h512` | 9.628 | 17.623 | 27.250 | 22.948 | 34.973 | 57.921 | 2.13x |
| `ffn_768` | 2.402 | 2.945 | 5.348 | 8.934 | 6.436 | 15.371 | 2.87x |
| `ffn_4096_stress` | 4.122 | 6.530 | 10.652 | 10.900 | 17.389 | 28.289 | 2.66x |
| `ffn_4096_h256_stress` | 6.606 | 11.092 | 17.699 | 14.362 | 27.881 | 42.243 | 2.39x |
| `ffn_4096_h512_stress` | 11.029 | 20.269 | 31.298 | 21.226 | 49.606 | 70.832 | 2.26x |

### Framework MoE stack probes

Measured on the same GPU on 2026-06-28 with the PG199 CUDA-event workaround
enabled. Rows are smaller full MoE FFN stacks from `benchmark_framework_moe.py`
using 5 iterations; these shapes are intentionally small enough for DeepSpeed's
native dense-dispatch path and Megatron's local legacy MoE path to run.
`DeepSpeed / CVMM` and `Megatron / CVMM` are total-time ratios, so lower is
faster and `1.0x` is parity with CVMM.

| Shape | CVMM total ms | DeepSpeed total ms | DeepSpeed / CVMM | Megatron total ms | Megatron / CVMM |
| --- | ---: | ---: | ---: | ---: | ---: |
| `tiny_d128_h32_e16_k4` | 5.125 | 10.967 | 2.14x | 18.761 | 3.66x |
| `small_d256_h64_e32_k4` | 5.044 | 21.692 | 4.30x | 32.933 | 6.53x |
| `small_d512_h128_e64_k4` | 4.835 | 37.624 | 7.78x | 56.159 | 11.62x |

### RTX 3090 probes

Measured on an `NVIDIA GeForce RTX 3090` on 2026-06-28 with
`CUDA_VISIBLE_DEVICES=4`. The PG199 workaround was disabled because this GPU
does not have the broken CUDA event counters. All projection shapes fit in 24 GB
VRAM. Projection rows are from `benchmark_moe_shapes.py` with 8 iterations; FFN
stack rows are from `benchmark_ffn_moe.py cvmm 5`.

| Projection shape | sel ms | fw ms | bw ms | total ms |
| --- | ---: | ---: | ---: | ---: |
| `ff_up_1024` | 0.357 | 1.499 | 3.162 | 5.018 |
| `ff_down_1024_weighted` | 0.393 | 2.576 | 2.612 | 5.580 |
| `ff_up_1024_h256` | 0.359 | 2.223 | 4.735 | 7.317 |
| `ff_down_1024_h256_weighted` | 0.324 | 3.195 | 4.650 | 8.170 |
| `ff_up_1024_h512` | 0.308 | 3.899 | 7.919 | 12.126 |
| `ff_down_1024_h512_weighted` | 0.251 | 5.054 | 8.764 | 14.069 |
| `ff_up_2048` | 0.402 | 2.148 | 4.037 | 6.588 |
| `ff_down_2048_weighted` | 0.319 | 2.638 | 3.356 | 6.313 |
| `ff_up_2048_h256` | 0.370 | 3.842 | 6.377 | 10.590 |
| `ff_down_2048_h256_weighted` | 0.305 | 3.846 | 6.184 | 10.335 |
| `ff_up_2048_h512` | 0.334 | 6.452 | 11.440 | 18.226 |
| `ff_down_2048_h512_weighted` | 0.247 | 6.547 | 11.335 | 18.130 |
| `switch_v_1024` | 0.291 | 2.093 | 6.039 | 8.424 |
| `switch_o_1024_weighted` | 0.282 | 4.279 | 3.922 | 8.483 |
| `switch_v_2048_h32` | 0.256 | 4.276 | 12.210 | 16.742 |
| `switch_o_2048_h32_weighted` | 0.262 | 8.092 | 7.338 | 15.692 |
| `ff_up_768` | 0.274 | 0.694 | 1.524 | 2.491 |
| `switch_o_768_weighted` | 0.374 | 1.808 | 1.767 | 3.949 |
| `ff_up_4096_stress` | 0.423 | 2.645 | 4.520 | 7.588 |
| `ff_down_4096_weighted_stress` | 0.278 | 2.698 | 4.197 | 7.173 |
| `ff_up_4096_h256_stress` | 0.289 | 4.242 | 7.424 | 11.954 |
| `ff_down_4096_h256_weighted_stress` | 0.249 | 4.101 | 7.568 | 11.918 |
| `ff_up_4096_h512_stress` | 0.254 | 7.677 | 13.642 | 21.573 |
| `ff_down_4096_h512_weighted_stress` | 0.211 | 7.306 | 13.890 | 21.407 |

| FFN shape | CVMM fw ms | CVMM bw ms | CVMM total ms |
| --- | ---: | ---: | ---: |
| `ffn_1024` | 4.387 | 6.455 | 10.842 |
| `ffn_1024_h256` | 6.205 | 10.297 | 16.502 |
| `ffn_1024_h512` | 10.090 | 17.755 | 27.845 |
| `ffn_2048` | 5.006 | 7.724 | 12.731 |
| `ffn_2048_h256` | 8.090 | 12.798 | 20.887 |
| `ffn_2048_h512` | 13.499 | 23.003 | 36.502 |
| `ffn_768` | 2.434 | 2.996 | 5.430 |
| `ffn_4096_stress` | 5.441 | 8.604 | 14.045 |
| `ffn_4096_h256_stress` | 8.666 | 15.019 | 23.685 |
| `ffn_4096_h512_stress` | 15.619 | 27.328 | 42.947 |

The weighted down/O projections are now close to the dense-equivalent bound on
the main 1024-2048 shapes. The remaining large gap is mostly in the unweighted
up/V projections, especially the 2048 and 4096 stress cases.
