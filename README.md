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
| `ff_up_1024` | 0.411 | 1.249 | 2.467 | 4.127 | 2.02x | 2.46x | 45.5% | 1.29x |
| `ff_down_1024_weighted` | 0.337 | 2.038 | 2.267 | 4.642 | 2.71x | 0.94x | 40.2% | 2.78x |
| `ff_up_2048` | 0.429 | 1.658 | 2.841 | 4.928 | 2.20x | 2.77x | 24.4% | 1.36x |
| `ff_down_2048_weighted` | 0.350 | 2.266 | 2.699 | 5.315 | 2.60x | 1.09x | 22.9% | 2.83x |
| `switch_v_1024` | 0.378 | 1.483 | 4.258 | 6.119 | 1.91x | 1.44x | 78.5% | 1.42x |
| `switch_o_1024_weighted` | 0.273 | 4.003 | 3.802 | 8.078 | 2.01x | 0.89x | 228.3% | 4.92x |
| `switch_v_2048_h32` | 0.273 | 2.663 | 8.220 | 11.156 | 2.07x | 1.43x | 50.0% | 1.40x |
| `switch_o_2048_h32_weighted` | 0.252 | 7.100 | 7.232 | 14.584 | 1.69x | 1.03x | 168.5% | 5.13x |
| `ff_up_768` | 0.462 | 0.858 | 1.430 | 2.750 | 1.66x | 2.21x | 45.8% | 1.49x |
| `switch_o_768_weighted` | 0.437 | 1.841 | 1.758 | 4.036 | 1.92x | 1.14x | 133.8% | 3.27x |
| `ff_up_4096_stress` | 0.379 | 1.917 | 3.398 | 5.694 | 2.39x | 5.33x | 6.3% | 1.52x |
| `ff_down_4096_weighted_stress` | 0.308 | 2.180 | 3.320 | 5.808 | 2.57x | 2.10x | 6.1% | 2.92x |


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
| `ff_up_1024_h256` | 0.412 | 1.770 | 3.532 | 5.714 | 12.435 | 2.35x |
| `ff_down_1024_h256_weighted` | 0.386 | 2.572 | 3.740 | 6.698 | 16.196 | 2.57x |
| `ff_up_1024_h512` | 0.368 | 2.714 | 6.138 | 9.220 | 22.114 | 2.50x |
| `ff_down_1024_h512_weighted` | 0.317 | 3.466 | 6.800 | 10.583 | 26.252 | 2.56x |
| `ff_up_2048_h256` | 0.382 | 2.579 | 4.370 | 7.331 | 17.321 | 2.49x |
| `ff_down_2048_h256_weighted` | 0.329 | 2.728 | 4.674 | 7.731 | 18.942 | 2.56x |
| `ff_up_2048_h512` | 0.357 | 4.580 | 8.702 | 13.639 | 32.856 | 2.47x |
| `ff_down_2048_h512_weighted` | 0.266 | 4.564 | 8.647 | 13.477 | 32.588 | 2.47x |
| `ff_up_4096_h256_stress` | 0.327 | 3.101 | 5.718 | 9.146 | 24.645 | 2.79x |
| `ff_down_4096_h256_weighted_stress` | 0.272 | 3.360 | 5.456 | 9.088 | 23.934 | 2.71x |
| `ff_up_4096_h512_stress` | 0.321 | 5.386 | 10.184 | 15.891 | 47.290 | 3.04x |
| `ff_down_4096_h512_weighted_stress` | 0.223 | 5.400 | 10.000 | 15.623 | 44.059 | 2.86x |

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
| `ff_up_1024` | 0.352 | 1.531 | 3.138 | 5.021 |
| `ff_down_1024_weighted` | 0.306 | 2.183 | 2.555 | 5.044 |
| `ff_up_1024_h256` | 0.420 | 2.264 | 4.730 | 7.414 |
| `ff_down_1024_h256_weighted` | 0.271 | 3.092 | 4.480 | 7.843 |
| `ff_up_1024_h512` | 0.291 | 3.907 | 7.853 | 12.051 |
| `ff_down_1024_h512_weighted` | 0.250 | 4.987 | 8.697 | 13.934 |
| `ff_up_2048` | 0.380 | 2.160 | 3.915 | 6.455 |
| `ff_down_2048_weighted` | 0.282 | 2.586 | 3.227 | 6.095 |
| `ff_up_2048_h256` | 0.324 | 3.772 | 6.299 | 10.395 |
| `ff_down_2048_h256_weighted` | 0.265 | 3.748 | 6.049 | 10.062 |
| `ff_up_2048_h512` | 0.357 | 6.774 | 11.304 | 18.435 |
| `ff_down_2048_h512_weighted` | 0.242 | 6.581 | 11.279 | 18.102 |
| `switch_v_1024` | 0.298 | 2.112 | 5.906 | 8.316 |
| `switch_o_1024_weighted` | 0.253 | 4.249 | 3.817 | 8.319 |
| `switch_v_2048_h32` | 0.291 | 4.285 | 12.073 | 16.649 |
| `switch_o_2048_h32_weighted` | 0.231 | 8.191 | 7.460 | 15.882 |
| `ff_up_768` | 0.405 | 0.913 | 1.670 | 2.988 |
| `switch_o_768_weighted` | 0.376 | 1.832 | 1.666 | 3.874 |
| `ff_up_4096_stress` | 0.332 | 2.584 | 4.330 | 7.246 |
| `ff_down_4096_weighted_stress` | 0.320 | 2.728 | 4.070 | 7.118 |
| `ff_up_4096_h256_stress` | 0.299 | 4.277 | 7.445 | 12.021 |
| `ff_down_4096_h256_weighted_stress` | 0.224 | 4.102 | 7.545 | 11.871 |
| `ff_up_4096_h512_stress` | 0.274 | 7.890 | 13.514 | 21.678 |
| `ff_down_4096_h512_weighted_stress` | 0.245 | 7.367 | 13.860 | 21.472 |

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
