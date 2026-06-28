# CVMM

Conditional vector-matrix multiplication kernels for MoE-style projections.

## Benchmarks

Measured on an `NVIDIA DRIVE-PG199-PROD` on 2026-06-27 with the PG199
CUDA-event workaround enabled in `benchmark_moe_shapes.py`.

The raw score columns are milliseconds per iteration from
`python3 benchmark_moe_shapes.py 8`.

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
| `switch_v_1024` | 0.337 | 1.427 | 5.836 | 7.600 | 1.51x | 1.82x | 78.5% | 1.14x |
| `switch_o_1024_weighted` | 0.292 | 4.263 | 3.879 | 8.434 | 1.93x | 0.93x | 228.3% | 4.71x |
| `switch_v_2048_h32` | 0.267 | 2.641 | 11.154 | 14.062 | 1.63x | 1.82x | 50.0% | 1.11x |
| `switch_o_2048_h32_weighted` | 0.256 | 7.074 | 7.286 | 14.616 | 1.69x | 1.03x | 168.5% | 5.12x |
| `ff_up_768` | 0.420 | 0.789 | 1.692 | 2.901 | 1.53x | 2.40x | 45.8% | 1.41x |
| `switch_o_768_weighted` | 0.377 | 1.796 | 1.803 | 3.977 | 1.92x | 1.14x | 133.8% | 3.32x |
| `ff_up_4096_stress` | 0.381 | 1.898 | 3.652 | 5.931 | 2.29x | 5.57x | 6.3% | 1.46x |
| `ff_down_4096_weighted_stress` | 0.300 | 2.192 | 3.348 | 5.840 | 2.55x | 2.12x | 6.1% | 2.90x |

MegaBlocks and Tutel are full MoE FFN systems rather than per-projection CVMM
primitive replacements. They should be compared with a separate FFN-level
benchmark covering routing/dispatch, up projection, activation, down projection,
and combine.

| Backend | Comparable unit | Local status | Blocker before timing |
| --- | --- | --- | --- |
| MegaBlocks 0.10.0 | Dropless MoE FFN (`dMoE` / `ParallelDroplessMLP`): token grouping, expert MLP, and scatter/combine. This can represent the MoE FFN path, not an isolated CVMM projection. | Not timed. Source import works with the isolated `/tmp/cvmm3_compare_vendor` install. | `mlp_impl="sparse"` is rejected by MegaBlocks with Triton >= 3.2; `mlp_impl="grouped"` needs `grouped_gemm`. `grouped_gemm==0.3.0` failed to build against `/usr/bin/nvcc` 12.0 and PyTorch 2.12.1+cu130, and the STK sparse smoke path hit an illegal memory access. |
| Tutel `add1bf1` | Full `moe_layer`: top-k routing, `fast_encode` dispatch, batched expert FFN, and `fast_decode` combine. This can represent the MoE FFN path, but not the SwitchHead projection-only cases. | Not timed. Source import works, but the CUDA extension is not installed. | Building `tutel_custom_kernel` failed because `nccl.h` is missing. Installing an NCCL development package matching the CUDA/PyTorch stack, or building a single-GPU/no-NCCL Tutel variant if supported, is needed before an FFN benchmark can run. |

The weighted down/O projections are now close to the dense-equivalent bound on
the main 1024-2048 shapes. The remaining large gap is mostly in the unweighted
up/V projections, especially the 2048 and 4096 stress cases.
