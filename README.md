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
`Tutel / CVMM` is the total-time ratio, so lower is faster and `1.0x` is parity
with the CVMM FFN stack.

| Shape | CVMM fw ms | CVMM bw ms | CVMM total ms | Tutel fw ms | Tutel bw ms | Tutel total ms | Tutel / CVMM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ffn_1024` | 3.954 | 5.232 | 9.186 | 22.870 | 17.100 | 39.970 | 4.35x |
| `ffn_2048` | 4.131 | 5.871 | 10.001 | 18.580 | 17.724 | 36.304 | 3.63x |
| `ffn_768` | 2.402 | 2.945 | 5.348 | 8.934 | 6.436 | 15.371 | 2.87x |
| `ffn_4096_stress` | 4.122 | 6.530 | 10.652 | 10.900 | 17.389 | 28.289 | 2.66x |

The weighted down/O projections are now close to the dense-equivalent bound on
the main 1024-2048 shapes. The remaining large gap is mostly in the unweighted
up/V projections, especially the 2048 and 4096 stress cases.
