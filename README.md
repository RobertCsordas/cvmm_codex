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

| Shape | sel ms | fw ms | bw ms | total ms | vs original | vs dense |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ff_up_1024` | 0.508 | 1.235 | 2.817 | 4.560 | 1.85x | 2.68x |
| `ff_down_1024_weighted` | 0.356 | 2.049 | 2.505 | 4.911 | 2.57x | 1.00x |
| `ff_up_2048` | 0.472 | 1.655 | 3.314 | 5.441 | 1.99x | 3.05x |
| `ff_down_2048_weighted` | 0.382 | 2.238 | 2.988 | 5.608 | 2.46x | 1.15x |
| `switch_v_1024` | 0.373 | 1.441 | 6.508 | 8.321 | 1.38x | 2.00x |
| `switch_o_1024_weighted` | 0.285 | 4.141 | 3.873 | 8.300 | 1.96x | 0.92x |
| `switch_v_2048_h32` | 0.326 | 2.664 | 11.139 | 14.129 | 1.63x | 1.82x |
| `switch_o_2048_h32_weighted` | 0.289 | 7.109 | 7.345 | 14.743 | 1.68x | 1.03x |
| `ff_up_768` | 0.500 | 0.817 | 1.732 | 3.050 | 1.48x | 2.46x |
| `switch_o_768_weighted` | 0.442 | 1.877 | 1.840 | 4.160 | 1.86x | 1.18x |
| `ff_up_4096_stress` | 0.417 | 1.880 | 3.661 | 5.958 | 2.29x | 5.56x |
| `ff_down_4096_weighted_stress` | 0.337 | 2.174 | 3.406 | 5.916 | 2.54x | 2.13x |

The weighted down/O projections are now close to the dense-equivalent bound on
the main 1024-2048 shapes. The remaining large gap is mostly in the unweighted
up/V projections, especially the 2048 and 4096 stress cases.
