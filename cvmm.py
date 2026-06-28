from typing import Union, Optional
import os
import torch
from dataclasses import dataclass
import triton
import triton.language as tl
from packaging import version

# Based on https://github.com/openai/triton/blob/main/python/tutorials/03-matrix-multiplication.py
# torch.compile() fixes by Julian Büchel <jub@zurich.ibm.com>, based on https://github.com/pytorch/pytorch/issues/115344

@dataclass
class CVMMSel:
    raw_sel: torch.Tensor
    sel: torch.Tensor
    sel_index: torch.Tensor
    out_index: Optional[torch.Tensor] = None
    reduction_weight: Optional[torch.Tensor] = None

    def clone(self) -> 'CVMMSel':
        return CVMMSel(self.raw_sel, self.sel, self.sel_index, self.out_index, self.reduction_weight)


def cvmm_prepare_sel(sel: torch.Tensor, n_experts: int) -> CVMMSel:
    fsel = sel.flatten().to(torch.int32)
    ssel, sel_index = fsel.sort()
    return CVMMSel(fsel.view_as(sel), ssel.view_as(sel), sel_index, None)


def get_dtype():
    if not torch.is_autocast_enabled():
        return torch.float32
    return torch.get_autocast_gpu_dtype()


def dtype_to_type_id(dtype: torch.dtype):
    if dtype == torch.float32:
        return 0
    elif dtype == torch.float16:
        return 1
    elif dtype == torch.bfloat16:
        return 2

    raise ValueError("Unknown dtype")


cvmm_triton_call = None
cvmm_triton_into_call = None
cvmm_triton_accumulate_call = None
cvmm_triton_reduction_call = None
cvmm_group_routes_call = None
cvmm_grouped_backward_call = None
cvmm_grouped_backward_lowmem_call = None
cvmm_grouped_backward_weighted_w_call = None
cvmm_weighted_route_bwd_x_call = None
cvmm_reduction_tuned_shapes = set()
cvmm_use_lowmem_grouped_backward = os.environ.get("CVMM_DISABLE_LOWMEM_GROUPED_BACKWARD", "0").lower() not in ("1", "true", "yes", "on")
cvmm_lowmem_min_top_k = int(os.environ.get("CVMM_LOWMEM_MIN_TOP_K", "32"))
cvmm_lowmem_chunk_size = int(os.environ.get("CVMM_LOWMEM_CHUNK_SIZE", "0"))
cvmm_force_gather_x_grad_w = os.environ.get("CVMM_GATHER_X_GRAD_W", "0").lower() in ("1", "true", "yes", "on")
cvmm_gather_x_grad_w_max_top_k = int(os.environ.get("CVMM_GATHER_X_GRAD_W_MAX_TOP_K", "8"))



def _cvmm_expert_offsets(sorted_experts: torch.Tensor, n_experts: int) -> torch.Tensor:
    unique_experts, counts = torch.unique_consecutive(sorted_experts, return_counts=True)
    if unique_experts.numel() == n_experts:
        return counts.cumsum(0)

    expert_counts = torch.zeros((n_experts,), device=sorted_experts.device, dtype=counts.dtype)
    expert_counts[unique_experts.long()] = counts
    return expert_counts.cumsum(0)

def _cvmm_lowmem_chunk_size(top_k: int) -> int:
    chunk_size = cvmm_lowmem_chunk_size
    if chunk_size <= 0:
        chunk_size = top_k
    return min(top_k, chunk_size)

def create_kernels():
    global cvmm_backward_kernel3, cvmm_triton_call, cvmm_triton_into_call, cvmm_triton_accumulate_call, cvmm_triton_reduction_call, cvmm_group_routes_call, cvmm_grouped_backward_call, cvmm_grouped_backward_lowmem_call, cvmm_grouped_backward_weighted_w_call, cvmm_weighted_route_bwd_x_call

    if cvmm_triton_call is not None:
        return

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
            triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        ],
        key=['M', 'N', 'K', 'dtype_id', 'out_dtype_id', 'allow_tf32', 'FAST_SINGLE_EXPERT']
    )
    @triton.jit
    def cvmm_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr, index_ptr, sel_ptr, out_index_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,
        stride_bo, stride_bk, stride_bn,
        stride_cm, stride_cn,
        stride_index, stride_sel, stride_out_index,
        out_index_is_none: tl.constexpr,
        dtype_id: tl.constexpr, out_dtype_id: tl.constexpr, allow_tf32: tl.constexpr,
        ACCUMULATE_OUTPUT: tl.constexpr, FAST_SINGLE_EXPERT: tl.constexpr,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr
    ):
        """Kernel for computing the matmul C = A x B.
        A has shape (M, K), B has shape (K, N) and C has shape (M, N)
        """
        # -----------------------------------------------------------
        # Map program ids `pid` to the block of C it should compute.
        # This is done in a grouped ordering to promote L2 data reuse.
        # See above `L2 Cache Optimizations` section for details.
        pid = tl.program_id(axis=0)

        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_n = (pid % num_pid_in_group) // group_size_m

        pid_m = first_pid_m + (pid % group_size_m)

        sel_first = tl.load(sel_ptr + pid_m * BLOCK_SIZE_M * stride_sel)
        sel_last = tl.load(sel_ptr + (min((pid_m + 1) * BLOCK_SIZE_M, M) - 1) * stride_sel)
        if FAST_SINGLE_EXPERT:
            single_expert_block = sel_first == sel_last
        else:
            sel_all = tl.load(sel_ptr + stride_sel * ((pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M))

        for matrix_id in range(sel_first, sel_last + 1):
            # ----------------------------------------------------------
            # Create pointers for the first blocks of A and B.
            # We will advance this pointer as we move in the K direction
            # and accumulate
            # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
            # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
            # See above `Pointer Arithmetics` section for details
            offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
            offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

            remap_offs_am = tl.load(index_ptr + stride_index * offs_am)

            # Create offset pointers
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + (remap_offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b_ptrs = b_ptr + matrix_id * stride_bo + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

            # -----------------------------------------------------------
            # Iterate to compute a block of the C matrix.
            # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
            # of fp32 values for higher accuracy.
            # `accumulator` will be converted back to fp16 after the loop.
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                # Load the next block of A and B, generate a mask by checking the K dimension.
                # If it is out of bounds, set it to 0.
                a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
                b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_bn[None, :] < N), other=0.0)
                # We accumulate along the K dimension.

                # Triton was unhappy with passing dtypes as vars.
                if dtype_id == 1:
                    a = a.to(tl.float16)
                    b = b.to(tl.float16)
                elif dtype_id == 2:
                    a = a.to(tl.bfloat16)
                    b = b.to(tl.bfloat16)

                accumulator += tl.dot(a, b, allow_tf32=allow_tf32)

                # Advance the ptrs to the next K block.
                a_ptrs += BLOCK_SIZE_K * stride_ak
                b_ptrs += BLOCK_SIZE_K * stride_bk


            if out_dtype_id == 1:
                c = accumulator.to(tl.float16)
            elif out_dtype_id == 2:
                c = accumulator.to(tl.bfloat16)
            else:
                c = accumulator

            # -----------------------------------------------------------
            # Write back the block of the output matrix C with masks.
            offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

            if out_index_is_none:
                remap_offs_cm = remap_offs_am
            else:
                remap_offs_cm = tl.load(out_index_ptr + stride_out_index * offs_am)

            offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + stride_cm * remap_offs_cm[:, None] + stride_cn * offs_cn[None, :]
            if FAST_SINGLE_EXPERT:
                if single_expert_block:
                    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
                else:
                    sel_all = tl.load(sel_ptr + stride_sel * (offs_cm % M))
                    c_mask = ((offs_cm[:, None] < M) & (sel_all[:, None] == matrix_id)) & (offs_cn[None, :] < N)
            else:
                c_mask = ((offs_cm[:, None] < M) & (sel_all[:, None] == matrix_id)) & (offs_cn[None, :] < N)
            if ACCUMULATE_OUTPUT:
                c_prev = tl.load(c_ptrs, mask=c_mask, other=0.0)
                tl.store(c_ptrs, c_prev + c, mask=c_mask)
            else:
                tl.store(c_ptrs, c, mask=c_mask)


    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        ],
        key=['M', 'N', 'K', 'dtype_id', 'REDUCE_SIZE', 'allow_tf32']
    )
    @triton.jit
    def cvmm_reduction_kernel(
        a_ptr, b_ptr, c_ptr, index_ptr, sel_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bo, stride_bk, stride_bn,
        stride_cm, stride_cn,
        stride_index, stride_sel,
        dtype_id: tl.constexpr, allow_tf32: tl.constexpr, REDUCE_SIZE: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr
    ):
        pid = tl.program_id(axis=0)

        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_n = (pid % num_pid_in_group) // group_size_m
        pid_m = first_pid_m + (pid % group_size_m)

        sel_first = tl.load(sel_ptr + pid_m * BLOCK_SIZE_M * stride_sel)
        sel_last = tl.load(sel_ptr + (min((pid_m + 1) * BLOCK_SIZE_M, M) - 1) * stride_sel)
        sel_all = tl.load(sel_ptr + stride_sel * ((pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M))

        for matrix_id in range(sel_first, sel_last + 1):
            offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
            offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            remap_offs_am = tl.load(index_ptr + stride_index * offs_am)

            offs_k = tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + (remap_offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b_ptrs = b_ptr + matrix_id * stride_bo + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
                b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_bn[None, :] < N), other=0.0)
                if dtype_id == 1:
                    a = a.to(tl.float16)
                    b = b.to(tl.float16)
                elif dtype_id == 2:
                    a = a.to(tl.bfloat16)
                    b = b.to(tl.bfloat16)
                accumulator += tl.dot(a, b, allow_tf32=allow_tf32)
                a_ptrs += BLOCK_SIZE_K * stride_ak
                b_ptrs += BLOCK_SIZE_K * stride_bk

            offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            remap_offs_cm = remap_offs_am // REDUCE_SIZE
            c_ptrs = c_ptr + stride_cm * remap_offs_cm[:, None] + stride_cn * offs_cn[None, :]
            c_mask = ((offs_cm[:, None] < M) & (sel_all[:, None] == matrix_id)) & (offs_cn[None, :] < N)
            tl.atomic_add(c_ptrs, accumulator, mask=c_mask)


    @triton.jit
    def cvmm_group_routes_kernel(
        src_ptr, index_ptr, out_ptr,
        N_ROUTES: tl.constexpr, D: tl.constexpr, FAN_OUT: tl.constexpr,
        stride_src_m, stride_src_d, stride_index, stride_out_m, stride_out_d,
        BLOCK_ROUTES: tl.constexpr, BLOCK_D: tl.constexpr
    ):
        pid_m = tl.program_id(0)
        pid_d = tl.program_id(1)
        offs_m = pid_m * BLOCK_ROUTES + tl.arange(0, BLOCK_ROUTES)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        route_idx = tl.load(index_ptr + offs_m * stride_index, mask=offs_m < N_ROUTES, other=0)
        src_idx = route_idx // FAN_OUT
        vals = tl.load(
            src_ptr + src_idx[:, None] * stride_src_m + offs_d[None, :] * stride_src_d,
            mask=(offs_m[:, None] < N_ROUTES) & (offs_d[None, :] < D),
            other=0.0
        )
        tl.store(
            out_ptr + offs_m[:, None] * stride_out_m + offs_d[None, :] * stride_out_d,
            vals,
            mask=(offs_m[:, None] < N_ROUTES) & (offs_d[None, :] < D)
        )


    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_ROUTES': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
            triton.Config({'BLOCK_ROUTES': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
            triton.Config({'BLOCK_ROUTES': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
            triton.Config({'BLOCK_ROUTES': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        ],
        key=['M', 'N', 'K', 'dtype_id', 'out_dtype_id', 'allow_tf32']
    )
    @triton.jit
    def cvmm_grouped_bwd_x_kernel(
        grad_ptr, key_ptr, out_ptr, route_index_ptr, sel_ptr,
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, E: tl.constexpr,
        stride_gm, stride_gk,
        stride_ke, stride_kk, stride_kn,
        stride_om, stride_on,
        stride_route_index, stride_sel,
        dtype_id: tl.constexpr, out_dtype_id: tl.constexpr, allow_tf32: tl.constexpr,
        BLOCK_ROUTES: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_ROUTES + tl.arange(0, BLOCK_ROUTES)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = offs_m < M
        n_mask = offs_n < N
        expert_ids = tl.load(sel_ptr + offs_m * stride_sel, mask=m_mask, other=E)
        first_expert = tl.min(expert_ids)
        last_expert = tl.minimum(tl.max(expert_ids), E - 1)

        for expert in range(first_expert, last_expert + 1):
            row_mask = expert_ids == expert
            acc = tl.zeros((BLOCK_ROUTES, BLOCK_N), dtype=tl.float32)
            offs_k = tl.arange(0, BLOCK_K)
            grad_ptrs = grad_ptr + offs_m[:, None] * stride_gm + offs_k[None, :] * stride_gk
            key_ptrs = key_ptr + expert * stride_ke + offs_k[:, None] * stride_kk + offs_n[None, :] * stride_kn
            for k_start in range(0, tl.cdiv(K, BLOCK_K)):
                k_mask = offs_k < K - k_start * BLOCK_K
                g = tl.load(grad_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)
                w = tl.load(key_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
                if dtype_id == 1:
                    g = g.to(tl.float16)
                    w = w.to(tl.float16)
                elif dtype_id == 2:
                    g = g.to(tl.bfloat16)
                    w = w.to(tl.bfloat16)
                acc += tl.dot(g, w, allow_tf32=allow_tf32)
                grad_ptrs += BLOCK_K * stride_gk
                key_ptrs += BLOCK_K * stride_kk

            if out_dtype_id == 1:
                out_vals = acc.to(tl.float16)
            elif out_dtype_id == 2:
                out_vals = acc.to(tl.bfloat16)
            else:
                out_vals = acc
            route_idx = tl.load(route_index_ptr + offs_m * stride_route_index, mask=m_mask, other=0)
            out_ptrs = out_ptr + route_idx[:, None] * stride_om + offs_n[None, :] * stride_on
            tl.store(out_ptrs, out_vals, mask=row_mask[:, None] & n_mask[None, :])


    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_ROUTES': 32, 'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_ROUTES': 32, 'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_ROUTES': 32, 'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_ROUTES': 64, 'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
        ],
        key=['M', 'N', 'TOTAL_ROUTES', 'dtype_id', 'allow_tf32']
    )
    @triton.jit
    def cvmm_grouped_bwd_w_kernel(
        x_ptr, grad_ptr, out_ptr, offsets_ptr,
        M: tl.constexpr, N: tl.constexpr, TOTAL_ROUTES: tl.constexpr,
        stride_xr, stride_xm,
        stride_gr, stride_gn,
        stride_oe, stride_om, stride_on,
        dtype_id: tl.constexpr, allow_tf32: tl.constexpr,
        BLOCK_ROUTES: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        expert = tl.program_id(2)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        zero = tl.full((), 0, dtype=tl.int64)
        start = tl.load(offsets_ptr + expert - 1) if expert > 0 else zero
        end = tl.load(offsets_ptr + expert)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        route_start = start
        while route_start < end:
            offs_r = route_start + tl.arange(0, BLOCK_ROUTES)
            route_mask = offs_r < end
            x = tl.load(
                x_ptr + offs_r[None, :] * stride_xr + offs_m[:, None] * stride_xm,
                mask=route_mask[None, :] & (offs_m[:, None] < M),
                other=0.0
            )
            g = tl.load(
                grad_ptr + offs_r[:, None] * stride_gr + offs_n[None, :] * stride_gn,
                mask=route_mask[:, None] & (offs_n[None, :] < N),
                other=0.0
            )
            if dtype_id == 1:
                x = x.to(tl.float16)
                g = g.to(tl.float16)
            elif dtype_id == 2:
                x = x.to(tl.bfloat16)
                g = g.to(tl.bfloat16)
            acc += tl.dot(x, g, allow_tf32=allow_tf32)
            route_start += BLOCK_ROUTES

        out_ptrs = out_ptr + expert * stride_oe + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        tl.store(out_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_ROUTES': 32, 'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_ROUTES': 32, 'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_ROUTES': 32, 'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_ROUTES': 64, 'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
        ],
        key=['M', 'N', 'TOTAL_ROUTES', 'TOP_K', 'dtype_id', 'allow_tf32']
    )
    @triton.jit
    def cvmm_grouped_bwd_w_gather_x_kernel(
        x_ptr, grad_ptr, out_ptr, offsets_ptr, route_index_ptr,
        M: tl.constexpr, N: tl.constexpr, TOTAL_ROUTES: tl.constexpr, TOP_K: tl.constexpr,
        stride_xt, stride_xm,
        stride_gr, stride_gn,
        stride_oe, stride_om, stride_on,
        stride_route_index,
        dtype_id: tl.constexpr, allow_tf32: tl.constexpr,
        BLOCK_ROUTES: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        expert = tl.program_id(2)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        zero = tl.full((), 0, dtype=tl.int64)
        start = tl.load(offsets_ptr + expert - 1) if expert > 0 else zero
        end = tl.load(offsets_ptr + expert)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        route_start = start
        while route_start < end:
            offs_r = route_start + tl.arange(0, BLOCK_ROUTES)
            route_mask = offs_r < end
            route_ids = tl.load(route_index_ptr + offs_r * stride_route_index, mask=route_mask, other=0)
            token_ids = route_ids // TOP_K
            x = tl.load(
                x_ptr + token_ids[None, :] * stride_xt + offs_m[:, None] * stride_xm,
                mask=route_mask[None, :] & (offs_m[:, None] < M),
                other=0.0
            )
            g = tl.load(
                grad_ptr + offs_r[:, None] * stride_gr + offs_n[None, :] * stride_gn,
                mask=route_mask[:, None] & (offs_n[None, :] < N),
                other=0.0
            )
            if dtype_id == 1:
                x = x.to(tl.float16)
                g = g.to(tl.float16)
            elif dtype_id == 2:
                x = x.to(tl.bfloat16)
                g = g.to(tl.bfloat16)
            acc += tl.dot(x, g, allow_tf32=allow_tf32)
            route_start += BLOCK_ROUTES

        out_ptrs = out_ptr + expert * stride_oe + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        tl.store(out_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_ROUTES': 128, 'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
            triton.Config({'BLOCK_ROUTES': 32, 'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_ROUTES': 32, 'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_ROUTES': 32, 'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_ROUTES': 64, 'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
        ],
        key=['M', 'N', 'TOTAL_ROUTES', 'REDUCE_SIZE', 'dtype_id', 'allow_tf32']
    )
    @triton.jit
    def cvmm_grouped_bwd_w_weighted_kernel(
        x_ptr, grad_ptr, out_ptr, offsets_ptr, x_index_ptr, route_index_ptr, weight_ptr,
        M: tl.constexpr, N: tl.constexpr, TOTAL_ROUTES: tl.constexpr, REDUCE_SIZE: tl.constexpr,
        stride_xt, stride_xm,
        stride_gt, stride_gn,
        stride_oe, stride_om, stride_on,
        stride_x_index, stride_route_index, stride_weight,
        dtype_id: tl.constexpr, allow_tf32: tl.constexpr,
        BLOCK_ROUTES: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        expert = tl.program_id(2)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        zero = tl.full((), 0, dtype=tl.int64)
        start = tl.load(offsets_ptr + expert - 1) if expert > 0 else zero
        end = tl.load(offsets_ptr + expert)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        route_start = start
        while route_start < end:
            offs_r = route_start + tl.arange(0, BLOCK_ROUTES)
            route_mask = offs_r < end
            x_ids = tl.load(x_index_ptr + offs_r * stride_x_index, mask=route_mask, other=0)
            route_ids = tl.load(route_index_ptr + offs_r * stride_route_index, mask=route_mask, other=0)
            grad_ids = route_ids // REDUCE_SIZE
            x = tl.load(
                x_ptr + x_ids[None, :] * stride_xt + offs_m[:, None] * stride_xm,
                mask=route_mask[None, :] & (offs_m[:, None] < M),
                other=0.0
            )
            g = tl.load(
                grad_ptr + grad_ids[:, None] * stride_gt + offs_n[None, :] * stride_gn,
                mask=route_mask[:, None] & (offs_n[None, :] < N),
                other=0.0
            )
            weights = tl.load(weight_ptr + route_ids * stride_weight, mask=route_mask, other=0.0)
            if dtype_id == 1:
                x = x.to(tl.float16)
                g = g.to(tl.float16)
                weights = weights.to(tl.float16)
            elif dtype_id == 2:
                x = x.to(tl.bfloat16)
                g = g.to(tl.bfloat16)
                weights = weights.to(tl.bfloat16)
            g *= weights[:, None]
            acc += tl.dot(x, g, allow_tf32=allow_tf32)
            route_start += BLOCK_ROUTES

        out_ptrs = out_ptr + expert * stride_oe + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        tl.store(out_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


    def _prune_backward_configs(configs, named_args, **kwargs):
        k = named_args["K"]
        valid = []
        for config in configs:
            block_size_k = config.kwargs["BLOCK_SIZE_K"]
            k_blocks = config.kwargs["K_BLOCKS"]
            grid_y = (k + block_size_k * k_blocks - 1) // (block_size_k * k_blocks)
            if grid_y <= 65535:
                valid.append(config)
        return valid or [max(configs, key=lambda config: config.kwargs["K_BLOCKS"])]

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 512}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 256}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 128}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 64}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 32}, num_stages=4, num_warps=4),

            triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 512}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 128}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 64}, num_stages=4, num_warps=4),

            triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 512}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 512}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 128}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 128}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 64}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 64}, num_stages=4, num_warps=4),
        ],
        key=['M', 'N', 'K', 'out_dtype_id', 'allow_tf32', 'dtype_id', 'HAS_WEIGHT', 'REDUCE_SIZE'],
        prune_configs_by={"early_config_prune": _prune_backward_configs},
        reset_to_zero=['c_ptr']
    )
    @triton.jit
    def cvmm_backward_kernel3(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr, index_ptr, sel_ptr, out_index_ptr, weight_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_co, stride_cm, stride_cn,
        stride_index, stride_sel, stride_out_index, stride_weight,
        out_index_is_none: tl.constexpr,
        out_dtype_id: tl.constexpr, allow_tf32: tl.constexpr, dtype_id: tl.constexpr,
        HAS_WEIGHT: tl.constexpr, REDUCE_SIZE: tl.constexpr,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr, K_BLOCKS: tl.constexpr
    ):
        """Kernel for computing the matmul C = A x B.
        A has shape (M, K), B has shape (K, N) and C has shape (M, N)
        """
        # -----------------------------------------------------------
        # Map program ids `pid` to the block of C it should compute.
        # This is done in a grouped ordering to promote L2 data reuse.
        # See above `L2 Cache Optimizations` section for details.
        pid = tl.program_id(axis=0)
        k_block_id = tl.program_id(axis=1)

        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        # ----------------------------------------------------------
        # Create pointers for the first blocks of A and B.
        # We will advance this pointer as we move in the K direction
        # and accumulate
        # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
        # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
        # See above `Pointer Arithmetics` section for details
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

        # -----------------------------------------------------------
        # Iterate to compute a block of the C matrix.
        # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
        # of fp32 values for higher accuracy.
        # `accumulator` will be converted back to fp16 after the loop.

        a_ptrs_this = a_ptr + offs_am[:, None] * stride_am
        b_ptrs_this = b_ptr + offs_bn[None, :] * stride_bn

        # Kactual = end_i - start_i
        # Nblocks = (Kactual + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K

        # WORK_PER_WORKER = (Nblocks + K_BLOCKS - 1) // K_BLOCKS
        # WORK_PER_WORKER = WORK_PER_WORKER if WORK_PER_WORKER > MIN_WORK_SIZE else MIN_WORK_SIZE


        # # Kloop_start = (Kactual + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K

        # first_block_k = k_block_id * WORK_PER_WORKER
        # last_block_k = min((k_block_id+1) * WORK_PER_WORKER, Nblocks)

        block_start_index = k_block_id * BLOCK_SIZE_K * K_BLOCKS
        block_end_index = min(block_start_index + BLOCK_SIZE_K * K_BLOCKS, K) - 1

        first_mat = tl.load(sel_ptr + stride_sel * block_start_index)
        last_mat = tl.load(sel_ptr + stride_sel * block_end_index)


        for matrix_index in range(first_mat, last_mat + 1):
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

            start_i = block_start_index
            end_i = block_end_index + 1
            while start_i < end_i:
                middle = (start_i + end_i) // 2
                middle_matrix = tl.load(sel_ptr + middle * stride_sel)
                if middle_matrix < matrix_index:
                    start_i = middle + 1
                else:
                    end_i = middle


            # # Continue binary search: find the first matrix that is > matrix_index
            start_i2 = start_i
            end_i = block_end_index + 1
            while start_i2 < end_i:
                middle = (start_i2 + end_i) // 2
                middle_matrix = tl.load(sel_ptr + middle * stride_sel)
                if middle_matrix <= matrix_index:
                    start_i2 = middle + 1
                else:
                    end_i = middle

            end_i = start_i2

            count = end_i - start_i

            block_mem_indices_f_base = start_i  + tl.arange(0, BLOCK_SIZE_K)

            if count > 0:
                for k in range((count + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K):
                    # block_mem_indices = (k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)) % K
                    block_mem_indices_f = block_mem_indices_f_base + k * BLOCK_SIZE_K
                    block_mem_indices = block_mem_indices_f % K
                    a_index = tl.load(index_ptr + stride_index * block_mem_indices)
                    if out_index_is_none:
                        route_index = a_index
                    else:
                        route_index = tl.load(out_index_ptr + stride_out_index * block_mem_indices)
                    if HAS_WEIGHT:
                        b_index = route_index // REDUCE_SIZE
                    else:
                        b_index = route_index
                    sel_ok = block_mem_indices_f < end_i

                    a_ptrs = a_ptrs_this + a_index[None, :] * stride_ak
                    b_ptrs = b_ptrs_this + b_index[:, None] * stride_bk

                    # Load the next block of A and B, generate a mask by checking the K dimension.
                    # If it is out of bounds, set it to 0.
                    a = tl.load(a_ptrs, mask=sel_ok[None, :], other=0.0)
                    b = tl.load(b_ptrs, mask=sel_ok[:, None], other=0.0)

                    if dtype_id == 1:
                        a = a.to(tl.float16)
                        b = b.to(tl.float16)
                    elif dtype_id == 2:
                        a = a.to(tl.bfloat16)
                        b = b.to(tl.bfloat16)

                    if HAS_WEIGHT:
                        weights = tl.load(weight_ptr + route_index * stride_weight, mask=sel_ok, other=0.0)
                        if dtype_id == 1:
                            weights = weights.to(tl.float16)
                        elif dtype_id == 2:
                            weights = weights.to(tl.bfloat16)
                        b *= weights[:, None]

                    # We accumulate along the K dimension.
                    accumulator += tl.dot(a, b, allow_tf32=allow_tf32)

                if out_dtype_id == 1:
                    c = accumulator.to(tl.float16)
                elif out_dtype_id == 2:
                    c = accumulator.to(tl.bfloat16)
                else:
                    c = accumulator

                # -----------------------------------------------------------
                # Write back the block of the output matrix C with masks.
                offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                c_ptrs = c_ptr + stride_co * matrix_index + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
                c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
                # tl.store(c_ptrs, c, mask=c_mask)
                tl.atomic_add(c_ptrs, c, mask=c_mask)


    if version.parse(torch.__version__) >= version.parse("2.2.0"):
        torch.library.define("mylib::cvmm_triton", "(Tensor x, Tensor sel_index, Tensor sel, Tensor keys, ScalarType out_dtype, Tensor out_index, ScalarType op_dtype) -> Tensor")
        lib_decorator = torch.library.impl("mylib::cvmm_triton", "default")
    else:
        lib_decorator = lambda x: x

    @lib_decorator
    def cvmm_triton(
        x: torch.Tensor,
        sel_index: torch.Tensor,
        sel: torch.Tensor,
        keys: torch.Tensor,
        out_dtype: torch.dtype,
        out_index: torch.Tensor,
        op_dtype: torch.dtype
    ):
        x = x.flatten(end_dim=-2)
        assert x.shape[-1] == keys.shape[1]

        sel_shape = sel.shape
        sel = sel.flatten()

        M = sel.shape[0]
        O, K, N = keys.shape
        # Allocates output.
        out = torch.empty((M, N), device=x.device, dtype=out_dtype)
        # out = torch.zeros((M, N), device=x.device, dtype=out_dtype)
        # 1D launch kernel where each block gets its own program.

        # expected_m_per_matrix = int(math.ceil(M / O * 1.5))
        # expected_m_per_matrix = M

        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        )

        out_index_is_none = False
        if out_index.numel() == 1 and out_index == -1:
            out_index_is_none = True

        cvmm_kernel[grid](
            x, keys, out, sel_index, sel, out_index,
            M, N, K,
            x.stride(0), x.stride(1),
            keys.stride(0), keys.stride(1), keys.stride(2),
            out.stride(0), out.stride(1),
            sel_index.stride(0), sel.stride(0), 0 if out_index_is_none else out_index.stride(0),
            out_index_is_none=out_index_is_none,
            dtype_id=dtype_to_type_id(op_dtype),
            out_dtype_id=dtype_to_type_id(out.dtype),
            allow_tf32=False, #torch.backends.cuda.matmul.allow_tf32
            ACCUMULATE_OUTPUT=False,
            FAST_SINGLE_EXPERT=out_index_is_none and N <= 1024,
        )

        return out.view(*sel_shape, N)


    def cvmm_triton_into(
        x: torch.Tensor,
        sel_index: torch.Tensor,
        sel: torch.Tensor,
        keys: torch.Tensor,
        out: torch.Tensor,
        out_index: torch.Tensor,
        op_dtype: torch.dtype
    ):
        x = x.flatten(end_dim=-2)
        assert x.shape[-1] == keys.shape[1]
        sel = sel.flatten()
        M = sel.shape[0]
        O, K, N = keys.shape
        assert out.shape == (M, N)

        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        )

        out_index_is_none = False
        if out_index.numel() == 1 and out_index == -1:
            out_index_is_none = True

        cvmm_kernel[grid](
            x, keys, out, sel_index, sel, out_index,
            M, N, K,
            x.stride(0), x.stride(1),
            keys.stride(0), keys.stride(1), keys.stride(2),
            out.stride(0), out.stride(1),
            sel_index.stride(0), sel.stride(0), 0 if out_index_is_none else out_index.stride(0),
            out_index_is_none=out_index_is_none,
            dtype_id=dtype_to_type_id(op_dtype),
            out_dtype_id=dtype_to_type_id(out.dtype),
            allow_tf32=False, #torch.backends.cuda.matmul.allow_tf32
            ACCUMULATE_OUTPUT=False,
            FAST_SINGLE_EXPERT=out_index_is_none and N <= 1024,
        )


    def cvmm_triton_accumulate(
        x: torch.Tensor,
        sel_index: torch.Tensor,
        sel: torch.Tensor,
        keys: torch.Tensor,
        out: torch.Tensor,
        op_dtype: torch.dtype,
        accumulate: bool
    ):
        x = x.flatten(end_dim=-2)
        assert x.shape[-1] == keys.shape[1]

        sel = sel.flatten()
        M = sel.shape[0]
        O, K, N = keys.shape
        assert out.shape == (M, N)

        out_index = sel_index.new_tensor(-1)
        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        )

        cvmm_kernel[grid](
            x, keys, out, sel_index, sel, out_index,
            M, N, K,
            x.stride(0), x.stride(1),
            keys.stride(0), keys.stride(1), keys.stride(2),
            out.stride(0), out.stride(1),
            sel_index.stride(0), sel.stride(0), 0,
            out_index_is_none=True,
            dtype_id=dtype_to_type_id(op_dtype),
            out_dtype_id=dtype_to_type_id(out.dtype),
            allow_tf32=False, #torch.backends.cuda.matmul.allow_tf32
            ACCUMULATE_OUTPUT=accumulate,
            FAST_SINGLE_EXPERT=False,
        )


    def cvmm_triton_reduction(
        x: torch.Tensor,
        sel_index: torch.Tensor,
        sel: torch.Tensor,
        keys: torch.Tensor,
        op_dtype: torch.dtype,
        out_dtype: torch.dtype,
        reduce_size: int
    ):
        x = x.flatten(end_dim=-2)
        assert x.shape[-1] == keys.shape[1]

        sel_shape = sel.shape
        sel = sel.flatten()

        M = sel.shape[0]
        O, K, N = keys.shape
        assert M % reduce_size == 0

        out = torch.zeros((M // reduce_size, N), device=x.device, dtype=out_dtype)
        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        )

        dtype_id = dtype_to_type_id(op_dtype)
        tune_key = (x.device.index, M, N, K, dtype_id, reduce_size)
        if tune_key not in cvmm_reduction_tuned_shapes:
            scratch = torch.empty_like(out)
            cvmm_reduction_kernel[grid](
                x, keys, scratch, sel_index, sel,
                M, N, K,
                x.stride(0), x.stride(1),
                keys.stride(0), keys.stride(1), keys.stride(2),
                scratch.stride(0), scratch.stride(1),
                sel_index.stride(0), sel.stride(0),
                dtype_id=dtype_id,
                allow_tf32=False, #torch.backends.cuda.matmul.allow_tf32
                REDUCE_SIZE=reduce_size,
            )
            cvmm_reduction_tuned_shapes.add(tune_key)

        cvmm_reduction_kernel[grid](
            x, keys, out, sel_index, sel,
            M, N, K,
            x.stride(0), x.stride(1),
            keys.stride(0), keys.stride(1), keys.stride(2),
            out.stride(0), out.stride(1),
            sel_index.stride(0), sel.stride(0),
            dtype_id=dtype_id,
            allow_tf32=False, #torch.backends.cuda.matmul.allow_tf32
            REDUCE_SIZE=reduce_size,
        )

        return out.view(*sel_shape[:-1], N)


    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        ],
        key=['M', 'N', 'K', 'dtype_id', 'out_dtype_id', 'NEED_GRAD_X', 'NEED_GRAD_WEIGHT', 'allow_tf32'],
        reset_to_zero=['grad_weight_ptr']
    )
    @triton.jit
    def cvmm_weighted_route_bwd_x_kernel(
        grad_ptr, key_ptr, x_ptr, grad_x_ptr, grad_weight_ptr, route_index_ptr, sel_ptr, weight_ptr,
        M, N, K,
        stride_gt, stride_gk,
        stride_ke, stride_kk, stride_kn,
        stride_xr, stride_xn,
        stride_gxr, stride_gxn,
        stride_route_index, stride_sel, stride_weight, stride_grad_weight,
        dtype_id: tl.constexpr, out_dtype_id: tl.constexpr, allow_tf32: tl.constexpr, REDUCE_SIZE: tl.constexpr,
        NEED_GRAD_X: tl.constexpr, NEED_GRAD_WEIGHT: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr
    ):
        pid = tl.program_id(axis=0)

        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_n = (pid % num_pid_in_group) // group_size_m
        pid_m = first_pid_m + (pid % group_size_m)

        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        route_ids = tl.load(route_index_ptr + (offs_m % M) * stride_route_index)
        token_ids = route_ids // REDUCE_SIZE
        route_weight = tl.load(weight_ptr + route_ids * stride_weight)
        sel_first = tl.load(sel_ptr + pid_m * BLOCK_SIZE_M * stride_sel)
        sel_last = tl.load(sel_ptr + (min((pid_m + 1) * BLOCK_SIZE_M, M) - 1) * stride_sel)
        sel_all = tl.load(sel_ptr + stride_sel * (offs_m % M))

        for expert in range(sel_first, sel_last + 1):
            offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            grad_ptrs = grad_ptr + token_ids[:, None] * stride_gt + offs_k[None, :] * stride_gk
            key_ptrs = key_ptr + expert * stride_ke + offs_k[:, None] * stride_kk + offs_n[None, :] * stride_kn

            acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                k_mask = offs_k < K - k * BLOCK_SIZE_K
                g = tl.load(grad_ptrs, mask=k_mask[None, :], other=0.0)
                w = tl.load(key_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
                if dtype_id == 1:
                    g = g.to(tl.float16)
                    w = w.to(tl.float16)
                elif dtype_id == 2:
                    g = g.to(tl.bfloat16)
                    w = w.to(tl.bfloat16)
                acc += tl.dot(g, w, allow_tf32=allow_tf32)
                grad_ptrs += BLOCK_SIZE_K * stride_gk
                key_ptrs += BLOCK_SIZE_K * stride_kk

            route_mask = (offs_m[:, None] < M) & (sel_all[:, None] == expert) & (offs_n[None, :] < N)
            if NEED_GRAD_X:
                if dtype_id == 1:
                    weighted_grad = acc.to(tl.float16) * route_weight.to(tl.float16)[:, None]
                elif dtype_id == 2:
                    weighted_grad = acc.to(tl.bfloat16) * route_weight.to(tl.bfloat16)[:, None]
                else:
                    weighted_grad = acc * route_weight[:, None]

                if out_dtype_id == 1:
                    weighted_out = weighted_grad.to(tl.float16)
                elif out_dtype_id == 2:
                    weighted_out = weighted_grad.to(tl.bfloat16)
                else:
                    weighted_out = weighted_grad
                gx_ptrs = grad_x_ptr + route_ids[:, None] * stride_gxr + offs_n[None, :] * stride_gxn
                tl.store(gx_ptrs, weighted_out, mask=route_mask)

            if NEED_GRAD_WEIGHT:
                x_vals = tl.load(
                    x_ptr + route_ids[:, None] * stride_xr + offs_n[None, :] * stride_xn,
                    mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
                    other=0.0
                )
                if dtype_id == 1:
                    gate_vals = acc.to(tl.float16).to(tl.float32) * x_vals.to(tl.float32)
                elif dtype_id == 2:
                    gate_vals = acc.to(tl.bfloat16).to(tl.float32) * x_vals.to(tl.float32)
                else:
                    gate_vals = acc * x_vals.to(tl.float32)
                gate_part = tl.sum(tl.where(route_mask, gate_vals, 0.0), axis=1)
                tl.atomic_add(grad_weight_ptr + route_ids * stride_grad_weight, gate_part, mask=offs_m < M)


    def cvmm_group_routes(src: torch.Tensor, route_index: torch.Tensor, fan_out: int):
        src = src.flatten(end_dim=-2)
        route_index = route_index.flatten()
        n_routes = route_index.numel()
        d = src.shape[-1]
        out = torch.empty((n_routes, d), device=src.device, dtype=src.dtype)
        grid = (triton.cdiv(n_routes, 256), triton.cdiv(d, 128))
        cvmm_group_routes_kernel[grid](
            src, route_index, out,
            N_ROUTES=n_routes, D=d, FAN_OUT=fan_out,
            stride_src_m=src.stride(0), stride_src_d=src.stride(1),
            stride_index=route_index.stride(0),
            stride_out_m=out.stride(0), stride_out_d=out.stride(1),
            BLOCK_ROUTES=256, BLOCK_D=128,
        )
        return out


    def cvmm_grouped_backward(
        x: torch.Tensor,
        sel_index: torch.Tensor,
        sel: torch.Tensor,
        grads: torch.Tensor,
        keys: torch.Tensor,
        n_experts: int,
        key_dtype: torch.dtype,
        op_dtype: torch.dtype,
        top_k: int,
        need_grad_x: bool,
        need_grad_w: bool
    ):
        x_flat = x.flatten(end_dim=-2)
        grads_flat = grads.flatten(end_dim=-2)
        sel_flat = sel.flatten()
        total_routes = sel_flat.numel()
        assert total_routes == grads_flat.shape[0]

        offsets = _cvmm_expert_offsets(sel_flat, n_experts)
        grouped_grads = cvmm_group_routes(grads_flat, sel_index, 1)

        grouped_x = None
        grad_w = None
        if need_grad_w:
            M = x_flat.shape[-1]
            N = grads_flat.shape[-1]
            grad_w = torch.empty((n_experts, M, N), device=x.device, dtype=torch.float32)
            grid = lambda META: (
                triton.cdiv(M, META['BLOCK_M']),
                triton.cdiv(N, META['BLOCK_N']),
                n_experts,
            )
            use_gather_x_grad_w = cvmm_force_gather_x_grad_w or (cvmm_gather_x_grad_w_max_top_k > 0 and top_k <= cvmm_gather_x_grad_w_max_top_k)
            if use_gather_x_grad_w:
                cvmm_grouped_bwd_w_gather_x_kernel[grid](
                    x_flat, grouped_grads, grad_w, offsets, sel_index,
                    M, N, total_routes, top_k,
                    x_flat.stride(0), x_flat.stride(1),
                    grouped_grads.stride(0), grouped_grads.stride(1),
                    grad_w.stride(0), grad_w.stride(1), grad_w.stride(2),
                    sel_index.stride(0),
                    dtype_id=dtype_to_type_id(op_dtype),
                    allow_tf32=False, #torch.backends.cuda.matmul.allow_tf32
                )
            else:
                grouped_x = cvmm_group_routes(x_flat, sel_index, top_k)
                cvmm_grouped_bwd_w_kernel[grid](
                    grouped_x, grouped_grads, grad_w, offsets,
                    M, N, total_routes,
                    grouped_x.stride(0), grouped_x.stride(1),
                    grouped_grads.stride(0), grouped_grads.stride(1),
                    grad_w.stride(0), grad_w.stride(1), grad_w.stride(2),
                    dtype_id=dtype_to_type_id(op_dtype),
                    allow_tf32=False, #torch.backends.cuda.matmul.allow_tf32
                )
            if key_dtype != torch.float32:
                grad_w = grad_w.to(key_dtype)

        grad_x = None
        if need_grad_x:
            if grouped_x is None:
                grouped_x = torch.empty((total_routes, x_flat.shape[-1]), device=x.device, dtype=x.dtype)
            keys_t = keys.transpose(1, 2)
            M = total_routes
            N = x_flat.shape[-1]
            K = grads_flat.shape[-1]
            grid = lambda META: (
                triton.cdiv(M, META['BLOCK_ROUTES']),
                triton.cdiv(N, META['BLOCK_N']),
            )
            cvmm_grouped_bwd_x_kernel[grid](
                grouped_grads, keys_t, grouped_x, sel_index, sel_flat,
                M, N, K, n_experts,
                grouped_grads.stride(0), grouped_grads.stride(1),
                keys_t.stride(0), keys_t.stride(1), keys_t.stride(2),
                grouped_x.stride(0), grouped_x.stride(1),
                sel_index.stride(0), sel_flat.stride(0),
                dtype_id=dtype_to_type_id(op_dtype),
                out_dtype_id=dtype_to_type_id(grouped_x.dtype),
                allow_tf32=False, #torch.backends.cuda.matmul.allow_tf32
            )
            grad_x = grouped_x.view(*x.shape[:-1], top_k, x.shape[-1]).sum(-2).to(x.dtype).view_as(x)

        return grad_x, grad_w


    def cvmm_grouped_backward_lowmem(
        x: torch.Tensor,
        sel_index: torch.Tensor,
        sel: torch.Tensor,
        grads: torch.Tensor,
        keys: torch.Tensor,
        n_experts: int,
        key_dtype: torch.dtype,
        op_dtype: torch.dtype,
        top_k: int,
        need_grad_x: bool,
        need_grad_w: bool
    ):
        x_flat = x.flatten(end_dim=-2)
        grads_flat = grads.flatten(end_dim=-2)
        sel_flat = sel.flatten()
        total_routes = sel_flat.numel()
        assert total_routes == grads_flat.shape[0]

        grad_w = None
        if need_grad_w:
            offsets = _cvmm_expert_offsets(sel_flat, n_experts)
            grouped_grads = cvmm_group_routes(grads_flat, sel_index, 1)
            M = x_flat.shape[-1]
            N = grads_flat.shape[-1]
            grad_w = torch.empty((n_experts, M, N), device=x.device, dtype=torch.float32)
            grid = lambda META: (
                triton.cdiv(M, META['BLOCK_M']),
                triton.cdiv(N, META['BLOCK_N']),
                n_experts,
            )
            cvmm_grouped_bwd_w_gather_x_kernel[grid](
                x_flat, grouped_grads, grad_w, offsets, sel_index,
                M, N, total_routes, top_k,
                x_flat.stride(0), x_flat.stride(1),
                grouped_grads.stride(0), grouped_grads.stride(1),
                grad_w.stride(0), grad_w.stride(1), grad_w.stride(2),
                sel_index.stride(0),
                dtype_id=dtype_to_type_id(op_dtype),
                allow_tf32=False, #torch.backends.cuda.matmul.allow_tf32
            )
            if key_dtype != torch.float32:
                grad_w = grad_w.to(key_dtype)

        grad_x = None
        if need_grad_x:
            tokens = x_flat.shape[0]
            chunk_size = _cvmm_lowmem_chunk_size(top_k)
            keys_t = keys.transpose(1, 2)
            N = x_flat.shape[-1]
            K = grads_flat.shape[-1]

            if chunk_size >= top_k:
                grouped_grad_chunk = cvmm_group_routes(grads_flat, sel_index, 1)
                chunk_grad_x = torch.empty((total_routes, N), device=x.device, dtype=x.dtype)
                M = total_routes
                grid = lambda META: (
                    triton.cdiv(M, META['BLOCK_ROUTES']),
                    triton.cdiv(N, META['BLOCK_N']),
                )
                cvmm_grouped_bwd_x_kernel[grid](
                    grouped_grad_chunk, keys_t, chunk_grad_x, sel_index, sel_flat,
                    M, N, K, n_experts,
                    grouped_grad_chunk.stride(0), grouped_grad_chunk.stride(1),
                    keys_t.stride(0), keys_t.stride(1), keys_t.stride(2),
                    chunk_grad_x.stride(0), chunk_grad_x.stride(1),
                    sel_index.stride(0), sel_flat.stride(0),
                    dtype_id=dtype_to_type_id(op_dtype),
                    out_dtype_id=dtype_to_type_id(chunk_grad_x.dtype),
                    allow_tf32=False, #torch.backends.cuda.matmul.allow_tf32
                )
                grad_x = chunk_grad_x.view(tokens, top_k, N).sum(1).view_as(x)
            else:
                grad_x = torch.zeros_like(x_flat)
                raw_sel = torch.empty_like(sel_flat)
                raw_sel[sel_index] = sel_flat
                raw_sel = raw_sel.view(tokens, top_k)
                grad_routes = grads_flat.view(tokens, top_k, grads_flat.shape[-1])

                for chunk_start in range(0, top_k, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, top_k)
                    route_count = chunk_end - chunk_start
                    sel_chunk = raw_sel[:, chunk_start:chunk_end].reshape(-1)
                    sel_chunk, sel_chunk_index = sel_chunk.sort()
                    grad_chunk = grad_routes[:, chunk_start:chunk_end, :].reshape(-1, grads_flat.shape[-1])
                    grouped_grad_chunk = cvmm_group_routes(grad_chunk, sel_chunk_index, 1)
                    chunk_routes = sel_chunk.numel()
                    chunk_grad_x = torch.empty((chunk_routes, N), device=x.device, dtype=x.dtype)
                    M = chunk_routes
                    grid = lambda META: (
                        triton.cdiv(M, META['BLOCK_ROUTES']),
                        triton.cdiv(N, META['BLOCK_N']),
                    )
                    cvmm_grouped_bwd_x_kernel[grid](
                        grouped_grad_chunk, keys_t, chunk_grad_x, sel_chunk_index, sel_chunk,
                        M, N, K, n_experts,
                        grouped_grad_chunk.stride(0), grouped_grad_chunk.stride(1),
                        keys_t.stride(0), keys_t.stride(1), keys_t.stride(2),
                        chunk_grad_x.stride(0), chunk_grad_x.stride(1),
                        sel_chunk_index.stride(0), sel_chunk.stride(0),
                        dtype_id=dtype_to_type_id(op_dtype),
                        out_dtype_id=dtype_to_type_id(chunk_grad_x.dtype),
                        allow_tf32=False, #torch.backends.cuda.matmul.allow_tf32
                    )
                    grad_x += chunk_grad_x.view(tokens, route_count, N).sum(1)

                grad_x = grad_x.view_as(x)

        return grad_x, grad_w


    def cvmm_grouped_backward_weighted_w(
        x: torch.Tensor,
        x_index: torch.Tensor,
        route_index: torch.Tensor,
        sel: torch.Tensor,
        grads: torch.Tensor,
        reduction_weight: torch.Tensor,
        n_experts: int,
        key_dtype: torch.dtype,
        op_dtype: torch.dtype,
        reduce_size: int,
    ):
        x_flat = x.flatten(end_dim=-2)
        grads_flat = grads.flatten(end_dim=-2)
        sel_flat = sel.flatten()
        x_index = x_index.flatten()
        route_index = route_index.flatten()
        reduction_weight = reduction_weight.flatten()
        total_routes = sel_flat.numel()
        assert x_index.numel() == total_routes
        assert route_index.numel() == total_routes
        assert reduction_weight.numel() == total_routes

        offsets = _cvmm_expert_offsets(sel_flat, n_experts)
        M = x_flat.shape[-1]
        N = grads_flat.shape[-1]
        grad_w = torch.empty((n_experts, M, N), device=x.device, dtype=torch.float32)
        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_M']),
            triton.cdiv(N, META['BLOCK_N']),
            n_experts,
        )
        cvmm_grouped_bwd_w_weighted_kernel[grid](
            x_flat, grads_flat, grad_w, offsets, x_index, route_index, reduction_weight,
            M, N, total_routes, reduce_size,
            x_flat.stride(0), x_flat.stride(1),
            grads_flat.stride(0), grads_flat.stride(1),
            grad_w.stride(0), grad_w.stride(1), grad_w.stride(2),
            x_index.stride(0), route_index.stride(0), reduction_weight.stride(0),
            dtype_id=dtype_to_type_id(op_dtype),
            allow_tf32=False, #torch.backends.cuda.matmul.allow_tf32
        )
        return grad_w if key_dtype == torch.float32 else grad_w.to(key_dtype)


    def cvmm_weighted_route_bwd_x(
        x: torch.Tensor,
        sel_index: torch.Tensor,
        sel: torch.Tensor,
        grads: torch.Tensor,
        keys: torch.Tensor,
        reduction_weight: torch.Tensor,
        op_dtype: torch.dtype,
        need_grad_x: bool,
        need_grad_weight: bool,
    ):
        x_flat = x.flatten(end_dim=-2)
        grads_flat = grads.flatten(end_dim=-2)
        keys_t = keys.transpose(1, 2)
        sel_flat = sel.flatten()
        route_index = sel_index.flatten()
        reduction_weight_flat = reduction_weight.flatten()
        total_routes = sel_flat.numel()
        assert x_flat.shape[0] == total_routes
        assert route_index.numel() == total_routes
        assert reduction_weight_flat.numel() == total_routes

        M = total_routes
        N = x_flat.shape[-1]
        K = grads_flat.shape[-1]
        grad_x = torch.empty_like(x_flat) if need_grad_x else x_flat.new_empty((1, 1))
        grad_weight = torch.zeros((M,), device=x.device, dtype=torch.float32) if need_grad_weight else x_flat.new_empty((1,), dtype=torch.float32)
        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        )
        cvmm_weighted_route_bwd_x_kernel[grid](
            grads_flat, keys_t, x_flat, grad_x, grad_weight, route_index, sel_flat, reduction_weight_flat,
            M, N, K,
            grads_flat.stride(0), grads_flat.stride(1),
            keys_t.stride(0), keys_t.stride(1), keys_t.stride(2),
            x_flat.stride(0), x_flat.stride(1),
            grad_x.stride(0), grad_x.stride(1),
            route_index.stride(0), sel_flat.stride(0), reduction_weight_flat.stride(0), grad_weight.stride(0),
            dtype_id=dtype_to_type_id(op_dtype),
            out_dtype_id=dtype_to_type_id(x.dtype),
            allow_tf32=False, #torch.backends.cuda.matmul.allow_tf32
            REDUCE_SIZE=reduction_weight.shape[-1],
            NEED_GRAD_X=need_grad_x,
            NEED_GRAD_WEIGHT=need_grad_weight,
        )
        grad_x_out = grad_x.view_as(x) if need_grad_x else None
        grad_weight_out = grad_weight.view_as(reduction_weight).to(reduction_weight.dtype) if need_grad_weight else None
        return grad_x_out, grad_weight_out


    if version.parse(torch.__version__) >= version.parse("2.2.0"):
        if version.parse(triton.__version__) >= version.parse("3.0.0"):
            decorator = torch.library.register_fake
        else:
            decorator = torch.library.impl_abstract

        @decorator("mylib::cvmm_triton", cvmm_triton)
        def cvmm_triton_abstract(x, sel_idx, sel, keys, out_dtype, out_index, op_dtype):
            sel_shape = sel.shape
            sel = sel.flatten()
            M = sel.shape[0]
            O, K, N = keys.shape
            out = torch.empty((M, N), device=x.device, dtype=out_dtype)
            sel_shape = sel.shape
            return out.view(*sel_shape, N)


    if version.parse(torch.__version__) >= version.parse("2.2.0"):
        cvmm_triton_call = torch.ops.mylib.cvmm_triton
    else:
        cvmm_triton_call = cvmm_triton
    cvmm_triton_into_call = cvmm_triton_into
    cvmm_triton_accumulate_call = cvmm_triton_accumulate
    cvmm_triton_reduction_call = cvmm_triton_reduction
    cvmm_group_routes_call = cvmm_group_routes
    cvmm_grouped_backward_call = cvmm_grouped_backward
    cvmm_grouped_backward_lowmem_call = cvmm_grouped_backward_lowmem
    cvmm_grouped_backward_weighted_w_call = cvmm_grouped_backward_weighted_w
    cvmm_weighted_route_bwd_x_call = cvmm_weighted_route_bwd_x

# torch.library.define("mylib::cvmm_triton_backward", "(Tensor x, Tensor sel_index, Tensor sel, Tensor grads, int n_experts, ScalarType key_dtype, bool op_float16, Tensor out_index) -> Tensor")

# @torch.library.impl("mylib::cvmm_triton_backward", "default")
def cvmm_triton_backward(
    x: torch.Tensor,
    sel_index: torch.Tensor,
    sel: torch.Tensor,
    grads: torch.Tensor,
    n_experts: int,
    key_dtype: torch.dtype,
    op_dtype: torch.dtype,
    out_index: torch.Tensor,
    reduction_weight: Optional[torch.Tensor] = None,
    reduce_size: int = 1
):
    x = x.flatten(end_dim=-2)
    x = x.transpose(0, 1)
    grads = grads.flatten(end_dim=-2)
    sel = sel.flatten()
    M, _ = x.shape
    grad_rows, N = grads.shape
    K = sel.shape[0]
    has_weight = reduction_weight is not None
    if not has_weight:
        assert grad_rows == K
        reduction_weight = grads.new_empty((1,))
    else:
        reduction_weight = reduction_weight.flatten()
        assert reduction_weight.numel() == K
    out = torch.zeros((n_experts, M, N), device=x.device, dtype=torch.float32)
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), triton.cdiv(K, META['BLOCK_SIZE_K'] * META['K_BLOCKS'])
    )
    out_index_is_none = False
    if out_index.numel() == 1 and out_index == -1:
        out_index_is_none = True

    cvmm_backward_kernel3[grid](
        x, grads, out, sel_index, sel, out_index, reduction_weight,
        M, N, K,
        x.stride(0), x.stride(1),
        grads.stride(0), grads.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        sel_index.stride(0), sel.stride(0), 0 if out_index_is_none else out_index.stride(0), reduction_weight.stride(0),
        out_index_is_none=out_index_is_none,
        out_dtype_id=dtype_to_type_id(out.dtype),
        dtype_id=dtype_to_type_id(op_dtype),
        HAS_WEIGHT=has_weight,
        REDUCE_SIZE=reduce_size,
        allow_tf32=False #torch.backends.cuda.matmul.allow_tf32
    )
    return out if key_dtype == torch.float32 else out.to(key_dtype)


class CVMM(torch.autograd.Function):
    warned = False

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        raw_sel: torch.Tensor,
        sel_index: torch.Tensor,
        sel: torch.Tensor,
        keys: torch.Tensor,
        out_index: Optional[torch.Tensor] = None,
        reduction_weight: Optional[torch.Tensor] = None
    ):
        ctx.save_for_backward(x, keys, raw_sel, sel, sel_index, out_index, reduction_weight)

        out_type = get_dtype()
        if out_index is None:
            out_index = torch.tensor(-1, device=x.device)

        res = cvmm_triton_call(x, sel_index, sel, keys, out_type, out_index, out_type)

        if reduction_weight is not None:
            res = res.view(*reduction_weight.shape, res.shape[-1])
            res = (reduction_weight.unsqueeze(-2).type_as(res) @ res).squeeze(-2)

        ctx.op_type = out_type
        ctx.keys_type = keys.dtype
        ctx.dtype = out_type
        return res

    @staticmethod
    def backward(ctx, grad_output):
        x, keys, raw_sel, sel, sel_index, out_index, reduction_weight = ctx.saved_tensors
        keys_dt = keys

        need_grad_x = ctx.needs_input_grad[0]
        need_grad_keys = ctx.needs_input_grad[4]
        need_grad_reduction_weight = ctx.needs_input_grad[6] and reduction_weight is not None

        out_index_is_none = False
        if out_index is None:
            out_index_is_none = True
            out_index = torch.tensor(-1, device=x.device)

        top_k = raw_sel.shape[-1] if raw_sel.ndim > 0 else 1
        grad_w = None
        grad_x = None
        grad_w_off = None

        can_grouped_backward = (
            reduction_weight is None and
            not out_index_is_none and
            top_k > 1 and
            (need_grad_x or need_grad_keys)
        )
        use_lowmem_grouped_backward = False
        if can_grouped_backward:
            total_routes = sel.numel()
            op_bytes = torch.finfo(ctx.op_type).bits // 8
            use_lowmem_grouped_backward = (
                cvmm_use_lowmem_grouped_backward and
                need_grad_x and
                top_k >= cvmm_lowmem_min_top_k
            )
            if use_lowmem_grouped_backward:
                tokens = total_routes // top_k
                chunk_routes = tokens * _cvmm_lowmem_chunk_size(top_k)
                grouped_bytes = 0
                if need_grad_keys:
                    grouped_bytes += total_routes * grad_output.shape[-1] * op_bytes
                    grouped_bytes += keys_dt.numel() * torch.finfo(torch.float32).bits // 8
                if need_grad_x:
                    grouped_bytes += x.numel() * x.element_size()
                    grouped_bytes += total_routes * sel.element_size()
                    grouped_bytes += chunk_routes * (x.shape[-1] + grad_output.shape[-1]) * op_bytes
                    grouped_bytes += chunk_routes * (sel.element_size() + sel_index.element_size())
            else:
                grouped_bytes = total_routes * (x.shape[-1] + grad_output.shape[-1]) * op_bytes
                if need_grad_keys:
                    grouped_bytes += keys_dt.numel() * torch.finfo(torch.float32).bits // 8
                if need_grad_x:
                    grouped_bytes += total_routes * out_index.element_size()

            free_vram, total_vram = torch.cuda.mem_get_info(x.device)
            reserve_bytes = max(total_vram // 20, 512 * 1024 * 1024)
            scratch_limit = max(free_vram - reserve_bytes, 0) // 2
            if grouped_bytes <= scratch_limit:
                can_grouped_backward = True
            else:
                allocator_slack = torch.cuda.memory_reserved(x.device) - torch.cuda.memory_allocated(x.device)
                available_vram = free_vram + max(allocator_slack, 0)
                scratch_limit = max(available_vram - reserve_bytes, 0) // 2
                can_grouped_backward = grouped_bytes <= scratch_limit

        if can_grouped_backward:
            grouped_backward_call = cvmm_grouped_backward_lowmem_call if use_lowmem_grouped_backward else cvmm_grouped_backward_call
            try:
                grad_x, grad_w = grouped_backward_call(
                    x,
                    out_index,
                    sel,
                    grad_output,
                    keys_dt,
                    keys_dt.shape[0],
                    ctx.keys_type,
                    ctx.dtype,
                    top_k,
                    need_grad_x,
                    need_grad_keys
                )
            except torch.OutOfMemoryError:
                grad_x = None
                grad_w = None

        if need_grad_keys and grad_w is None and reduction_weight is not None:
            route_index = sel_index if out_index_is_none else out_index
            try:
                grad_w = cvmm_grouped_backward_weighted_w_call(
                    x,
                    sel_index,
                    route_index,
                    sel,
                    grad_output,
                    reduction_weight,
                    keys_dt.shape[0],
                    ctx.keys_type,
                    ctx.dtype,
                    reduction_weight.shape[-1],
                )
            except torch.OutOfMemoryError:
                grad_w = None

        if need_grad_keys and grad_w is None:
            # Backward for weight
            grad_w = cvmm_triton_backward(
                x,
                sel_index,
                sel,
                grad_output,
                keys_dt.shape[0],
                ctx.keys_type,
                ctx.dtype,
                out_index=out_index,
                reduction_weight=reduction_weight,
                reduce_size=1 if reduction_weight is None else reduction_weight.shape[-1]
            )

        need_weighted_x_work = (need_grad_x and grad_x is None) or (need_grad_reduction_weight and grad_w_off is None)
        if need_weighted_x_work and reduction_weight is not None and out_index_is_none and x.shape[:-1] == reduction_weight.shape:
            try:
                grad_x, grad_w_off = cvmm_weighted_route_bwd_x_call(
                    x,
                    sel_index,
                    sel,
                    grad_output,
                    keys_dt,
                    reduction_weight,
                    ctx.op_type,
                    need_grad_x,
                    need_grad_reduction_weight,
                )
            except torch.OutOfMemoryError:
                grad_x = None
                grad_w_off = None

        need_weighted_x_work = (need_grad_x and grad_x is None) or (need_grad_reduction_weight and grad_w_off is None)
        if need_weighted_x_work:
            bw_index = sel_index if out_index_is_none else out_index
            bw_index_out = torch.tensor(-1, device=x.device)
            if reduction_weight is not None:
                # Hack the output indices to emulate repeats.
                bw_index_out = bw_index
                bw_index = bw_index // reduction_weight.shape[-1]

            can_reduce_routes = reduction_weight is None and need_grad_x and not out_index_is_none and top_k > 1
            grad_x_elements = x.numel()
            use_atomic_reduction = can_reduce_routes and grad_x_elements <= 16 * 1024 * 1024
            use_route_accumulation = can_reduce_routes and not use_atomic_reduction and x.shape[-1] <= 64

            if use_atomic_reduction:
                grad_x = cvmm_triton_reduction_call(
                    grad_output,
                    bw_index,
                    sel,
                    keys_dt.transpose(1, 2),
                    ctx.op_type,
                    x.dtype,
                    top_k
                )
            elif use_route_accumulation:
                route_sel, route_sel_index = raw_sel.reshape(-1, top_k).transpose(0, 1).sort(dim=1)
                grad_output_routes = grad_output.flatten(end_dim=-3)
                grad_x = torch.empty((grad_output_routes.shape[0], x.shape[-1]), device=x.device, dtype=x.dtype)
                keys_t = keys_dt.transpose(1, 2)
                for route in range(route_sel.shape[0]):
                    cvmm_triton_accumulate_call(
                        grad_output_routes[:, route, :],
                        route_sel_index[route],
                        route_sel[route],
                        keys_t,
                        grad_x,
                        ctx.op_type,
                        route != 0
                    )
            else:
                grad_x_full = cvmm_triton_call(
                    grad_output,
                    bw_index,
                    sel,
                    keys_dt.transpose(1, 2),
                    x.dtype if reduction_weight is None else ctx.op_type,
                    bw_index_out,
                    ctx.op_type
                )

                grad_x_full = grad_x_full.view(*x.shape[:-1], -1, x.shape[-1])
                if reduction_weight is not None:
                    if need_grad_x:
                        weights = reduction_weight.view(*grad_x_full.shape[:-1]).type_as(grad_x_full)
                        if grad_x_full.shape[-2] == 1:
                            grad_x = grad_x_full.squeeze(-2) * weights.squeeze(-1).unsqueeze(-1)
                        else:
                            grad_x = (weights.unsqueeze(-2) @ grad_x_full).squeeze(-2)
                    if need_grad_reduction_weight:
                        grad_w_off = (grad_x_full.type_as(reduction_weight) @ x.unsqueeze(-1).type_as(reduction_weight)).squeeze(-1).view_as(reduction_weight)
                elif need_grad_x and grad_x_full.shape[-2] != 1:
                    grad_x = grad_x_full.sum(-2)
                elif need_grad_x:
                    grad_x = grad_x_full

            if grad_x is not None:
                grad_x = grad_x.to(x.dtype).view_as(x)

        return grad_x, None, None, None, grad_w, None, grad_w_off

known_shapes = set()

def cvmm(x: torch.Tensor, sel: Union[torch.Tensor, CVMMSel], keys: torch.Tensor):
     # Torch 2.2 on Volta GPUs is broken.
    if (version.parse(torch.__version__) >= version.parse("2.2.0") and
            torch.cuda.get_device_properties(0).major == 7 and
            torch.cuda.get_device_properties(0).minor < 5 and
            torch.is_autocast_enabled()):
        print("------------------------------- ERROR -------------------------------")
        print("ERROR: PyTorch >= 2.2 with AMP is be broken on Volta GPUs.")
        print("Triton kernels returns zeroes only. Please downgrade to 2.1 series.")
        print("Alternatively, disable mixed precision training")
        print("See: https://github.com/pytorch/pytorch/issues/127157")
        print("---------------------------------------------------------------------")
        raise RuntimeError("PyTorch >= 2.2 Triton with AMP is to be broken on Volta GPUs.")

    create_kernels()

    if not isinstance(sel, CVMMSel):
        sel = cvmm_prepare_sel(sel, keys.shape[0])

    sh = (x.shape, keys.shape)
    if sh not in known_shapes:
        print("New shape:", sh)
        known_shapes.add(sh)

    return CVMM.apply(x, sel.raw_sel, sel.sel_index, sel.sel, keys, sel.out_index, sel.reduction_weight)


def cvmm_prepare_sel2(sel: Union[torch.Tensor, CVMMSel], w: Optional[torch.Tensor] = None, route_input: bool = False) -> CVMMSel:
    # Has multiple selections for each batch element
    if isinstance(sel, CVMMSel):
        if route_input:
            route_index = sel.out_index if sel.out_index is not None else sel.sel_index
            return CVMMSel(sel.raw_sel, sel.sel, route_index, None, w)
        return CVMMSel(sel.raw_sel, sel.sel, sel.sel_index, sel.out_index, w)

    n_per_batch = sel.shape[-1]

    fsel = sel.flatten().to(torch.int32)
    ssel, sel_index = fsel.sort()

    if route_input:
        return CVMMSel(fsel.view_as(sel), ssel.view_as(sel), sel_index, None, w)

    in_index = sel_index // n_per_batch
    return CVMMSel(fsel.view_as(sel), ssel.view_as(sel), in_index, sel_index, w)
