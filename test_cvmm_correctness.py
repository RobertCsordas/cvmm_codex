import torch

import cvmm as cvmm_module
from cvmm import cvmm, cvmm_prepare_sel2


def _reference(x, sel, keys, weights=None):
    selected = keys[sel].float()
    if x.shape[:-1] == sel.shape:
        out = (x.float().unsqueeze(-1) * selected).sum(dim=-2)
    else:
        out = (x.float().unsqueeze(-2).unsqueeze(-1) * selected).sum(dim=-2)
    if weights is not None:
        out = (weights.unsqueeze(-1).float() * out).sum(dim=-2)
    return out


def _check_case(weighted: bool, route_input: bool = False):
    torch.manual_seed(1234 + int(weighted) + 17 * int(route_input))
    device = "cuda"
    b, t, d, out_d, n_experts, top_k = 3, 5, 16, 11, 7, 3

    x_shape = (b, t, top_k, d) if route_input else (b, t, d)
    x = torch.randn(*x_shape, device=device, dtype=torch.bfloat16, requires_grad=True)
    keys = torch.randn(n_experts, d, out_d, device=device, dtype=torch.float32, requires_grad=True)
    sel = torch.randint(0, n_experts, (b, t, top_k), device=device)
    weights = None
    if weighted:
        weights = torch.randn(b, t, top_k, device=device, dtype=torch.float32, requires_grad=True)

    sel2 = cvmm_prepare_sel2(sel, weights, route_input=route_input)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        actual = cvmm(x, sel2, keys)
    expected = _reference(x, sel, keys, weights)
    ref_rtol = 5e-2 if route_input else 2e-2
    ref_atol = 1e-1 if route_input else 2e-2
    torch.testing.assert_close(actual.float(), expected.float(), rtol=ref_rtol, atol=ref_atol)

    grad = torch.randn_like(actual.float())
    actual.backward(grad.to(actual.dtype))
    actual_grads = [x.grad.float(), keys.grad.float()]
    if weighted:
        actual_grads.append(weights.grad.float())

    x_ref = x.detach().clone().requires_grad_(True)
    keys_ref = keys.detach().clone().requires_grad_(True)
    weights_ref = None if weights is None else weights.detach().clone().requires_grad_(True)
    expected = _reference(x_ref, sel, keys_ref, weights_ref)
    expected.backward(grad)
    expected_grads = [x_ref.grad.float(), keys_ref.grad.float()]
    if weighted:
        expected_grads.append(weights_ref.grad.float())

    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        grad_rtol = 6e-2 if route_input else 3e-2
        grad_atol = 1e-1 if route_input else 3e-2
        torch.testing.assert_close(actual_grad, expected_grad, rtol=grad_rtol, atol=grad_atol)


def test_unweighted():
    _check_case(False)


def test_weighted():
    _check_case(True)


def test_weighted_route_input():
    _check_case(True, route_input=True)


def test_expert_offsets_kernel():
    cvmm_module.create_kernels()
    old_call = cvmm_module.cvmm_expert_offsets_call
    try:
        for n_experts in (1, 7, 64, 512):
            for total_routes in (1, 13, 1000):
                sel = torch.randint(0, n_experts, (total_routes,), device="cuda", dtype=torch.int32).sort().values
                fast = cvmm_module._cvmm_expert_offsets(sel, n_experts)
                cvmm_module.cvmm_expert_offsets_call = None
                ref = cvmm_module._cvmm_expert_offsets(sel, n_experts)
                cvmm_module.cvmm_expert_offsets_call = old_call
                torch.testing.assert_close(fast.cpu(), ref.cpu(), rtol=0, atol=0)
    finally:
        cvmm_module.cvmm_expert_offsets_call = old_call


def test_top32_lowmem_unweighted():
    torch.manual_seed(2026)
    device = "cuda"
    b, t, d, out_d, n_experts, top_k = 2, 5, 32, 16, 64, 32

    x = torch.randn(b, t, d, device=device, dtype=torch.bfloat16, requires_grad=True)
    keys = torch.randn(n_experts, d, out_d, device=device, dtype=torch.float32, requires_grad=True)
    sel = torch.randint(0, n_experts, (b, t, top_k), device=device)
    old_chunk_size = cvmm_module.cvmm_lowmem_chunk_size
    cvmm_module.cvmm_lowmem_chunk_size = 0
    try:
        sel2 = cvmm_prepare_sel2(sel)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            actual = cvmm(x, sel2, keys)
        expected = _reference(x, sel, keys)
        torch.testing.assert_close(actual.float(), expected.float(), rtol=5e-2, atol=1e-1)

        grad = torch.randn_like(actual.float())
        actual.backward(grad.to(actual.dtype))
        actual_x_grad = x.grad.float()
        actual_keys_grad = keys.grad.float()

        x_ref = x.detach().clone().requires_grad_(True)
        keys_ref = keys.detach().clone().requires_grad_(True)
        expected = _reference(x_ref, sel, keys_ref)
        expected.backward(grad)
        torch.testing.assert_close(actual_x_grad, x_ref.grad.float(), rtol=3e-1, atol=6e-1)
        torch.testing.assert_close(actual_keys_grad, keys_ref.grad.float(), rtol=1e-1, atol=1e-1)
    finally:
        cvmm_module.cvmm_lowmem_chunk_size = old_chunk_size


def test_weighted_route_input_reused_selector():
    torch.manual_seed(1251)
    device = "cuda"
    b, t, d, out_d, n_experts, top_k = 3, 5, 16, 11, 7, 3
    x = torch.randn(b, t, top_k, d, device=device, dtype=torch.bfloat16, requires_grad=True)
    keys = torch.randn(n_experts, d, out_d, device=device, dtype=torch.float32, requires_grad=True)
    sel = torch.randint(0, n_experts, (b, t, top_k), device=device)
    weights = torch.randn(b, t, top_k, device=device, dtype=torch.float32, requires_grad=True)

    base_sel = cvmm_prepare_sel2(sel)
    reused = cvmm_prepare_sel2(base_sel, weights, route_input=True)
    fresh = cvmm_prepare_sel2(sel, weights, route_input=True)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        actual = cvmm(x, reused, keys)
        expected = cvmm(x, fresh, keys)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for cvmm correctness tests")
    test_unweighted()
    test_weighted()
    test_weighted_route_input()
    test_expert_offsets_kernel()
    test_top32_lowmem_unweighted()
    test_weighted_route_input_reused_selector()
    print("cvmm correctness tests passed")
