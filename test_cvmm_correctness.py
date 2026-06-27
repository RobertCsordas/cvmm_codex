import torch

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


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for cvmm correctness tests")
    test_unweighted()
    test_weighted()
    test_weighted_route_input()
    print("cvmm correctness tests passed")
