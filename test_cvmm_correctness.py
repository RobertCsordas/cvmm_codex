import torch

from cvmm import cvmm, cvmm_prepare_sel2


def _reference(x, sel, keys, weights=None):
    x_flat = x.flatten(end_dim=-2).float()
    sel_flat = sel.flatten(end_dim=-2)
    selected = keys[sel_flat].float()
    out = (x_flat[:, None, :, None].float() * selected).sum(dim=-2)
    out = out.view(*sel.shape, keys.shape[-1])
    if weights is not None:
        out = (weights.unsqueeze(-1).float() * out).sum(dim=-2)
    return out


def _check_case(weighted: bool):
    torch.manual_seed(1234 + int(weighted))
    device = "cuda"
    b, t, d, out_d, n_experts, top_k = 3, 5, 16, 11, 7, 3

    x = torch.randn(b, t, d, device=device, dtype=torch.bfloat16, requires_grad=True)
    keys = torch.randn(n_experts, d, out_d, device=device, dtype=torch.float32, requires_grad=True)
    sel = torch.randint(0, n_experts, (b, t, top_k), device=device)
    weights = None
    if weighted:
        weights = torch.randn(b, t, top_k, device=device, dtype=torch.float32, requires_grad=True)

    sel2 = cvmm_prepare_sel2(sel, weights)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        actual = cvmm(x, sel2, keys)
    expected = _reference(x, sel, keys, weights)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)

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
        torch.testing.assert_close(actual_grad, expected_grad, rtol=3e-2, atol=3e-2)


def test_unweighted():
    _check_case(False)


def test_weighted():
    _check_case(True)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for cvmm correctness tests")
    test_unweighted()
    test_weighted()
    print("cvmm correctness tests passed")
