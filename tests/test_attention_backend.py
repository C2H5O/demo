from copy import deepcopy
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from models.attention_backend import AttentionBackend, configure_da3_attention, eager_attention
from scripts.check_attention_backend import load_attention


@pytest.mark.parametrize("mask_kind", ["none", "boolean", "additive"])
@pytest.mark.parametrize("scale", [0.125, 0.37])
def test_math_mask_scale_and_backward(mask_kind, scale):
    torch.manual_seed(7)
    q, k, v = [torch.randn(2, 3, 7, 8, requires_grad=True) for _ in range(3)]
    mask = None
    if mask_kind != "none":
        keep = torch.ones(2, 1, 7, 7, dtype=torch.bool)
        keep[..., -2:] = False
        mask = keep if mask_kind == "boolean" else torch.zeros_like(keep, dtype=q.dtype).masked_fill(~keep, -torch.inf)
    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
    actual = eager_attention(q, k, v, attn_mask=mask, scale=scale)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    gradients = torch.autograd.grad(actual.square().sum(), (q, k, v), retain_graph=True)
    reference_gradients = torch.autograd.grad(expected.square().sum(), (q, k, v))
    for value, reference in zip(gradients, reference_gradients):
        torch.testing.assert_close(value, reference, atol=5e-6, rtol=2e-4)


@pytest.mark.parametrize("additive", [False, True])
def test_eager_fully_masked_rows_are_zero_with_finite_backward(additive):
    # Older PyTorch math SDPA returns NaN on empty rows; test the safe eager
    # behavior independently instead of using that old kernel as an oracle.
    q = torch.randn(1, 2, 4, 8, requires_grad=True)
    keep = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    keep[..., 0, :] = False
    mask = torch.zeros_like(keep, dtype=q.dtype).masked_fill(~keep, -torch.inf) if additive else keep
    output = eager_attention(q, q, q, attn_mask=mask)
    assert torch.isfinite(output).all()
    assert not output[..., 0, :].count_nonzero()
    output.square().sum().backward()
    assert torch.isfinite(q.grad).all()


def test_auto_cpu_fallback_and_strict_flash(capsys):
    q = torch.randn(1, 2, 4, 8)
    policy = AttentionBackend("auto")
    policy(q, q, q)
    policy(q, q, q)
    assert policy.effective == "SDPA_MATH"
    assert capsys.readouterr().out.count("[Attention Backend]") == 1
    with pytest.raises(RuntimeError, match="usable Flash"):
        AttentionBackend("flash")(q, q, q)
    with pytest.raises(ValueError):
        AttentionBackend("fake_flash")


@pytest.fixture
def official_class():
    source = Path(os.environ.get("DA3_SOURCE_ROOT", "external/Depth-Anything-3"))
    if not (source / "src/depth_anything_3/model/dinov2/layers/attention.py").is_file():
        pytest.skip("Run with installed pinned DA3 source or DA3_SOURCE_ROOT")
    return load_attention(source)


@pytest.mark.parametrize("backend", ["auto", "sdpa", "eager"])
@pytest.mark.parametrize("masked", [False, True])
def test_official_forward_norm_rope_projection_and_eval_dropout(official_class, backend, masked):
    class Rope(torch.nn.Module):
        def forward(self, value, pos):
            # A position-dependent orthogonal rotation exercises the exact call order.
            return value.roll(1, dims=-1) * pos[:, None, :, :1]

    torch.manual_seed(3)
    original = official_class(32, num_heads=4, qkv_bias=True, qk_norm=True,
                              attn_drop=0.25, proj_drop=0.3, rope=Rope()).eval()
    candidate = deepcopy(original)
    keys = tuple(candidate.state_dict())
    params = [id(p) for p in candidate.parameters()]
    configure_da3_attention(candidate, backend)
    x = torch.randn(2, 9, 32)
    pos = torch.ones(2, 9, 2)
    pos[:, ::2] = -1
    mask = torch.ones(2, 9, 9, dtype=torch.bool) if masked else None
    if mask is not None:
        mask[..., -2:] = False
    reference = original(x, pos=pos, attn_mask=mask)
    actual = candidate(x, pos=pos, attn_mask=mask)
    torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-5)
    assert tuple(candidate.state_dict()) == keys
    assert [id(p) for p in candidate.parameters()] == params
    assert not hasattr(candidate, "q") and not hasattr(candidate, "k")


def test_training_dropout_and_qk_hooks_survive_checkpoint(official_class):
    from torch.utils.checkpoint import checkpoint

    torch.manual_seed(9)
    model = official_class(32, num_heads=4, qk_norm=True, attn_drop=0.2).train()
    configure_da3_attention(model, "sdpa")
    captures = {}
    def save(name):
        def hook(_module, _inputs, output):
            output.retain_grad()
            captures[name] = output
        return hook
    handles = [getattr(model, name + "_norm").register_forward_hook(save(name)) for name in ("q", "k")]
    x = torch.randn(2, 8, 32, requires_grad=True)
    out = checkpoint(model, x, use_reentrant=False)
    retained = dict(captures)
    loss = out.square().mean() + sum(value.square().mean() for value in retained.values())
    loss.backward()
    for value in retained.values():
        assert value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().max() > 0
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert model.qkv.weight.grad is not None
    for handle in handles:
        handle.remove()


def test_saved_checkpoint_backend_can_be_overridden(monkeypatch):
    import evaluation.evaluate_crossclip_projection as module
    recorded = {}
    class Dummy(torch.nn.Module):
        def __init__(self, config, device):
            super().__init__()
            recorded.update(config)
    monkeypatch.setattr(module, "DA3SmallStudent", Dummy)
    monkeypatch.setattr(module, "require_student_cache_protocol", lambda *args: None)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {
        "config": {"student": {"checkpoint": "preserved", "attention_backend": "flash"}},
        "model": {},
    })
    module._load_model(Path("unused.pt"), {"student": {"attention_backend": "eager"}}, torch.device("cpu"))
    assert recorded == {"checkpoint": "preserved", "attention_backend": "eager"}
