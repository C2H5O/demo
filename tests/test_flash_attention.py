from __future__ import annotations

from copy import deepcopy
import importlib.util
import math
import os
from pathlib import Path
import sys
import types

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint

from losses.attention_distillation_loss import CrossFrameAttentionDistillationLoss
from models.flash_attention import SDPAKernelPolicy, configure_flash_attention
from utils.config import load_config


def _feature(q, k):
    return {"q": q, "k": k, "metadata": {
        "patch_grid_h": 1, "patch_grid_w": q.shape[-2], "patch_size": 1,
    }}


def _full_reference(tq, tk, sq, sk, cfg):
    # Deliberately independent of the production probability/divergence helpers.
    values = []
    for source in range(sq.shape[1]):
        for offset in cfg["frame_offsets"]:
            target = source + offset
            if not 0 <= target < sq.shape[1]:
                continue
            probabilities = []
            for q, k, temperature in ((tq, tk, cfg["temperature_teacher"]),
                                      (sq, sk, cfg["temperature_student"])):
                logits = q[:, source].float() @ k[:, target].float().transpose(-2, -1)
                p = (logits / (math.sqrt(q.shape[-1]) * temperature)).softmax(-1).mean(1)
                p = p + cfg["eps"]
                probabilities.append(p / p.sum(-1, keepdim=True))
            t, s = probabilities
            if cfg["divergence"] == "kl":
                value = (t * (t.log() - s.log())).sum(-1)
            else:
                m = (t + s) * 0.5
                value = ((t * (t.log() - m.log())).sum(-1)
                         + (s * (s.log() - m.log())).sum(-1)) * 0.5
            values.append(value.clamp_min(0).reshape(-1))
    return torch.cat(values).mean()


@pytest.mark.parametrize("kind", ["js", "kl"])
@pytest.mark.parametrize("chunk", [16, 32, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_full_vs_chunk_loss_and_gradients(kind, chunk, dtype):
    torch.manual_seed(2026)
    cfg = load_config("configs/baselines/C.yaml")["attention_distill"]
    cfg.update(divergence=kind, query_chunk_size=chunk, teacher_layers=[4],
               student_layers=[5], temperature_teacher=0.8, temperature_student=1.3)
    # Uneven final chunk, unequal head counts AND dimensions, all frame pairs.
    tq = torch.randn(1, 3, 4, 129, 64).to(dtype)
    tk = torch.randn_like(tq)
    sq = torch.randn(1, 3, 2, 129, 32).to(dtype).requires_grad_()
    sk = torch.randn_like(sq).requires_grad_()
    full = _full_reference(tq, tk, sq, sk, cfg)
    full.backward()
    gradients = [sq.grad.clone(), sk.grad.clone()]
    sq.grad = sk.grad = None
    loss, _ = CrossFrameAttentionDistillationLoss(cfg)(
        {4: _feature(tq, tk)}, {5: _feature(sq, sk)}
    )
    loss.backward()
    grad_error = max((a.float() - b.grad.float()).abs().max().item()
                     for a, b in zip(gradients, (sq, sk)))
    print(f"EQUIVALENCE {kind} {dtype} chunk={chunk} "
          f"loss_abs={(full-loss).abs().item():.9g} grad_max_abs={grad_error:.9g}")
    torch.testing.assert_close(loss, full, atol=2e-7, rtol=2e-6)
    for expected, actual in zip(gradients, (sq.grad, sk.grad)):
        torch.testing.assert_close(actual, expected,
                                   atol=3e-6 if dtype == torch.bfloat16 else 2e-9,
                                   rtol=0.04 if dtype == torch.bfloat16 else 2e-4)


def test_cpu_fallback_mask_scale_dropout_and_no_global_change():
    torch.manual_seed(42)
    q = torch.randn(1, 4, 17, 16, requires_grad=True)
    mask = torch.ones(17, 17, dtype=torch.bool).tril()
    policy = SDPAKernelPolicy("CPU")
    original = F.scaled_dot_product_attention
    torch.manual_seed(7)
    with sdpa_kernel(SDPBackend.MATH):
        expected = original(q, q, q, attn_mask=mask, scale=0.3, dropout_p=0.2)
    torch.manual_seed(7)
    with pytest.warns(RuntimeWarning, match="non-CUDA"):
        actual = policy(q, q, q, attn_mask=mask, scale=0.3, dropout_p=0.2)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert F.scaled_dot_product_attention is original
    assert policy.observed == {"math_sdpa"}
    with pytest.raises(RuntimeError, match="no fused"):
        SDPAKernelPolicy("strict", allow_math_fallback=False)(q, q, q)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_backend_output_and_gradients():
    torch.manual_seed(42)
    tensors = [torch.randn(1, 4, 128, 64, device="cuda", dtype=torch.bfloat16,
                           requires_grad=True) for _ in range(3)]
    with sdpa_kernel(SDPBackend.MATH):
        expected = F.scaled_dot_product_attention(*tensors)
    expected.float().square().mean().backward()
    grads = [t.grad.clone() for t in tensors]
    for t in tensors:
        t.grad = None
    policy = SDPAKernelPolicy("CUDA synthetic")
    actual = policy(*tensors)
    actual.float().square().mean().backward()
    error = (expected.float() - actual.float()).abs()
    print(f"CUDA output backend={policy.observed} max_abs={error.max().item():.9g} "
          f"mean_abs={error.mean().item():.9g}")
    torch.testing.assert_close(actual, expected, atol=0.008, rtol=0.03)
    for t, grad in zip(tensors, grads):
        torch.testing.assert_close(t.grad, grad, atol=3e-6, rtol=0.05)


def _load_attention(monkeypatch, kind):
    env = "DA3_ATTENTION_SOURCE" if kind == "student" else "VGGT_ATTENTION_SOURCE"
    source = os.environ.get(env)
    if not source:
        pytest.skip(f"Set {env} to the installed upstream attention.py for module tests")
    name = ("depth_anything_3.model.dinov2.layers.attention" if kind == "student"
            else "vggt_omega.models.layers.attention")
    parent = name.rsplit(".", 1)[0]
    package = types.ModuleType(parent)
    package.__path__ = [str(Path(source).parent)]
    monkeypatch.setitem(sys.modules, parent, package)
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module.Attention if kind == "student" else module.SelfAttention


@pytest.mark.parametrize("kind", ["student", "teacher"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_upstream_module_equivalence_and_checkpoint(monkeypatch, kind, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    cls = _load_attention(monkeypatch, kind)
    torch.manual_seed(7)
    settings = {"qk_norm": True, "fused_attn": False} if kind == "student" else {"use_qk_norm": True}
    original = cls(dim=128, num_heads=4, qkv_bias=True, **settings).to(device).eval()
    candidate = deepcopy(original)
    before_keys = set(candidate.state_dict())
    policies = configure_flash_attention(candidate, {
        "enabled": True, "allow_math_fallback": device != "cuda",
    }, kind)
    assert set(candidate.state_dict()) == before_keys
    x = torch.randn(1, 33, 128, device=device, requires_grad=True)
    cast = torch.bfloat16 if device == "cuda" else torch.float32
    with torch.autocast(device_type=device, dtype=cast, enabled=device == "cuda"):
        with sdpa_kernel(SDPBackend.MATH):
            expected = original(x)
        # Capture hooks remain active and differentiable under checkpoint replay.
        captured = []
        handle = candidate.q_norm.register_forward_hook(lambda m, a, out: captured.append(out))
        actual = checkpoint(candidate, x, use_reentrant=False)
    error = (actual.float() - expected.float()).abs()
    print(f"MODULE {kind} {device} backend={policies[0].observed} "
          f"max_abs={error.max().item():.9g} mean_abs={error.mean().item():.9g}")
    torch.testing.assert_close(actual, expected,
                               atol=0.01 if device == "cuda" else 3e-7,
                               rtol=0.03 if device == "cuda" else 3e-5)
    (actual.float().square().mean() + captured[0].float().square().mean()).backward()
    assert candidate.qkv.weight.grad is not None
    assert torch.isfinite(candidate.qkv.weight.grad).all()
    assert candidate.qkv.weight.grad.abs().max() > 0
    handle.remove()


def test_only_c_e_opt_in():
    for baseline in "BCDE":
        cfg = load_config(f"configs/baselines/{baseline}.yaml")
        assert cfg["flash_attention"]["enabled"] == (baseline in "CE")
        assert cfg["attention_distill"]["query_chunk_size"] == 128


def test_chunk_checkpoint_does_not_save_probability_matrices():
    cfg = load_config("configs/baselines/C.yaml")["attention_distill"]
    cfg.update(query_chunk_size=16, teacher_layers=[4], student_layers=[5])
    teacher = _feature(torch.randn(1, 2, 4, 129, 64), torch.randn(1, 2, 4, 129, 64))
    student = _feature(torch.randn(1, 2, 2, 129, 32, requires_grad=True),
                       torch.randn(1, 2, 2, 129, 32, requires_grad=True))
    saved = []
    def pack(tensor):
        saved.append(tuple(tensor.shape))
        return tensor
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        loss, _ = CrossFrameAttentionDistillationLoss(cfg)({4: teacher}, {5: student})
    assert not any(len(shape) in (3, 4) and shape[-1] == 129
                   and shape[-2] in (1, 16, 129) for shape in saved)
    loss.backward()
    assert student["q"].grad.abs().max() > 0
