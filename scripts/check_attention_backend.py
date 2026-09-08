"""Small real-DA3 attention check. No weights, datasets, or training required."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.attention_backend import AttentionBackend, DA3_ATTENTION_MODULE, configure_da3_attention, eager_attention
from utils.config import load_config


def load_attention(source_root):
    path = Path(source_root) / "src/depth_anything_3/model/dinov2/layers/attention.py"
    spec = importlib.util.spec_from_file_location(DA3_ATTENTION_MODULE, path)
    module = importlib.util.module_from_spec(spec)
    # Read-only source inspection, including when --da3-root is another checkout.
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    return module.Attention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--backend", choices=["auto", "flash", "sdpa", "eager"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--da3-root", default="external/Depth-Anything-3")
    args = parser.parse_args()
    config = load_config(args.config)
    backend = args.backend or config["student"].get("attention_backend", "auto")
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(42)
    q, k, v = [torch.randn(2, 6, 32, 64, device=args.device, dtype=dtype,
                           requires_grad=True) for _ in range(3)]
    policy = AttentionBackend(backend)
    results = []
    for kind in ("none", "boolean", "additive"):
        mask = None
        if kind != "none":
            keep = torch.ones(2, 1, 32, 32, device=args.device, dtype=torch.bool)
            keep[..., -4:] = False
            mask = keep if kind == "boolean" else torch.zeros_like(keep, dtype=dtype).masked_fill(~keep, float("-inf"))
        ref = eager_attention(q, k, v, attn_mask=mask, scale=0.125)
        selected = policy
        mask_note = None
        try:
            out = selected(q, k, v, attn_mask=mask, scale=0.125)
        except RuntimeError as error:
            # Some Flash versions do not support explicit masks. The unmasked
            # Flash probe remains strict; test masks with a clearly labeled SDPA fallback.
            if backend != "flash" or mask is None or "requires a usable Flash" not in str(error):
                raise
            mask_note = str(error)
            selected = AttentionBackend("sdpa")
            out = selected(q, k, v, attn_mask=mask, scale=0.125)
        diff = (ref.float() - out.float()).abs()
        assert torch.isfinite(out).all() and torch.isfinite(ref).all()
        torch.testing.assert_close(out, ref, atol=0.025 if dtype == torch.bfloat16 else 0.003, rtol=0.03)
        out.float().square().mean().backward()
        assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q, k, v))
        results.append(dict(mask=kind, effective=selected.effective, mask_note=mask_note,
                            max_abs_diff=diff.max().item(), mean_abs_diff=diff.mean().item()))
    # Exercise the actual pinned official class, including Q/K hooks and projection.
    cls = load_attention(args.da3_root)
    model = cls(384, num_heads=6, qkv_bias=True, qk_norm=True).to(args.device, dtype).eval()
    x = torch.randn(2, 32, 384, device=args.device, dtype=dtype)
    with torch.no_grad():
        original = model(x)
    state_keys = tuple(model.state_dict())
    parameter_ids = [id(p) for p in model.parameters()]
    trainability = [p.requires_grad for p in model.parameters()]
    installed = configure_da3_attention(model, backend)
    captured = {}
    def hook(name):
        def save(_module, _inputs, output):
            output.retain_grad()
            captured[name] = output
        return save
    handles = [getattr(model, name + "_norm").register_forward_hook(hook(name)) for name in ("q", "k")]
    from torch.profiler import profile, ProfilerActivity
    activities = [ProfilerActivity.CPU]
    if args.device.startswith("cuda"):
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities) as prof:
        actual = model(x)
        (actual.float().square().mean() + sum(t.float().square().mean() for t in captured.values())).backward()
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
    for handle in handles:
        handle.remove()
    assert state_keys == tuple(model.state_dict())
    assert parameter_ids == [id(p) for p in model.parameters()]
    assert trainability == [p.requires_grad for p in model.parameters()]
    assert all(t.grad is not None and torch.isfinite(t.grad).all() and t.grad.abs().max() > 0 for t in captured.values())
    torch.testing.assert_close(actual, original, atol=0.025, rtol=0.03)
    names = sorted({event.key for event in prof.key_averages()
                    if any(word in event.key.lower() for word in ("flash", "efficient", "scaled_dot"))})
    if installed.effective == "FLASH_ATTENTION":
        assert "aten::_scaled_dot_product_flash_attention" in names, names
    print(json.dumps(dict(torch=torch.__version__, device=args.device, dtype=str(dtype),
                          numerical=results, official_forward_max_abs_diff=(actual-original).abs().max().item(),
                          qk_backward="passed", state_and_trainability="unchanged",
                          profiler_operators=names, effective=installed.effective), indent=2))


if __name__ == "__main__":
    main()
