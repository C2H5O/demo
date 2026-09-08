"""Instance-local DA3 kernel selection; no new parameters or Q/K retention."""
from __future__ import annotations

from types import MethodType
import warnings

import torch
import torch.nn.functional as F

BACKENDS = {"auto", "flash", "sdpa", "eager"}
DA3_ATTENTION_MODULE = "depth_anything_3.model.dinov2.layers.attention"


def eager_attention(q, k, v, *, attn_mask=None, dropout_p=0.0, scale=None):
    """DA3's original pre-scaled Q formula, with SDPA boolean/additive masks."""
    scores = (q * (q.shape[-1] ** -0.5 if scale is None else scale)) @ k.transpose(-2, -1)
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask
    # SDPA returns zero for fully masked rows.
    fully_masked = torch.isneginf(scores).all(dim=-1, keepdim=True)
    # Avoid NaN softmax intermediates, including their backward for additive masks.
    probs = scores.masked_fill(fully_masked, 0.0).softmax(dim=-1)
    probs = probs.masked_fill(fully_masked, 0.0)
    return F.dropout(probs, p=dropout_p, training=True) @ v


class AttentionBackend:
    def __init__(self, requested="auto"):
        if requested not in BACKENDS:
            raise ValueError(f"attention_backend must be one of {sorted(BACKENDS)}")
        self.requested = requested
        self.effective = None
        self._reported = set()

    def _select(self, q, k, v, mask, dropout):
        if self.requested == "eager":
            return "EAGER", "explicit eager comparison"
        if not hasattr(F, "scaled_dot_product_attention"):
            if self.requested != "auto":
                raise RuntimeError("Requested SDPA is unavailable in this PyTorch build")
            return "EAGER", "PyTorch has no SDPA"
        cuda = torch.backends.cuda
        flash = efficient = False
        if q.is_cuda:
            try:
                params = cuda.SDPAParams(q, k, v, mask, dropout, False, False)
            except TypeError:
                # The repository's PyTorch 2.3 pin predates the enable_gqa field.
                params = cuda.SDPAParams(q, k, v, mask, dropout, False)
            flash = cuda.can_use_flash_attention(params)
            efficient = cuda.can_use_efficient_attention(params)
        if flash:
            return "FLASH_ATTENTION", "Flash supports these actual Q/K/V inputs"
        if self.requested == "flash":
            raise RuntimeError(
                "attention_backend=flash requires a usable Flash SDPA kernel for the actual "
                f"device={q.device}, dtype={q.dtype}, shape={tuple(q.shape)}, mask/dropout. "
                "Use auto for an explicitly reported fallback."
            )
        if efficient:
            return "EFFICIENT_ATTENTION", "Flash unavailable for these inputs/build"
        return "SDPA_MATH", "Flash/efficient unavailable for these inputs/build"

    def __call__(self, q, k, v, *, attn_mask=None, dropout_p=0.0, scale=None):
        effective, reason = self._select(q, k, v, attn_mask, dropout_p)
        if effective == "EAGER":
            result = eager_attention(q, k, v, attn_mask=attn_mask,
                                     dropout_p=dropout_p, scale=scale)
        else:
            from torch.nn.attention import SDPBackend, sdpa_kernel
            backend = {
                "FLASH_ATTENTION": SDPBackend.FLASH_ATTENTION,
                "EFFICIENT_ATTENTION": SDPBackend.EFFICIENT_ATTENTION,
                "SDPA_MATH": SDPBackend.MATH,
            }[effective]
            # Exactly one enabled backend: successful execution proves the selection.
            # Never catch OOM or arbitrary runtime errors and silently switch kernels.
            with sdpa_kernel(backend):
                result = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                    is_causal=False, scale=scale,
                )
        if not self._reported:
            device = torch.cuda.get_device_name(q.device) if q.is_cuda else str(q.device)
            flash_query = getattr(torch.backends.cuda, "is_flash_attention_available", None)
            flash_build = flash_query() if flash_query else "unknown (PyTorch build has no query)"
            print(
                "[Attention Backend]\n"
                f"requested: {self.requested}\ndevice: {device}\ndtype: {q.dtype}\n"
                f"PyTorch SDPA available: {hasattr(F, 'scaled_dot_product_attention')}\n"
                f"Flash SDP available (build): {flash_build}\n"
                f"DA3 effective backend: {effective}\nreason: {reason}",
                flush=True,
            )
        elif effective not in self._reported:
            warnings.warn(f"DA3 backend changed to {effective}: {reason}; dtype={q.dtype}")
        self._reported.add(effective)
        self.effective = effective
        return result


def _da3_forward(self, x, pos=None, attn_mask=None):
    # Matches pinned DA3 Attention.forward around the kernel exactly.
    batch, tokens, channels = x.shape
    qkv = self.qkv(x).reshape(
        batch, tokens, 3, self.num_heads, channels // self.num_heads
    ).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    q, k = self.q_norm(q), self.k_norm(k)
    if self.rope is not None and pos is not None:
        q, k = self.rope(q, pos), self.rope(k, pos)
    # Official mask is [B,N,N]. Broadcasting avoids the original head-wise copy.
    mask = attn_mask[:, None] if attn_mask is not None else None
    x = self._attention_backend(
        q, k, v, attn_mask=mask,
        dropout_p=self.attn_drop.p if self.training else 0.0, scale=self.scale,
    )
    x = x.transpose(1, 2).reshape(batch, tokens, channels)
    return self.proj_drop(self.proj(x))


def configure_da3_attention(network, requested="auto", *, required=True):
    """Patch only verified official DA3 instances, keeping all existing hooks."""
    policy = AttentionBackend(requested)
    count = 0
    for module in network.modules():
        if type(module).__module__ == DA3_ATTENTION_MODULE and type(module).__name__ == "Attention":
            module._attention_backend = policy
            module.forward = MethodType(_da3_forward, module)
            count += 1
    if required and count == 0:
        raise RuntimeError("No supported official DA3 Attention modules found")
    return policy
