"""Instance-local kernel policy for existing upstream SDPA calls."""

from __future__ import annotations

from functools import wraps
from typing import Any, Mapping
import warnings

import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode
from torch.nn.attention import SDPBackend, sdpa_kernel


_BACKENDS = {
    "flash_sdpa": SDPBackend.FLASH_ATTENTION,
    "efficient_sdpa": SDPBackend.EFFICIENT_ATTENTION,
    "math_sdpa": SDPBackend.MATH,
}


class SDPAKernelPolicy:
    def __init__(self, label: str, *, allow_math_fallback: bool = True) -> None:
        self.label = label
        self.allow_math_fallback = allow_math_fallback
        self.observed: set[str] = set()
        self._selected: dict[tuple, str] = {}

    def __call__(self, query, key, value, attn_mask=None, dropout_p=0.0,
                 is_causal=False, *, scale=None, **kwargs):
        # Eligibility checks run before SDPA's autocast dispatcher. Reproduce
        # its CUDA input cast explicitly, including FP32 Q/K from LayerNorm;
        # otherwise mixed Q/K/V dtypes falsely force math fallback. Capture
        # hooks retain their original tensors and precision upstream.
        if query.is_cuda and torch.is_autocast_enabled():
            dtype = torch.get_autocast_gpu_dtype()
            def cast(tensor):
                if tensor is not None and tensor.is_floating_point() and tensor.dtype != torch.float64:
                    return tensor.to(dtype=dtype)
                return tensor
            query, key, value = cast(query), cast(key), cast(value)
            attn_mask = cast(attn_mask)
        def layout(tensor):
            return None if tensor is None else (
                tensor.device, tensor.dtype, tuple(tensor.shape),
                tuple(tensor.stride()), tensor.requires_grad,
            )

        signature = (layout(query), layout(key), layout(value), layout(attn_mask),
                     dropout_p, is_causal, scale, torch.is_grad_enabled(),
                     tuple(sorted(kwargs.items())))
        backend = self._selected.get(signature)
        if backend is None:
            reason = "non-CUDA input"
            backend = "math_sdpa"
            if query.is_cuda:
                arguments = (query, key, value, attn_mask, dropout_p, is_causal)
                # PyTorch 2.3 uses six arguments; newer releases add enable_gqa.
                try:
                    params = torch.backends.cuda.SDPAParams(
                        *arguments, bool(kwargs.get("enable_gqa", False))
                    )
                except TypeError:
                    if kwargs.get("enable_gqa", False):
                        raise ValueError("GQA requires a newer PyTorch build")
                    params = torch.backends.cuda.SDPAParams(*arguments)
                if torch.backends.cuda.can_use_flash_attention(params):
                    backend = "flash_sdpa"
                else:
                    torch.backends.cuda.can_use_flash_attention(params, debug=True)
                    reason = "Flash rejected actual dtype/layout/mask/build; see PyTorch diagnostic"
                    if torch.backends.cuda.can_use_efficient_attention(params):
                        backend = "efficient_sdpa"
                    else:
                        torch.backends.cuda.can_use_efficient_attention(params, debug=True)
            if backend == "math_sdpa" and not self.allow_math_fallback:
                raise RuntimeError(f"{self.label}: no fused SDPA backend; {reason}")
            if backend != "flash_sdpa":
                warnings.warn(
                    f"{self.label}: fallback to {backend}; {reason}; "
                    f"Q={tuple(query.shape)} dtype={query.dtype} stride={query.stride()}",
                    RuntimeWarning, stacklevel=2,
                )

        # Enable exactly one backend, so successful execution proves the log.
        # Never catch OOM and retry a larger math kernel.
        with sdpa_kernel(_BACKENDS[backend]):
            result = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                is_causal=is_causal, scale=scale, **kwargs,
            )
        if signature not in self._selected:
            self._selected[signature] = backend
            self.observed.add(backend)
            print(f"{self.label} attention backend: {backend}; "
                  f"Q={tuple(query.shape)} dtype={query.dtype}", flush=True)
        return result


class _SDPAMode(TorchFunctionMode):
    def __init__(self, policy: SDPAKernelPolicy) -> None:
        super().__init__()
        self.policy = policy

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is F.scaled_dot_product_attention:
            return self.policy(*args, **kwargs)
        return func(*args, **kwargs)


def _wrap_method(method, policy):
    @wraps(method)
    def forward(*args, **kwargs):
        with _SDPAMode(policy):
            return method(*args, **kwargs)
    return forward


def configure_flash_attention(model: torch.nn.Module, config: Mapping[str, Any],
                              label: str) -> list[SDPAKernelPolicy]:
    if not bool(config.get("enabled", False)):
        return []
    if config.get("backend", "auto") != "auto":
        raise ValueError("flash_attention.backend must be auto")
    policies = []
    for name, module in model.named_modules():
        origin = type(module).__module__
        is_da3 = origin == "depth_anything_3.model.dinov2.layers.attention"
        is_teacher = origin == "vggt_omega.models.layers.attention"
        if not (is_da3 or is_teacher) or not hasattr(module, "qkv"):
            continue
        if hasattr(module, "_flash_sdpa_policy"):
            policies.append(module._flash_sdpa_policy)
            continue
        if is_da3:
            module.fused_attn = True
        policy = SDPAKernelPolicy(
            f"{label}.{name}",
            allow_math_fallback=bool(config.get("allow_math_fallback", True)),
        )
        # Teacher forward_list also uses compute_attention. Keeping the original
        # method preserves projection, RoPE, normalization, dropout and masks.
        method_name = "compute_attention" if hasattr(module, "compute_attention") else "forward"
        setattr(module, method_name, _wrap_method(getattr(module, method_name), policy))
        module._flash_sdpa_policy = policy
        policies.append(policy)
    if not policies:
        raise RuntimeError(f"{label}: no audited DA3/VGGT-Omega attention modules found")
    print(f"{label} backend: pending first forward ({len(policies)} SDPA modules)")
    return policies
