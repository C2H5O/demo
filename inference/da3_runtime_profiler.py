"""Observer-only CUDA-event profiling for the unmodified DA3 execution path."""
from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
from functools import wraps

import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

from models.attention_capture import _blocks


class _DenseSDPAMode(TorchFunctionMode):
    def __init__(self, profiler, layer):
        super().__init__()
        self.profiler, self.layer = profiler, layer

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is F.scaled_dot_product_attention:
            return self.profiler.measure_sdpa(self.layer, func, args, kwargs)
        return func(*args, **kwargs)


class DA3RuntimeProfiler:
    """Profile real DA3 modules without replacing their computations.

    Parents and children are intentionally retained as separate hierarchy levels;
    callers must compare siblings rather than summing every reported region.
    """

    def __init__(self, model, *, profile_dense_attention: bool):
        if not torch.cuda.is_available():
            raise RuntimeError("DA3 runtime profiling requires CUDA")
        self.model = model
        self.profile_dense_attention = bool(profile_dense_attention)
        self.stack = ExitStack()
        self.events = []
        self.seconds = Counter()
        self.calls = Counter()
        self.shapes = Counter()
        self._active_dense_layer = None
        encoder = getattr(getattr(model, "backbone", None), "pretrained", None)
        if encoder is None:
            raise RuntimeError("Profiler requires the real DA3 backbone.pretrained encoder")
        self.encoder = encoder
        self.blocks = _blocks(encoder)
        alt_start = int(encoder.alt_start)
        self.global_layers = [
            index for index in range(len(self.blocks))
            if alt_start != -1 and index >= alt_start and index % 2 == 1
        ]

    def _event(self):
        return torch.cuda.Event(enable_timing=True)

    def measure_call(self, name, operation):
        start, end = self._event(), self._event()
        start.record()
        result = operation()
        end.record()
        self.events.append((name, start, end))
        self.calls[name] += 1
        return result

    def measure_sdpa(self, layer, func, args, kwargs):
        q, k = args[:2]
        self.shapes[(layer, int(q.shape[-2]), int(k.shape[-2]))] += 1
        return self.measure_call(
            f"blocks.block{layer}.attention.sdpa", lambda: func(*args, **kwargs)
        )

    def _module_hooks(self, module, name):
        starts = []

        def before(_module, _args):
            event = self._event()
            event.record()
            starts.append(event)

        def after(_module, _args, output):
            end = self._event()
            end.record()
            self.events.append((name, starts.pop(), end))
            self.calls[name] += 1
            return output

        pre = module.register_forward_pre_hook(before)
        post = module.register_forward_hook(after)
        self.stack.callback(pre.remove)
        self.stack.callback(post.remove)

    def _replace_forward_for_sdpa(self, attention, layer):
        original = attention.forward

        @wraps(original)
        def observed(*args, **kwargs):
            prior_layer = self._active_dense_layer
            self._active_dense_layer = layer
            try:
                with _DenseSDPAMode(self, layer):
                    return original(*args, **kwargs)
            finally:
                self._active_dense_layer = prior_layer

        had_instance = "forward" in attention.__dict__
        prior = attention.__dict__.get("forward")
        attention.forward = observed

        def restore():
            if had_instance:
                attention.forward = prior
            else:
                attention.__dict__.pop("forward", None)

        self.stack.callback(restore)

    def _rope_hooks(self, module):
        starts = []

        def before(_module, _args):
            event = self._event()
            event.record()
            starts.append((self._active_dense_layer, event))

        def after(_module, _args, output):
            layer, start = starts.pop()
            if layer is not None:
                end = self._event()
                end.record()
                name = f"blocks.block{layer}.attention.rope"
                self.events.append((name, start, end))
                self.calls[name] += 1
            return output

        pre = module.register_forward_pre_hook(before)
        post = module.register_forward_hook(after)
        self.stack.callback(pre.remove)
        self.stack.callback(post.remove)

    def __enter__(self):
        if self.model.training:
            raise ValueError("Profiler requires model.eval()")
        self._module_hooks(self.model.backbone, "top_level.backbone")
        for index, block in enumerate(self.blocks):
            kind = "global" if index in self.global_layers else "local"
            self._module_hooks(block, f"blocks.block{index}.{kind}.total")
            self._module_hooks(block.attn, f"blocks.block{index}.{kind}.attention_total")
            if hasattr(block, "mlp"):
                self._module_hooks(block.mlp, f"blocks.block{index}.{kind}.mlp")
        if self.profile_dense_attention:
            ropes = {}
            for layer in self.global_layers:
                attention = self.blocks[layer].attn
                self._module_hooks(attention.qkv, f"blocks.block{layer}.attention.qkv_projection")
                self._module_hooks(attention.q_norm, f"blocks.block{layer}.attention.q_norm")
                self._module_hooks(attention.k_norm, f"blocks.block{layer}.attention.k_norm")
                self._module_hooks(attention.proj, f"blocks.block{layer}.attention.output_projection")
                self._replace_forward_for_sdpa(attention, layer)
                if attention.rope is not None:
                    ropes[id(attention.rope)] = attention.rope
            for rope in ropes.values():
                self._rope_hooks(rope)
        return self

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)

    def finish_iteration(self):
        torch.cuda.synchronize()
        for name, start, end in self.events:
            self.seconds[name] += start.elapsed_time(end) / 1000.0
        self.events.clear()

    def summary(self):
        regions = {}
        for name in sorted(self.calls):
            calls = self.calls[name]
            total = self.seconds[name]
            regions[name] = {
                "calls": calls,
                "total_seconds": total,
                "mean_ms": 1000.0 * total / calls if calls else 0.0,
            }
        if self.profile_dense_attention:
            for layer in self.global_layers:
                prefix = f"blocks.block{layer}.attention."
                components = ("q_norm", "k_norm", "rope")
                calls = min(self.calls[prefix + "q_norm"], self.calls[prefix + "k_norm"])
                total = sum(self.seconds[prefix + item] for item in components)
                regions[prefix + "qk_norm_rope"] = {
                    "calls": calls,
                    "total_seconds": total,
                    "mean_ms": 1000.0 * total / calls if calls else 0.0,
                    "note": "sum of q_norm, k_norm, and both RoPE calls per attention invocation",
                }
        return {
            "global_attention_layers": self.global_layers,
            "dense_attention_shapes": [
                {"layer": layer, "q_tokens": q, "kv_tokens": k, "calls": calls}
                for (layer, q, k), calls in sorted(self.shapes.items())
            ],
            "regions": regions,
            "aggregation_rule": "compare siblings only; parent regions include their children",
        }


__all__ = ["DA3RuntimeProfiler"]
