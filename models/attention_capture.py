"""Optional Q/K capture at the exact attention inputs used by Teacher and Student."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional

import torch
import torch.nn as nn


def _blocks(module: nn.Module) -> list[nn.Module]:
    blocks = getattr(module, "blocks", None)
    if blocks is None:
        raise RuntimeError("Attention capture requires a Transformer blocks collection")
    if bool(getattr(module, "chunked_blocks", False)):
        return [block for chunk in blocks for block in chunk]
    return list(blocks)


def _restore_original_frame_order(
    value: torch.Tensor, reference_indices: torch.Tensor
) -> torch.Tensor:
    """Undo DA3's [reference, remaining-in-original-order] view permutation."""
    batch, frames = value.shape[:2]
    positions = torch.arange(frames, device=value.device).unsqueeze(0).expand(batch, -1)
    reference = reference_indices.to(device=value.device, dtype=torch.long).unsqueeze(1)
    restore = torch.where(positions < reference, positions + 1, positions)
    restore = torch.scatter(restore, 1, reference, torch.zeros_like(reference))
    batch_indices = torch.arange(batch, device=value.device).unsqueeze(1)
    return value[batch_indices, restore]


class VGGTOmegaAttentionCapture:
    """Capture normalized Q/K from selected global inter-frame aggregator blocks."""

    def __init__(
        self,
        model: nn.Module,
        layer_indices: Iterable[int],
        output_dtype: torch.dtype = torch.float16,
        output_device: str = "cpu",
    ) -> None:
        aggregator = getattr(model, "aggregator", None)
        if aggregator is None:
            raise RuntimeError("VGGT-Omega model has no aggregator")
        self.aggregator = aggregator
        self.layer_indices = tuple(int(value) for value in layer_indices)
        self.output_dtype = output_dtype
        self.output_device = str(output_device)
        if self.output_device not in {"cpu", "source"}:
            raise ValueError("Teacher attention output_device must be cpu or source")
        self.patch_token_start = int(getattr(aggregator, "patch_token_start"))
        self.patch_size = int(getattr(aggregator, "patch_size"))
        attention_types = list(getattr(aggregator, "inter_frame_attention_types"))
        blocks = list(getattr(aggregator, "inter_frame_blocks"))
        self._features: Dict[int, Dict[str, Any]] = {}
        self._pending_q: Dict[int, torch.Tensor] = {}
        self._batch = self._frames = self._grid_h = self._grid_w = 0
        self._handles: list[Any] = []
        for layer in self.layer_indices:
            if not 0 <= layer < len(blocks):
                raise ValueError("Invalid VGGT-Omega attention layer {}".format(layer))
            if attention_types[layer] != "global":
                raise ValueError(
                    "VGGT-Omega layer {} is {!r}, not inter-frame global attention".format(
                        layer, attention_types[layer]
                    )
                )
            attention = blocks[layer].attn
            if not bool(getattr(attention, "use_qk_norm", False)):
                raise RuntimeError("VGGT-Omega layer {} does not use Q/K norm".format(layer))
            self._handles.append(
                attention.q_norm.register_forward_hook(self._make_q_hook(layer))
            )
            self._handles.append(
                attention.k_norm.register_forward_hook(self._make_k_hook(layer, attention))
            )

    def begin(self, images: torch.Tensor) -> None:
        self._features.clear()
        self._pending_q.clear()
        self._batch, self._frames = (int(images.shape[0]), int(images.shape[1]))
        self._grid_h = int(images.shape[-2]) // self.patch_size
        self._grid_w = int(images.shape[-1]) // self.patch_size

    def _make_q_hook(self, layer: int):
        def hook(_module: nn.Module, _inputs: Any, output: torch.Tensor) -> None:
            self._pending_q[layer] = output

        return hook

    def _make_k_hook(self, layer: int, attention: nn.Module):
        def hook(_module: nn.Module, _inputs: Any, output: torch.Tensor) -> None:
            if layer not in self._pending_q:
                raise RuntimeError("VGGT-Omega K was captured before Q at layer {}".format(layer))
            q = self._pending_q.pop(layer)
            k = output
            expected_tokens = self._frames * (
                self.patch_token_start + self._grid_h * self._grid_w
            )
            if q.ndim != 4 or tuple(q.shape) != tuple(k.shape):
                raise RuntimeError("VGGT-Omega Q/K shape mismatch at layer {}".format(layer))
            if int(q.shape[0]) != self._batch or int(q.shape[2]) != expected_tokens:
                raise RuntimeError(
                    "VGGT-Omega layer {} Q/K shape {} does not match B={} F={} grid={}x{}"
                    .format(layer, tuple(q.shape), self._batch, self._frames, self._grid_h, self._grid_w)
                )
            heads, head_dim = int(q.shape[1]), int(q.shape[-1])

            def patch_only(value: torch.Tensor) -> torch.Tensor:
                value = value.reshape(
                    self._batch,
                    heads,
                    self._frames,
                    self.patch_token_start + self._grid_h * self._grid_w,
                    head_dim,
                )
                value = value[:, :, :, self.patch_token_start :, :]
                value = value.permute(0, 2, 1, 3, 4).contiguous()
                value = value.detach().to(dtype=self.output_dtype)
                # Offline cache generation stages Q/K on CPU. Online training
                # keeps each small Teacher chunk on its source GPU and consumes
                # it immediately, avoiding a GPU->CPU->GPU round trip.
                return (
                    value.to(device="cpu")
                    if self.output_device == "cpu"
                    else value
                )

            self._features[layer] = {
                "q": patch_only(q),
                "k": patch_only(k),
                "metadata": {
                    "layer_index": layer,
                    "num_frames": self._frames,
                    "patch_grid_h": self._grid_h,
                    "patch_grid_w": self._grid_w,
                    "patch_size": self.patch_size,
                    "image_height": self._grid_h * self.patch_size,
                    "image_width": self._grid_w * self.patch_size,
                    "num_heads": heads,
                    "head_dim": head_dim,
                    "special_tokens_per_frame": self.patch_token_start,
                    "qk_stage": "post_qk_norm_no_rope",
                },
            }

        return hook

    def take(self) -> Dict[int, Dict[str, Any]]:
        missing = sorted(set(self.layer_indices) - set(self._features))
        if missing or self._pending_q:
            raise RuntimeError(
                "VGGT-Omega attention capture incomplete: missing={} pending_q={}".format(
                    missing, sorted(self._pending_q)
                )
            )
        result = self._features
        self._features = {}
        return result

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


class DA3AttentionCapture:
    """Capture post-QK-norm, post-RoPE Q/K from DA3 global blocks with gradients."""

    def __init__(
        self,
        backbone: nn.Module,
        layer_indices: Iterable[int],
        ref_view_strategy: str,
    ) -> None:
        encoder = getattr(backbone, "pretrained", None)
        if encoder is None:
            raise RuntimeError("Official DA3 backbone has no pretrained DINO encoder")
        self.encoder = encoder
        self.blocks = _blocks(encoder)
        self.layer_indices = tuple(int(value) for value in layer_indices)
        self.ref_view_strategy = str(ref_view_strategy)
        self.patch_size = int(getattr(encoder, "patch_size"))
        # Official DA3 strips CLS/register tokens with this same convention.
        self.patch_token_start = 1 + int(getattr(encoder, "num_register_tokens", 0))
        alt_start = int(getattr(encoder, "alt_start"))
        # The upstream loop selects the reference immediately before block
        # alt_start-1, using the output of block alt_start-2 in original order.
        self._reference_source_layer = alt_start - 2
        self._features: Dict[int, Dict[str, Any]] = {}
        self._pending_q: Dict[int, torch.Tensor] = {}
        self._pending_pos: Dict[int, Optional[torch.Tensor]] = {}
        self._reference_indices: Optional[torch.Tensor] = None
        self._batch = self._frames = self._grid_h = self._grid_w = 0
        self._retain_gradients = False
        self._handles: list[Any] = []
        for layer in self.layer_indices:
            if not 0 <= layer < len(self.blocks):
                raise ValueError("Invalid DA3 attention layer {}".format(layer))
            expected_global = alt_start != -1 and layer >= alt_start and layer % 2 == 1
            if not expected_global:
                raise ValueError("DA3 layer {} is not a global attention layer".format(layer))
            attention = self.blocks[layer].attn
            if not isinstance(getattr(attention, "q_norm", None), nn.Module):
                raise RuntimeError("DA3 layer {} has no Q norm".format(layer))
            if not isinstance(getattr(attention, "k_norm", None), nn.Module):
                raise RuntimeError("DA3 layer {} has no K norm".format(layer))
            self._handles.append(
                attention.register_forward_pre_hook(
                    self._make_attention_pre_hook(layer), with_kwargs=True
                )
            )
            self._handles.append(
                attention.q_norm.register_forward_hook(self._make_q_hook(layer))
            )
            self._handles.append(
                attention.k_norm.register_forward_hook(
                    self._make_k_hook(layer, attention)
                )
            )
        if not 0 <= self._reference_source_layer < len(self.blocks):
            raise RuntimeError("DA3 alt_start does not identify a reference-selection layer")
        self._handles.append(
            self.blocks[self._reference_source_layer].register_forward_hook(
                self._capture_reference_indices
            )
        )

    def retain_gradients(self, enabled: bool) -> None:
        self._retain_gradients = bool(enabled)

    def begin(self, images: torch.Tensor) -> None:
        self._features.clear()
        self._pending_q.clear()
        self._pending_pos.clear()
        self._reference_indices = None
        self._batch, self._frames = int(images.shape[0]), int(images.shape[1])
        self._grid_h = int(images.shape[-2]) // self.patch_size
        self._grid_w = int(images.shape[-1]) // self.patch_size

    def _capture_reference_indices(
        self, _module: nn.Module, _inputs: Any, output: torch.Tensor
    ) -> None:
        from depth_anything_3.model.reference_view_selector import select_reference_view

        value = output.detach().reshape(self._batch, self._frames, *output.shape[1:])
        self._reference_indices = select_reference_view(
            value, strategy=self.ref_view_strategy
        ).detach()

    def _make_attention_pre_hook(self, layer: int):
        def hook(_module: nn.Module, _args: Any, kwargs: Mapping[str, Any]) -> None:
            self._pending_pos[layer] = kwargs.get("pos")

        return hook

    def _make_q_hook(self, layer: int):
        def hook(_module: nn.Module, _inputs: Any, output: torch.Tensor) -> None:
            self._pending_q[layer] = output

        return hook

    def _make_k_hook(self, layer: int, attention: nn.Module):
        def hook(_module: nn.Module, _inputs: Any, output: torch.Tensor) -> None:
            if self._reference_indices is None:
                raise RuntimeError("DA3 reference-view permutation was not captured")
            q = self._pending_q.pop(layer)
            k = output
            pos = self._pending_pos.pop(layer, None)
            if getattr(attention, "rope", None) is not None and pos is not None:
                q = attention.rope(q, pos)
                k = attention.rope(k, pos)
            expected_per_frame = self.patch_token_start + self._grid_h * self._grid_w
            if q.ndim != 4 or tuple(q.shape) != tuple(k.shape):
                raise RuntimeError("DA3 Q/K shape mismatch at layer {}".format(layer))
            if int(q.shape[0]) != self._batch or int(q.shape[2]) != self._frames * expected_per_frame:
                raise RuntimeError(
                    "DA3 layer {} Q/K shape {} does not match B={} F={} grid={}x{}"
                    .format(layer, tuple(q.shape), self._batch, self._frames, self._grid_h, self._grid_w)
                )
            heads, head_dim = int(q.shape[1]), int(q.shape[-1])

            def patch_only(value: torch.Tensor) -> torch.Tensor:
                value = value.reshape(
                    self._batch,
                    heads,
                    self._frames,
                    expected_per_frame,
                    head_dim,
                ).permute(0, 2, 1, 3, 4)
                value = value[:, :, :, self.patch_token_start :, :].contiguous()
                value = _restore_original_frame_order(value, self._reference_indices)
                if self._retain_gradients and value.requires_grad:
                    value.retain_grad()
                return value

            self._features[layer] = {
                "q": patch_only(q),
                "k": patch_only(k),
                "metadata": {
                    "layer_index": layer,
                    "num_frames": self._frames,
                    "patch_grid_h": self._grid_h,
                    "patch_grid_w": self._grid_w,
                    "patch_size": self.patch_size,
                    "image_height": self._grid_h * self.patch_size,
                    "image_width": self._grid_w * self.patch_size,
                    "num_heads": heads,
                    "head_dim": head_dim,
                    "special_tokens_per_frame": self.patch_token_start,
                    "qk_stage": "post_qk_norm_post_global_rope",
                    "frame_order": "original_after_reference_restore",
                },
            }

        return hook

    def take(self) -> Dict[int, Dict[str, Any]]:
        missing = sorted(set(self.layer_indices) - set(self._features))
        if missing or self._pending_q or self._pending_pos:
            raise RuntimeError(
                "DA3 attention capture incomplete: missing={} pending_q={} pending_pos={}"
                .format(missing, sorted(self._pending_q), sorted(self._pending_pos))
            )
        result = self._features
        self._features = {}
        return result

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


__all__ = [
    "DA3AttentionCapture",
    "VGGTOmegaAttentionCapture",
]
