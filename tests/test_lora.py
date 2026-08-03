from __future__ import annotations

import torch
import torch.nn as nn

from models.teacher.lora import (
    LoRALinear,
    extract_lora_state_dict,
)
from models.teacher.lora_injection import (
    assert_only_lora_trainable,
    discover_mlp_linear_names,
    inject_lora_into_mlp,
    lora_config_kwargs,
    set_lora_training_mode,
)
from models.teacher.vggt_omega_wrapper import VGGTOmegaTeacher
from utils.checkpoint import load_lora_checkpoint, save_lora_checkpoint


class MLP(nn.Module):
    def __init__(self, dimension: int = 8) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dimension, dimension * 2)
        self.fc2 = nn.Linear(dimension * 2, dimension)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.nn.functional.gelu(self.fc1(value)))


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Linear(8, 8)
        self.mlp = MLP()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.mlp(value)


class PatchEmbed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([Block()])


class Aggregator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.frame_blocks = nn.ModuleList([Block()])
        self.inter_frame_blocks = nn.ModuleList([Block()])


class MockVGGTOmega(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.aggregator = Aggregator()
        self.depth_head = nn.Linear(8, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for family in (
            self.aggregator.patch_embed.blocks,
            self.aggregator.frame_blocks,
            self.aggregator.inter_frame_blocks,
        ):
            value = family[0](value)
        return self.depth_head(value)


def test_lora_shape_zero_initialization_and_freezing() -> None:
    torch.manual_seed(0)
    base = nn.Linear(7, 5)
    inputs = torch.randn(2, 3, 7)
    expected = base(inputs)
    layer = LoRALinear(base, rank=3, alpha=6, dropout=0.0)
    actual = layer(inputs)
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (2, 3, 5)
    assert not layer.base_layer.weight.requires_grad
    assert layer.lora_A.requires_grad
    assert layer.lora_B.requires_grad
    assert layer.lora_A.shape == (3, 7)
    assert layer.lora_B.shape == (5, 3)


def test_lora_matches_endodac_linear_formula() -> None:
    torch.manual_seed(3)
    base = nn.Linear(6, 4)
    layer = LoRALinear(base, rank=2, alpha=1, dropout=0.0)
    with torch.no_grad():
        layer.lora_A.normal_()
        layer.lora_B.normal_()
    inputs = torch.randn(2, 5, 6)
    expected = base(inputs) + (
        inputs @ layer.lora_A.T @ layer.lora_B.T
    ) * (1.0 / 2.0)
    torch.testing.assert_close(layer(inputs), expected)


def test_vggt_mlp_discovery_excludes_attention_and_heads() -> None:
    model = MockVGGTOmega()
    names = discover_mlp_linear_names(model)
    assert names == [
        "aggregator.patch_embed.blocks.0.mlp.fc1",
        "aggregator.patch_embed.blocks.0.mlp.fc2",
    ]
    expected = model(torch.randn(2, 4, 8))
    injected = inject_lora_into_mlp(model, rank=2, alpha=4)
    actual = model(torch.randn(2, 4, 8))
    assert injected == names
    assert actual.shape == expected.shape
    summary = assert_only_lora_trainable(model)
    assert summary.trainable_names
    assert all("lora_" in name for name in summary.trainable_names)
    assert not model.depth_head.weight.requires_grad
    assert not model.aggregator.frame_blocks[0].mlp.fc1.weight.requires_grad
    assert not model.aggregator.inter_frame_blocks[0].mlp.fc1.weight.requires_grad
    set_lora_training_mode(model)
    assert not model.depth_head.training
    for module in model.modules():
        if isinstance(module, LoRALinear):
            assert module.training
            assert module.dropout.training
            assert not module.base_layer.training
    assert set(extract_lora_state_dict(model)) == {
        "{}.{}".format(name, branch)
        for name in names
        for branch in ("lora_A", "lora_B")
    }


def test_lora_only_checkpoint_round_trip(tmp_path) -> None:
    source = MockVGGTOmega()
    inject_lora_into_mlp(source, rank=2, alpha=4)
    with torch.no_grad():
        for name, parameter in source.named_parameters():
            if name.endswith(".lora_B"):
                parameter.fill_(0.25)
    path = tmp_path / "teacher_lora.pt"
    save_lora_checkpoint(path, source, 3, 17, None, None, {"test": True})

    destination = MockVGGTOmega()
    inject_lora_into_mlp(destination, rank=2, alpha=4)
    state = load_lora_checkpoint(path, destination)
    assert state["epoch"] == 3
    assert state["global_step"] == 17
    source_state = extract_lora_state_dict(source)
    destination_state = extract_lora_state_dict(destination)
    assert source_state.keys() == destination_state.keys()
    for name in source_state:
        torch.testing.assert_close(source_state[name], destination_state[name])


def test_endodac_standard_lora_defaults_and_rejects_dvlora() -> None:
    kwargs = lora_config_kwargs({"type": "lora", "rank": 4})
    assert kwargs["rank"] == 4
    assert kwargs["alpha"] == 1.0
    assert kwargs["target_layers"] == "all"

    try:
        lora_config_kwargs({"type": "dvlora"})
    except ValueError as error:
        assert "DV-LoRA" in str(error)
    else:
        raise AssertionError("DV-LoRA configuration must be rejected")


def test_base_teacher_skips_lora_and_freezes_everything(monkeypatch) -> None:
    import models.teacher.vggt_omega_wrapper as wrapper_module

    reference_state = MockVGGTOmega().state_dict()
    monkeypatch.setattr(
        wrapper_module,
        "_import_vggt_omega_class",
        lambda: MockVGGTOmega,
    )
    monkeypatch.setattr(
        wrapper_module,
        "_checkpoint_state",
        lambda path: reference_state,
    )
    teacher = VGGTOmegaTeacher.from_config(
        {
            "pretrained_checkpoint": "unused-by-test.pt",
            "freeze_backbone": True,
            "freeze_heads": True,
            "lora_checkpoint": "must-not-be-read.pt",
            "lora": {
                "enabled": True,
                "type": "lora",
                "rank": 4,
                "alpha": 1,
            },
        },
        load_lora=False,
        inject_lora=False,
    )

    assert not teacher.uses_lora
    assert teacher.injected_module_names == ()
    assert not any(isinstance(module, LoRALinear) for module in teacher.modules())
    assert not any(parameter.requires_grad for parameter in teacher.parameters())


def test_base_teacher_rejects_lora_checkpoint_loading(monkeypatch) -> None:
    import models.teacher.vggt_omega_wrapper as wrapper_module

    reference_state = MockVGGTOmega().state_dict()
    monkeypatch.setattr(
        wrapper_module,
        "_import_vggt_omega_class",
        lambda: MockVGGTOmega,
    )
    monkeypatch.setattr(
        wrapper_module,
        "_checkpoint_state",
        lambda path: reference_state,
    )
    try:
        VGGTOmegaTeacher.from_config(
            {
                "pretrained_checkpoint": "unused-by-test.pt",
                "freeze_backbone": True,
                "freeze_heads": True,
            },
            load_lora=True,
            inject_lora=False,
        )
    except ValueError as error:
        assert "load_lora must be false" in str(error)
    else:
        raise AssertionError("Base teacher must not load a LoRA checkpoint")
