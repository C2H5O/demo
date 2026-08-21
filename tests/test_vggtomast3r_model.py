from __future__ import annotations

import torch
import torch.nn as nn

from models.student.dune_mast3r_adapter import DuneMast3RStudent
from trainers.vggtomast3r_trainer import build_v1_optimizer


class _MockMast3R(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = nn.Linear(1, 1)
        self.enc_blocks = nn.ModuleList([nn.Linear(1, 1)])
        self.decoder_embed = nn.Linear(1, 1)
        self.dec_blocks = nn.ModuleList([nn.Linear(1, 1)])
        self.dec_blocks2 = nn.ModuleList([nn.Linear(1, 1)])
        self.dec_norm = nn.LayerNorm(1)
        self.downstream_head1 = nn.Linear(1, 1)
        self.downstream_head2 = nn.Linear(1, 1)


class _MockOfficial(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dune_backbone = nn.Sequential(nn.Linear(1, 2), nn.Dropout())
        self.mast3r = _MockMast3R()

    def forward(self, view1, view2):
        image1, image2 = view1["img"], view2["img"]
        b, _, h, w = image1.shape
        assert view1["instance"] == ["reference_{}".format(i) for i in range(b)]
        assert view2["instance"] == ["other_{}".format(i) for i in range(b)]
        gain = self.mast3r.downstream_head1.weight.reshape(1, 1, 1)
        ref_z = image1.mean(dim=1) * gain
        other_z = image2.mean(dim=1) * 0 + 7.0 * gain
        zeros = torch.zeros(b, h, w, 2, device=image1.device)
        return {"pts3d": torch.cat((zeros, ref_z.unsqueeze(-1)), -1)}, {"pts3d_in_other_view": torch.cat((zeros, other_z.unsqueeze(-1)), -1)}


def _student() -> DuneMast3RStudent:
    return DuneMast3RStudent({}, model_factory=_MockOfficial)


def test_dune_mast3r_output_shapes() -> None:
    model = _student()
    outputs = model(torch.zeros(1, 2, 3, 448, 560))
    assert outputs["pts3d_ref"].shape == (1, 448, 560, 3)
    assert outputs["pts3d_other_in_ref"].shape == (1, 448, 560, 3)


def test_official_view_metadata_disables_false_symmetrization() -> None:
    student = _student()
    image = torch.zeros(4, 3, 448, 560)

    reference = student._view(image, "reference")
    other = student._view(image, "other")

    assert len(reference["instance"]) == len(other["instance"]) == 4
    assert all(left != right for left, right in zip(reference["instance"], other["instance"]))
    # Mirror the official is_symmetrized predicate: the ordinary batch must
    # not look like interleaved (A,B),(B,A) pairs.
    assert not all(
        reference["instance"][i] == other["instance"][i + 1]
        and reference["instance"][i + 1] == other["instance"][i]
        for i in range(0, 4, 2)
    )


def test_dune_frozen_and_eval_during_train() -> None:
    model = _student().train()
    assert not model.dune_encoder.training
    assert all(not parameter.requires_grad for parameter in model.dune_encoder.parameters())


def test_only_decoder_head_trainable() -> None:
    model = _student()
    trainable = [name for name, parameter in model.model.named_parameters() if parameter.requires_grad]
    assert trainable
    assert all(name.startswith(("mast3r.decoder", "mast3r.dec_", "mast3r.downstream_head")) for name in trainable)
    optimizer = build_v1_optimizer(model, {"learning_rate": 1e-4, "weight_decay": 0.05})
    optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert not optimizer_ids.intersection(id(p) for p in model.dune_encoder.parameters())


def test_reference_depth_is_z() -> None:
    model = _student()
    images = torch.ones(1, 2, 3, 448, 560)
    output = model(images)
    assert torch.equal(model.reference_depth(images), output["pts3d_ref"][..., 2])


def test_other_in_ref_is_not_other_camera_depth() -> None:
    model = _student()
    images = torch.stack((torch.ones(3, 448, 560), torch.full((3, 448, 560), 2.0))).unsqueeze(0)
    forward = model(images)
    reverse = model(images.flip(1))
    assert not torch.equal(forward["pts3d_other_in_ref"][..., 2], reverse["pts3d_ref"][..., 2])


def test_resolution_448x560() -> None:
    model = _student()
    try:
        model(torch.zeros(1, 2, 3, 448, 448))
    except ValueError as error:
        assert "448x560" in str(error)
    else:
        raise AssertionError("Square input must be rejected")
