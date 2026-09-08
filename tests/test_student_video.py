import numpy as np
import pytest
import torch
from torch import nn

from inference.student_video import infer_student_video


class DriftingModel(nn.Module):
    """Known scene with per-window affine disparity drift."""
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, images, include_global_points=False):
        assert not torch.is_grad_enabled()
        ids = images[0, :, 0, 0, 0]
        self.calls.append(ids.tolist())
        scene_disp = 2 + ids[:, None, None] / 100 + images[0, :, 1]
        drift = len(self.calls)
        disparity = (scene_disp - 0.1 * (drift - 1)) / drift
        return {"depth": disparity.reciprocal()[None],
                "intrinsics": torch.eye(3).repeat(1, len(ids), 1, 1)}


def scene_frames(length):
    frames = []
    for i in range(length):
        frame = torch.zeros(3, 2, 3)
        frame[0] = i
        frame[1] = torch.arange(6).reshape(2, 3) / 10
        frames.append(frame)
    return frames


@pytest.mark.parametrize("length", [1, 7, 16, 22, 24, 31, 32, 33, 44, 54, 55, 79])
def test_sequence_preserves_every_frame_and_corrects_window_drift(length):
    model = DriftingModel().eval()
    output = []
    starts = []
    def emit(start, disparities, ks):
        starts.append(start)
        output.extend(disparities.copy())
    stats = infer_student_video(model, scene_frames(length), emit, device="cpu")
    expected = 2 + np.arange(length)[:, None, None] / 100 + np.arange(6).reshape(1, 2, 3) / 10
    np.testing.assert_allclose(output, expected, rtol=1e-5)
    assert stats["output_frame_count"] == length
    assert stats["model_input_frame_count"] == 32 * len(model.calls)
    assert stats["mean_frame_inference_seconds"] == stats["model_forward_seconds"] / length
    assert starts == sorted(set(starts))
    if len(model.calls) > 1:
        assert model.calls[1][:10] == [min(i, length - 1) for i in [0, 12, 24, 25, 26, 27, 28, 29, 30, 31]]


def test_window_limit_is_a_prefix_without_padded_output_frames():
    model = DriftingModel().eval()
    output = []
    stats = infer_student_video(model, scene_frames(80), lambda i, d, k: output.extend(d),
                                device="cpu", max_windows=1)
    assert len(output) == stats["output_frame_count"] == 32
    assert stats["window_count"] == 1


def test_training_and_empty_input_rejected():
    with pytest.raises(ValueError, match="eval"):
        infer_student_video(DriftingModel(), scene_frames(2), None, device="cpu")
    with pytest.raises(ValueError, match="empty"):
        infer_student_video(DriftingModel().eval(), [], None, device="cpu")


def test_nonfinite_depth_rejected():
    class Broken(DriftingModel):
        def forward(self, *args, **kwargs):
            result = super().forward(*args, **kwargs)
            result["depth"][0, 0, 0, 0] = float("nan")
            return result
    with pytest.raises(FloatingPointError):
        infer_student_video(Broken().eval(), scene_frames(2), None, device="cpu")
