from __future__ import annotations

import torch
import torch.nn as nn

from trainers.teacher_lora_trainer import (
    _float_prediction_tensors,
    _gradient_issues,
)


def test_prediction_promotion_preserves_gradient_graph() -> None:
    source = torch.ones(3, dtype=torch.float16, requires_grad=True)
    promoted = _float_prediction_tensors(
        {"depth": source * 2, "mask": torch.ones(3, dtype=torch.bool)}
    )

    assert promoted["depth"].dtype == torch.float32
    assert promoted["mask"].dtype == torch.bool
    promoted["depth"].sum().backward()
    torch.testing.assert_close(source.grad, torch.full_like(source, 2))


def test_gradient_issues_separates_missing_and_non_finite() -> None:
    model = nn.Module()
    model.good = nn.Parameter(torch.ones(1))
    model.missing = nn.Parameter(torch.ones(1))
    model.bad = nn.Parameter(torch.ones(1))
    model.good.grad = torch.ones_like(model.good)
    model.bad.grad = torch.full_like(model.bad, torch.nan)

    missing, non_finite = _gradient_issues(model)

    assert missing == ["missing"]
    assert non_finite == ["bad"]
