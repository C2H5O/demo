"""CPU regression checks for the training-scoped SDPA policy."""
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import trainers.direct_teacher_distillation_trainer as trainer


@pytest.mark.parametrize("raise_error", [False, True])
def test_attention_training_disables_flash_and_restores_state(monkeypatch, raise_error):
    original = torch.backends.cuda.flash_sdp_enabled()
    config = {"attention_distill": {"enabled": True}, "flash_attention": {"enabled": True}}
    monkeypatch.setattr(trainer, "load_config", lambda path: config)

    def fake_train(config, dry_run, resume, max_steps):
        assert not config["flash_attention"]["enabled"]
        assert not torch.backends.cuda.flash_sdp_enabled()
        q = torch.randn(1, 2, 4, 8, requires_grad=True)
        def attention(value):
            assert not torch.backends.cuda.flash_sdp_enabled()
            return F.scaled_dot_product_attention(value, value, value)

        checkpoint(attention, q, use_reentrant=False).sum().backward()
        assert q.grad is not None and torch.isfinite(q.grad).all()
        if raise_error:
            raise RuntimeError("test failure")
        return {"ok": True}

    monkeypatch.setattr(trainer, "_train_direct_teacher_distillation", fake_train)
    if raise_error:
        with pytest.raises(RuntimeError, match="test failure"):
            trainer.train_direct_teacher_distillation(Path("unused.yaml"))
    else:
        assert trainer.train_direct_teacher_distillation(Path("unused.yaml")) == {"ok": True}
    assert torch.backends.cuda.flash_sdp_enabled() == original


def test_non_attention_training_keeps_backend_policy(monkeypatch):
    original = torch.backends.cuda.flash_sdp_enabled()
    config = {"attention_distill": {"enabled": False}}
    monkeypatch.setattr(trainer, "load_config", lambda path: config)

    def fake_train(config, *args):
        assert torch.backends.cuda.flash_sdp_enabled() == original
        assert "flash_attention" not in config
        return {}

    monkeypatch.setattr(trainer, "_train_direct_teacher_distillation", fake_train)
    trainer.train_direct_teacher_distillation(Path("unused.yaml"))
