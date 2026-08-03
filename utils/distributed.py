"""Minimal distributed-runtime helpers shared by both training stages."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def initialize_distributed(backend: str = "nccl") -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return DistributedContext(False, 0, 0, 1)
    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed is unavailable")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend=backend, init_method="env://")
    return DistributedContext(
        True,
        torch.distributed.get_rank(),
        local_rank,
        torch.distributed.get_world_size(),
    )


def barrier(context: DistributedContext) -> None:
    if context.enabled:
        torch.distributed.barrier()
