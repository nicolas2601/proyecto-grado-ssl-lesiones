"""torchrun-aware setup with a transparent single-process fallback.

Run with ``torchrun --nproc_per_node=2 script.py ...`` for two GPUs or plain
``python script.py ...`` for one GPU (or CPU). The same code path serves both.
"""

from __future__ import annotations

import os
import random
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def distributed(self) -> bool:
        return self.world_size > 1


def init_distributed(*, force_cpu: bool = False) -> DistContext:
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    use_cuda = torch.cuda.is_available() and not force_cpu
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl" if use_cuda else "gloo")
    return DistContext(rank, world_size, local_rank, device)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_generator(context: DistContext, *seed_parts: int) -> torch.Generator:
    """Per-rank generator derived from ``seed_parts`` (seed, epoch, ...)."""
    generator = torch.Generator(device=context.device)
    combined = 0
    for part in (*seed_parts, context.rank):
        combined = (combined * 1_000_003 + int(part) + 1) % (2**63 - 1)
    generator.manual_seed(combined)
    return generator


def autocast_context(device: torch.device, precision: str):
    """bf16 autocast on CUDA; a no-op on CPU or when precision is ``fp32``."""
    if device.type == "cuda" and precision == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def wrap_ddp(module: torch.nn.Module, context: DistContext) -> torch.nn.Module:
    if not context.distributed:
        return module
    device_ids = [context.local_rank] if context.device.type == "cuda" else None
    return torch.nn.parallel.DistributedDataParallel(
        module, device_ids=device_ids, broadcast_buffers=False
    )


def barrier(context: DistContext) -> None:
    if context.distributed:
        dist.barrier()


def all_reduce_sum(tensor: torch.Tensor, context: DistContext) -> torch.Tensor:
    if context.distributed:
        tensor = tensor.clone()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def all_gather_cat(tensor: torch.Tensor, context: DistContext) -> torch.Tensor:
    """Concatenate equally sized tensors from every rank along dim 0 (no gradient)."""
    if not context.distributed:
        return tensor
    gathered = [torch.empty_like(tensor) for _ in range(context.world_size)]
    dist.all_gather(gathered, tensor.contiguous())
    return torch.cat(gathered, dim=0)


def cleanup(context: DistContext) -> None:
    if context.distributed and dist.is_initialized():
        dist.destroy_process_group()
