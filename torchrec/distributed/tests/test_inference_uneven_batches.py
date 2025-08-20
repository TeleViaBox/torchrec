# mypy: ignore-errors
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from torchrec.utils.even_steps_iterator import EvenStepsIterator


class _Ping(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = torch.tensor([1], device=x.device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return x


def _init_pg(rank: int, world: int, backend: str = "gloo"):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(backend=backend, rank=rank, world_size=world)


def _rank_batches(rank: int):
    # rank0=2 批，rank1=4 批（不等長）
    num = [2, 4][rank]
    for _ in range(num):
        yield torch.randn(2, 4)


def _worker(rank: int, world: int):
    _init_pg(rank, world)
    try:
        model = _Ping()
        data = EvenStepsIterator(
            _rank_batches(rank), enabled=True, process_group=dist.group.WORLD
        )
        for batch in data:
            if isinstance(batch, torch.Tensor) and batch.numel() == 0:
                batch = batch.new_empty((0, 4))
            _ = model(batch)
    finally:
        dist.destroy_process_group()


@pytest.mark.timeout(20)
def test_inference_uneven_batches_spawn():
    world = 2
    ctx = mp.get_context("spawn")
    ps = []
    for r in range(world):
        p = ctx.Process(target=_worker, args=(r, world))
        p.start()
        ps.append(p)
    for p in ps:
        p.join(20)
        assert p.exitcode == 0
