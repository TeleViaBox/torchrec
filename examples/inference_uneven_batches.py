# examples/inference_uneven_batches.py
# mypy: ignore-errors
import argparse
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

parser = argparse.ArgumentParser()
parser.add_argument(
    "--even-steps", action="store_true", help="enable EvenStepsIterator (after mode)"
)
parser.add_argument("--world-size", type=int, default=2)
parser.add_argument(
    "--rank-steps",
    default="3,5",
    help="number of batches per rank, e.g. 3,5 means rank0=3, rank1=5",
)
parser.add_argument("--backend", default="gloo")
parser.add_argument(
    "--watchdog",
    type=int,
    default=0,
    help="seconds; >0 enables self-timeout, used to safely observe hang in before mode",
)
args = parser.parse_args()


class _Ping(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = torch.ones(1, device=x.device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return x


def _init_pg(rank: int, world: int, backend: str = "gloo"):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(backend=backend, rank=rank, world_size=world)


def _rank_batches(rank: int, per_rank_steps: list[int]):
    for _ in range(per_rank_steps[rank]):
        yield torch.randn(4, 8)


def _import_even_steps_iterator():
    """
    Lazy loading: first try "file path import" (to avoid triggering torchrec top-level import and fbgemm_gpu),
    if not found then fall back to package import (works in PR/CI environments).
    """
    root = Path(__file__).resolve().parents[1]
    candidates = [
        root / "torchrec" / "utils" / "even_steps_iterator.py",  # flat
        root / "torchrec" / "torchrec" / "utils" / "even_steps_iterator.py",  # nested
    ]
    for p in candidates:
        if p.is_file():
            import importlib.util

            spec = importlib.util.spec_from_file_location("even_steps_iterator", str(p))
            mod = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
            return mod.EvenStepsIterator  # type: ignore[attr-defined]

    from torchrec.utils.even_steps_iterator import EvenStepsIterator  # type: ignore

    return EvenStepsIterator


def _worker(
    rank: int,
    world: int,
    per_rank_steps: list[int],
    use_even_steps: bool,
    backend: str,
    watchdog: int,
):
    _init_pg(rank, world, backend)
    start = time.time()
    try:
        model = _Ping()
        if use_even_steps:
            EvenStepsIterator = _import_even_steps_iterator()
            data_iter = EvenStepsIterator(
                _rank_batches(rank, per_rank_steps),
                enabled=True,
                process_group=dist.group.WORLD,
            )
        else:
            data_iter = _rank_batches(rank, per_rank_steps)

        for batch in data_iter:
            if isinstance(batch, torch.Tensor) and batch.numel() == 0:
                batch = batch.new_empty((0, 8))
            _ = model(batch)

            if watchdog and time.time() - start > watchdog:
                raise TimeoutError(
                    "watchdog timeout (Simulate the deadlock protection)"
                )

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    per_rank = [int(x) for x in args.rank_steps.split(",")]
    assert len(per_rank) == args.world_size

    mp.set_start_method("spawn", force=True)
    procs = []
    for r in range(args.world_size):
        p = mp.Process(
            target=_worker,
            args=(
                r,
                args.world_size,
                per_rank,
                args.even_steps,
                args.backend,
                args.watchdog,
            ),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
        assert p.exitcode == 0
