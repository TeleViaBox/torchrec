# mypy: ignore-errors
from __future__ import annotations

from typing import Any, Callable, Generic, Iterable, Iterator, Optional, TypeVar

import torch
import torch.distributed as dist

T = TypeVar("T")

try:
    from torchrec.sparse.jagged_tensor import (  # type: ignore[assignment]
        KeyedJaggedTensor,
    )
except Exception:
    KeyedJaggedTensor = None  # type: ignore[assignment]


def _empty_like(obj: Any) -> Any:
    """
    Create "empty batch":
      - Tensor  -> keep subsequent dimensions, set batch dimension to 0
      - dict    -> recursively process values
      - list/tuple -> recursively process elements and return same container type
      - KeyedJaggedTensor -> use official empty_like (if available)
    """
    if isinstance(obj, torch.Tensor):
        return obj.new_empty((0,) + tuple(obj.shape[1:]))

    if KeyedJaggedTensor is not None and isinstance(obj, KeyedJaggedTensor):
        return KeyedJaggedTensor.empty_like(obj)  # type: ignore[union-attr]

    if isinstance(obj, dict):
        return {k: _empty_like(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        seq = [_empty_like(v) for v in obj]
        return type(obj)(seq)

    raise TypeError(f"Don't know how to create an empty batch for type: {type(obj)}")


class EvenStepsIterator(Generic[T]):
    """
    Purpose: During inference, if some ranks run out of data first, still continue outputting "empty batches" to participate in collective,
            until all ranks have no data, ensuring the number of collectives per step is consistent → avoid hang.

    Usage (minimal change):
        data_iter = EvenStepsIterator(dataloader, enabled=True, process_group=pg)
        for batch in data_iter:
            _ = model(batch)  # even if it's an empty batch, still call forward to participate in collective

    Parameters:
    - enabled: default False; only when turned on will all_reduce sync be performed.
    - process_group: existing dist process group (usually dist.group.WORLD).
    - make_empty: (optional) custom empty batch factory, suitable when batch contains custom types or needs customized "empty samples".
    - device: device of the tensor used for all_reduce. Default: current GPU if CUDA exists, otherwise CPU.
    - dtype: dtype of the all_reduce flag (default int64).
    """

    def __init__(
        self,
        iterable: Iterable[T],
        *,
        enabled: bool = False,
        process_group: Optional[dist.ProcessGroup] = None,
        make_empty: Optional[Callable[[Optional[T]], T]] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.int64,
    ) -> None:
        self._src = iter(iterable)
        self._pg = (
            process_group if (dist.is_available() and dist.is_initialized()) else None
        )
        self._enabled = bool(enabled) and (self._pg is not None)
        self._make_empty = make_empty
        self._device = (
            device
            if device is not None
            else (
                torch.device(torch.cuda.current_device())
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        )
        self._dtype = dtype

    def __iter__(self) -> Iterator[T]:
        last_real: Optional[T] = None
        finished_local = False

        def advance() -> Optional[T]:
            nonlocal finished_local, last_real
            if finished_local:
                return None
            try:
                nxt = next(self._src)
                last_real = nxt
                return nxt
            except StopIteration:
                finished_local = True
                return None

        while True:
            buf = advance()
            alive_local = 1 if buf is not None else 0

            if not self._enabled:
                if alive_local == 0:
                    break
                yield buf  # type: ignore[arg-type]
                continue

            alive = torch.tensor([alive_local], device=self._device, dtype=self._dtype)
            dist.all_reduce(alive, op=dist.ReduceOp.SUM, group=self._pg)
            world_alive = int(alive.item())
            if world_alive == 0:
                break

            if alive_local == 1:
                yield buf  # type: ignore[arg-type]
            else:
                if self._make_empty is not None:
                    empty = self._make_empty(last_real)
                else:
                    if last_real is None:
                        raise RuntimeError(
                            "Cannot infer empty batch without seeing a real batch. "
                            "Pass make_empty=... (especially if your batch contains custom types/KJT)."
                        )
                    empty = _empty_like(last_real)
                yield empty
