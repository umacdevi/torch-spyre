from typing import Any, Callable, Optional, Sequence

from sympy import Expr
import torch
from torch._inductor.utils import ir_dataclass
from torch._inductor.ir import (
    IRNode,
    Pointwise,
    Reduction,
    ReductionHint,
    TensorBox,
)


@ir_dataclass
class SpyrePointwise(Pointwise):
    op_info: Any


@ir_dataclass
class SpyreReduction(Reduction):
    op_info: Any

    @classmethod
    def create(
        cls,
        device: torch.device,
        dst_dtype: torch.dtype,
        src_dtype: torch.dtype,
        inner_fn: Callable[..., Any],
        ranges: Sequence[Expr],
        reduction_ranges: Sequence[Expr],
        reduction_type,
        op_info=None,
        reduction_hint: ReductionHint = ReductionHint.DEFAULT,
        input_node: Optional[IRNode] = None,
    ) -> TensorBox:
        return TensorBox.create(
            SpyreReduction(
                device=device,
                dtype=dst_dtype,
                inner_fn=inner_fn,
                ranges=ranges,
                reduction_ranges=reduction_ranges,
                reduction_type=reduction_type,
                src_dtype=src_dtype,
                reduction_hint=reduction_hint,
                op_info=op_info,
            )
        )
