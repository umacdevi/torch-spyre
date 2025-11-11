import dataclasses
import torch


@dataclasses.dataclass
class TensorArg:
    is_input: bool
    arg_index: int
    dtype: torch.dtype


@dataclasses.dataclass
class ConstantArg:
    value: str
    dtype: str


@dataclasses.dataclass
class KernelSpec:
    op: str
    is_reduction: bool
    dimensions: list[int]
    args: list[TensorArg | ConstantArg]
    scales: list[list[int]]
    op_info: dict[list, str]


@dataclasses.dataclass
class UnimplementedOp:
    op: str
