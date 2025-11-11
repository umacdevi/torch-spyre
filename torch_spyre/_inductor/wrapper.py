from typing import Optional

import sympy
from torch._inductor.codegen.wrapper import (
    BufferLike,
    PythonWrapperCodegen,
    SubgraphPythonWrapperCodegen,
)
from torch._inductor.ir import GraphPartitionSignature
from torch._inductor.virtualized import V
from torch._inductor.sizevars import SizeVarAllocator
from .stickify import SpyreFixedLayout


class SpyrePythonWrapperCodegen(PythonWrapperCodegen):
    def __init__(self):
        super().__init__()
        V.graph.sizevars._simplify_loops_impl = noop_simplify_loops_impl.__get__(
            V.graph.sizevars, SizeVarAllocator
        )

    @staticmethod
    def create(
        is_subgraph: bool,
        subgraph_name: Optional[str],
        parent_wrapper: Optional[PythonWrapperCodegen],
        partition_signatures: Optional[GraphPartitionSignature] = None,
    ):
        if is_subgraph:
            assert subgraph_name is not None
            assert parent_wrapper is not None
            return SubgraphPythonWrapperCodegen(
                subgraph_name, parent_wrapper, partition_signatures
            )
        return SpyrePythonWrapperCodegen()

    def write_header(self) -> None:
        super().write_header()
        self.imports.splice(
            """
                from torch_spyre._inductor.runtime import ConstantArg, TensorArg, KernelSpec, UnimplementedOp
                from torch_spyre._inductor.runtime.async_compile import SpyreAsyncCompile
                import subprocess
            """,
            strip=True,
        )
        self.header.writeline("del async_compile")
        self.header.writeline("async_compile = SpyreAsyncCompile()")

    def make_buffer_allocation(self, buffer: BufferLike):
        layout = buffer.get_layout()
        if not isinstance(layout, SpyreFixedLayout):
            return super().make_buffer_allocation(buffer)

        name = buffer.get_name()
        codegen_shape_tuple = self.codegen_python_shape_tuple(tuple(layout.size))
        codegen_allocation_shape_tuple = self.codegen_python_shape_tuple(
            tuple(layout.get_allocation_size())
        )
        codegen_stride_tuple = self.codegen_python_shape_tuple(tuple(layout.stride))

        out = (
            f"{name} = empty_strided("
            f"{codegen_allocation_shape_tuple}, "
            f"{codegen_stride_tuple}, "
            f"device='{layout.device.type}', dtype={layout.dtype})"
        )
        if codegen_shape_tuple != codegen_allocation_shape_tuple:
            out = out + f".as_strided({codegen_shape_tuple}, {codegen_stride_tuple})"

        out = out + f" {self.comment} {layout.dci}"
        return out


def noop_simplify_loops_impl(
    self, index_vars: list[sympy.Symbol], sizes, index_formulas
):
    """
    This is a noop implementation of SizeVarAllocator._simplify_loops_impl.

    We do this because the memory layout of tensors on the Spyre device is not
    entirely visible to Inductor.  Therefore Inductor's understanding of which
    tensor dimensions are actually contiguous is not accurate.
    """
    return sizes, lambda x: x, lambda x: x
