# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import contextmanager

import torch
from torch._inductor.graph import GraphLowering
from torch._inductor.utils import InputType
from torch._inductor.virtualized import V
from typing import Callable, Optional

from torch._inductor.sizevars import SizeVarAllocator
from torch._inductor.utils import sympy_index_symbol_with_prefix, SymT
from torch._inductor import dependencies


@contextmanager
def spyre_data_types():
    saved = torch._prims_common._computation_dtype_map
    torch._prims_common._computation_dtype_map = {
        torch.bfloat16: torch.bfloat16,
        torch.float16: torch.float16,
        torch.complex32: torch.complex32,
    }
    try:
        yield
    finally:
        torch._prims_common._computation_dtype_map = saved


@contextmanager
def enable_spyre_context(
    example_inputs: list[InputType],
    decomps: Optional[dict[torch._ops.OperatorBase, Callable]] = None,
):
    """
    Context manager that sets up the complete Spyre compilation environment.

    This CM configures PyTorch Inductor to compile graphs for the Spyre device by:
      - Enabling Spyre-specific data type handling
      - Activating Spyre lowerings and decompositions
      - Configuring Inductor settings optimized for Spyre
      - Setting up custom pre/post compilation passes
      - Disabling incompatible optimizations (e.g., reduction splitting, permute fusion)

    Args:
        example_inputs: List of example inputs to the graph being compiled. Used to
            set real inputs in the virtualized context for shape inference and
            optimization decisions.
        decomps: Decomposition table to be populated with Spyre-specific
            decompositions. Maps operator overloads to their decomposition implementations.
            This is typically a clone of PyTorch Inductor's global decomposition registry.
    """

    if decomps is None:
        decomps = torch._inductor.decomposition.decompositions

    from torch_spyre._inductor.lowering import enable_spyre_lowerings  # your CM

    # Ensure decorators run (custom ops/decomp/lowerings modules)
    import torch_spyre._inductor.customops  # noqa: F401
    from torch_spyre._inductor.decompositions import (
        enable_spyre_decompositions,
    )

    import torch_spyre._inductor.lowering  # noqa: F401
    from torch_spyre._inductor.choices import SpyreHeuristics
    from torch_spyre._inductor.passes import (
        CustomPreGradPasses,
        CustomPrePasses,
        CustomPostPasses,
        CustomPreFusionPasses,
        CustomPostFusionPasses,
        CustomPreSchedulingPasses,
    )

    # *) Inductor config tweaks (saved/restored)
    new_config = {
        "split_reductions": False,
        "benchmark_harness": False,
        "pre_grad_custom_pass": CustomPreGradPasses(),
        "post_grad_custom_pre_pass": CustomPrePasses(),
        "post_grad_custom_post_pass": CustomPostPasses(),
        "_pre_fusion_custom_pass": CustomPreFusionPasses(),
        "_post_fusion_custom_pass": CustomPostFusionPasses(),
        # Adding this configuration in so as to avoid the optimization of turning small matmuls into non-matmuls
        # found here: https://github.com/pytorch/pytorch/blob/main/torch/_inductor/ir.py#L1580
        "unroll_reductions_threshold": 1,
        # Disable fusing of mm + permute/transpose for now.
        "permute_fusion": False,
        "allow_buffer_reuse": False,  # For now, as buffer reuse does not consider stride_map.
    }

    from torch._inductor.ir import Loops

    # Force all operations to be realized when LoopLevel IR is initially constructed
    old_loop = Loops.has_large_inner_fn
    Loops.has_large_inner_fn = lambda self, threshold=None: True

    '''
    # === NEW: Preserve size-1 dimensions in index variables ===
    old_index = Loops._index

    @staticmethod
    def _spyre_index(ranges, prefix=SymT.INDEX):
        """
        Preserve all index variables including those with range=1.

        The default implementation replaces size-1 dims with sympy.S.Zero.
        Spyre needs all dimensions for correct SDSC coordinate generation.
        """
        from torch._inductor.utils import sympy_index_symbol_with_prefix, SymT
        return [
            sympy_index_symbol_with_prefix(prefix, n)
            for n, s in enumerate(ranges)
        ]

    Loops._index = _spyre_index
    # === END NEW ===

    # === NEW: Patch _simplify_loops_impl BEFORE Scheduler construction ===
    def noop_simplify_loops_impl(self, index_vars, sizes, index_formulas):
        """
        No-op implementation that preserves all dimensions including size-1.
        Must be applied before Scheduler construction.
        """
        return sizes, lambda x: x, lambda x: x

    old_simplify_loops_impl = SizeVarAllocator._simplify_loops_impl
    SizeVarAllocator._simplify_loops_impl = noop_simplify_loops_impl
    # === END NEW ===


    ### Prevent size-1 dimension elimination
    old_simplify_and_reorder = SizeVarAllocator._simplify_loops_impl

    def _spyre_simplify_loops_impl(self, sizes, index_vars, index_formulas):
        """
        Preserve all dimensions including size-1.
        
        Inductor's default drops size-1 dims and merges contiguous dims.
        Spyre needs all original dimensions for correct SDSC generation,
        especially for convolution where G=1 must be retained.
        """
        print(f"######## In _spyre_simplify_loops_impl ########################")
        # Return unchanged - don't filter size-1 dims, don't merge dims
        return sizes, lambda x: x, lambda x: x

    SizeVarAllocator._simplify_loops_impl = _spyre_simplify_loops_impl
    ### End of Prevent size-1 dimension elimination

    # === NEW: Prevent index_vars_squeeze from dropping size-1 dims ===
    old_index_vars_squeeze = dependencies.index_vars_squeeze
    
    # Replace with no_squeeze version
    dependencies.index_vars_squeeze = dependencies.index_vars_no_squeeze
    # === END NEW ===
    '''

    from torch._inductor.fx_passes import joint_graph

    origin_pass = list(joint_graph.pass_patterns)
    # disable mul_softmax_pattern and div_softmax_pattern for now
    joint_graph.pass_patterns.pop()

    # Inject the pre_scheduling_passes before the Scheduler is constructed,
    # allowing the passes to modify the graph IR (buffers, inputs, constants).
    old_update_scheduler = GraphLowering._update_scheduler

    _pre_scheduling_pass = CustomPreSchedulingPasses()

    def _spyre_update_scheduler(self: GraphLowering) -> None:
        '''
        # Patch the instance to ensure this specific graph uses noop
        self.sizevars._simplify_loops_impl = lambda index_vars, sizes, index_formulas: (
         sizes, lambda x: x, lambda x: x
        )
        '''
        _pre_scheduling_pass(self.operations)
        old_update_scheduler(self)

    GraphLowering._update_scheduler = _spyre_update_scheduler  # type: ignore[method-assign]

    with (
        spyre_data_types(),
        enable_spyre_lowerings(),
        enable_spyre_decompositions(decomps=decomps) as spyre_context_decompositions,
        V.set_real_inputs(example_inputs),
        V.set_choices_handler(SpyreHeuristics()),
        torch._inductor.config.patch(new_config),
    ):
        try:
            yield spyre_context_decompositions
        finally:
            joint_graph.pass_patterns[:] = origin_pass
            Loops.has_large_inner_fn = old_loop
            GraphLowering._update_scheduler = old_update_scheduler  # type: ignore[method-assign]
