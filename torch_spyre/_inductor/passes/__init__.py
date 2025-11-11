from typing import Optional, Any, Callable, List

import torch
from torch._inductor.custom_graph_pass import (
    CustomGraphPass,
    get_hash_for_files,
)

from .conversions import RemoveElementTypeConversions


custom_pre_passes: List[Callable[[torch.fx.graph.Graph], None]] = [
    RemoveElementTypeConversions(),
]
custom_post_passes: List[Callable[[torch.fx.graph.Graph], None]] = []


class CustomPrePasses(CustomGraphPass):
    def __call__(self, graph: torch.fx.graph.Graph) -> None:
        for p in custom_pre_passes:
            p(graph)

    def uuid(self) -> Optional[Any]:
        files = [c.file() for c in custom_pre_passes]
        files = tuple(set(files + [__file__]))
        return get_hash_for_files(files)


class CustomPostPasses(CustomGraphPass):
    def __call__(self, graph: torch.fx.graph.Graph) -> None:
        for p in custom_post_passes:
            p(graph)

    def uuid(self) -> Optional[Any]:
        files = [c.file() for c in custom_post_passes]
        files = tuple(set(files + [__file__]))
        return get_hash_for_files(files)
