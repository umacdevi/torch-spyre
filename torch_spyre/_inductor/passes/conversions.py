import torch

aten = torch.ops.aten
prims = torch.ops.prims


class RemoveElementTypeConversions:
    """
    We assume all operations on the device are in fp16.
    Noop all data conversion operations.
    """

    def __call__(self, graph: torch.fx.graph.Graph) -> None:
        conversion_ops = [prims.convert_element_type.default]
        conversion_nodes = [n for n in graph.nodes if n.target in conversion_ops]
        for node in conversion_nodes:
            print(f"WARNING: NO-OPING: {node}")
            input = node.args[0]
            node.replace_all_uses_with(input)
            graph.erase_node(node)

    def file(self) -> str:
        return __file__
