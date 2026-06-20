import math
from typing import List, Optional

import networkx as nx

from .GraphBasedModel import GraphBasedModelV2


def _parse_hidden_layers(hidden_layers) -> List[int]:
    if isinstance(hidden_layers, str):
        return [int(part.strip()) for part in hidden_layers.split(",") if part.strip()]
    return [int(width) for width in hidden_layers]


def build_synthetic_binn_graph(input_size: int, hidden_layers, fan_in: Optional[int] = None) -> nx.MultiDiGraph:
    """
    Adapter-side graph builder for synthetic or placeholder runs.

    The actual BINN model implementation remains the copied
    ``GraphBasedModelV2``. This helper only constructs a DAG with the edge
    orientation that model expects, so FL can run before the medical ontology
    files arrive.
    """
    hidden_layers = _parse_hidden_layers(hidden_layers)
    graph = nx.MultiDiGraph()

    layer_nodes: List[List[str]] = []
    input_nodes = [f"gene_{idx}" for idx in range(int(input_size))]
    graph.add_nodes_from(input_nodes, type="Entity")
    layer_nodes.append(input_nodes)

    for layer_idx, width in enumerate(hidden_layers, start=1):
        current_nodes = [f"layer{layer_idx}_node{node_idx}" for node_idx in range(int(width))]
        graph.add_nodes_from(current_nodes, type="Term")
        layer_nodes.append(current_nodes)

    for layer_idx in range(1, len(layer_nodes)):
        previous_nodes = layer_nodes[layer_idx - 1]
        current_nodes = layer_nodes[layer_idx]
        current_fan_in = fan_in or max(2, math.ceil(len(previous_nodes) / max(len(current_nodes), 1)))
        current_fan_in = min(len(previous_nodes), max(1, current_fan_in))

        for node_idx, node_name in enumerate(current_nodes):
            center = int(round(node_idx * len(previous_nodes) / max(len(current_nodes), 1)))
            for offset in range(current_fan_in):
                parent_name = previous_nodes[(center + offset) % len(previous_nodes)]
                graph.add_edge(parent_name, node_name)
            graph.add_edge(previous_nodes[(node_idx * 997 + layer_idx * 37) % len(previous_nodes)], node_name)

    graph.name = "SyntheticBINN"
    return graph


class BINNAdapter(GraphBasedModelV2):
    task_type = "binary"

    def __init__(
        self,
        input_size: int = 128,
        output_size: int = 1,
        hidden_layers="128,64,32",
        dropout_prob: float = 0.2,
        non_linearity: str = "relu",
        output_last_layers: int = 1,
        graph: Optional[nx.MultiDiGraph] = None,
        **kwargs,
    ):
        if graph is None:
            graph = build_synthetic_binn_graph(input_size, hidden_layers)
        super().__init__(
            graph=graph,
            output_size=output_size,
            output_last_layers=output_last_layers,
            dropout_prob=dropout_prob,
            non_linearity=non_linearity,
            **kwargs,
        )
        self.task_type = "binary" if int(output_size) == 1 else "multiclass"


def binn(
    num_classes=None,
    input_size: int = 128,
    output_size: Optional[int] = None,
    hidden_layers="128,64,32",
    dropout_prob: float = 0.2,
    non_linearity: str = "relu",
    output_last_layers: int = 1,
    graph: Optional[nx.MultiDiGraph] = None,
    **kwargs,
):
    if output_size is None:
        output_size = 1 if num_classes in (None, 1, 2) else int(num_classes)
    return BINNAdapter(
        input_size=input_size,
        output_size=output_size,
        hidden_layers=hidden_layers,
        dropout_prob=dropout_prob,
        non_linearity=non_linearity,
        output_last_layers=output_last_layers,
        graph=graph,
        **kwargs,
    )
