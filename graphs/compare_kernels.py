import copy
from grakel import Graph
from grakel.kernels import WeisfeilerLehman, VertexHistogram, SubgraphMatching
from graphs.utils import parse_graph_dict, print_kernel_table
from sklearn.preprocessing import normalize

def add_missing_terminal_nodes(graph_dict, terminal_label="TERMINAL"):
    graph_dict = dict(graph_dict)
    graph_dict["groups"] = list(graph_dict["groups"])
    graph_dict["edges"] = list(graph_dict["edges"])

    group_ids = {g["group"] for g in graph_dict["groups"]}

    referenced_nodes = set()
    for edge in graph_dict["edges"]:
        referenced_nodes.add(edge["from_group"])
        referenced_nodes.add(edge["to_group"])

    missing_nodes = referenced_nodes - group_ids

    for node_id in missing_nodes:
        graph_dict["groups"].append({
            "group": node_id,
            "translation": terminal_label
        })

    return graph_dict

def compare_explanation_graphs(
    graph_dicts,
    node_label_mode="translation",
    edge_label_mode="action"
):
    graphs = [
        parse_graph_dict(
            add_missing_terminal_nodes(g),
            node_label_mode=node_label_mode,
            edge_label_mode=edge_label_mode
        )
        for g in graph_dicts
    ]

    WLkernel = WeisfeilerLehman(
        base_graph_kernel=VertexHistogram,
        normalize=True
    )

    SMkernel = SubgraphMatching(normalize=True)

    K_wl = WLkernel.fit_transform(graphs)
    K_sm = SMkernel.fit_transform(graphs)

    return K_wl, K_sm 