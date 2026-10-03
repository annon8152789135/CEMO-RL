import numpy as np
import networkx as nx
from networkx.algorithms import isomorphism


def graph_dict_to_nx(graph_dict, include_terminal=False):
    G = nx.DiGraph()

    groups = graph_dict["groups"]

    if isinstance(groups, dict):
        group_iter = groups.items()
    else:
        group_iter = [(g["group"], g) for g in groups]

    valid_group_ids = set()

    # Add group nodes
    for group_id, group_info in group_iter:
        valid_group_ids.add(group_id)

        # Use translation if available, otherwise fallback
        node_label = (
            group_info.get("translation")
            if isinstance(group_info, dict)
            else str(group_info)
        )

        if node_label is None:
            node_label = str(group_info)

        G.add_node(
            group_id,
            label=str(node_label),
            node_type="group"
        )

    # Add missing terminal nodes
    if include_terminal:
        edge_node_ids = set()

        for edge in graph_dict["edges"]:
            edge_node_ids.add(edge["from_group"])
            edge_node_ids.add(edge["to_group"])

        missing_nodes = edge_node_ids - valid_group_ids

        for node_id in missing_nodes:
            G.add_node(
                node_id,
                label="TERMINAL",
                node_type="terminal"
            )

    # Add edges
    for edge in graph_dict["edges"]:
        G.add_edge(
            edge["from_group"],
            edge["to_group"],
            action=edge.get("action", None),
            probability=edge.get("probability", None)
        )

    return G


## first approach using direct comparison of graph dicts

def transition_signature(edge):
    source = edge.get("source", edge.get("from_group", edge.get("origin")))
    target = edge.get("target", edge.get("to_group", edge.get("destination")))
    action = edge.get("action", edge.get("label"))

    if source is None or target is None:
        raise KeyError(f"Could not find source/target in edge: {edge}")

    return source, target, action

def get_transition_set(graph_dict):
    return {
        transition_signature(edge)
        for edge in graph_dict["edges"]
    }


def compare_transition_sets(g1, g2):
    e1 = get_transition_set(g1)
    e2 = get_transition_set(g2)

    common = e1 & e2
    only_g1 = e1 - e2
    only_g2 = e2 - e1

    return common, only_g1, only_g2

## seccond approach using networkx isomorphism

def are_graphs_isomorphic(g1, g2):
    nx_g1 = graph_dict_to_nx(g1)
    nx_g2 = graph_dict_to_nx(g2)

    # Define a node matcher that compares the 'label' attribute
    node_matcher = isomorphism.categorical_node_match('label', None)

    # Define an edge matcher that compares the 'action' attribute
    edge_matcher = isomorphism.categorical_edge_match('action', None)

    # Create a DiGraphMatcher object
    gm = isomorphism.DiGraphMatcher(nx_g1, nx_g2, node_match=node_matcher, edge_match=edge_matcher)

    return gm.is_isomorphic()

def get_isomorphism_mapping(g1, g2):
    nx_g1 = graph_dict_to_nx(g1)
    nx_g2 = graph_dict_to_nx(g2)

    # Define a node matcher that compares the 'label' attribute
    node_matcher = isomorphism.categorical_node_match('label', None)

    # Define an edge matcher that compares the 'action' attribute
    edge_matcher = isomorphism.categorical_edge_match('action', None)

    # Create a DiGraphMatcher object
    gm = isomorphism.DiGraphMatcher(nx_g1, nx_g2, node_match=node_matcher, edge_match=edge_matcher)

    if gm.is_isomorphic():
        return gm.mapping  # Returns a dictionary mapping nodes in g1 to nodes in g2
    else:
        return None  # Not isomorphic

def get_maximum_common_subgraph(g1, g2):
    nx_g1 = graph_dict_to_nx(g1)
    nx_g2 = graph_dict_to_nx(g2)

    node_matcher = isomorphism.categorical_node_match("label", None)
    edge_matcher = isomorphism.categorical_edge_match("action", None)

    ismags = isomorphism.ISMAGS(
        nx_g1,
        nx_g2,
        node_match=node_matcher,
        edge_match=edge_matcher
    )

    mappings = list(ismags.largest_common_subgraph())

    if not mappings:
        return None, None

    mapping = mappings[0]

    # Keep only nodes that really exist in g1
    common_nodes_g1 = [
        node for node in mapping.keys()
        if node in nx_g1.nodes
    ]

    common_subgraph_g1 = nx_g1.subgraph(common_nodes_g1).copy()

    return common_subgraph_g1, mapping

def get_transition_differences_at_common_nodes(g1, g2, mapping):
    """
    Find outgoing transition differences at nodes belonging to the
    maximum common subgraph.

    mapping is assumed to be:
        g1_node -> g2_node

    Returns one entry for every common node where the outgoing
    transitions differ.
    """

    nx_g1 = graph_dict_to_nx(g1)
    print("Graph 1: \n")
    print(g1)
    print(nx_g1.nodes(data=True))
    print(nx_g1.edges(data=True))
    nx_g2 = graph_dict_to_nx(g2)
    print("Graph 2: \n")
    print(g2)
    print(nx_g2.nodes(data=True))
    print(nx_g2.edges(data=True))

    differences = []
 
    reverse_mapping = {
        g2_node: g1_node
        for g1_node, g2_node in mapping.items()
    }

    for node_g1, node_g2 in mapping.items():
 
        transitions_g1 = []

        for _, dst, data in nx_g1.out_edges(node_g1, data=True):
            transitions_g1.append({
                "from": node_g1,
                "to": dst,
                "action": data.get("action"),
                "probability": data.get("probability"),
                "mapped_destination": mapping.get(dst)
            })
 
        transitions_g2 = []

        for _, dst, data in nx_g2.out_edges(node_g2, data=True):
            transitions_g2.append({
                "from": node_g2,
                "to": dst,
                "action": data.get("action"),
                "probability": data.get("probability"),
                "mapped_destination": reverse_mapping.get(dst)
            })
 

        matched_g2 = set()
        only_g1 = []

        for i, t1 in enumerate(transitions_g1):

            match_found = False

            for j, t2 in enumerate(transitions_g2):

                if j in matched_g2:
                    continue

                # Same action
                same_action = (
                    t1["action"] == t2["action"]
                )
 
                same_destination = (
                    t1["mapped_destination"] == t2["to"]
                )

                if same_action and same_destination:
                    matched_g2.add(j)
                    match_found = True
                    break

            if not match_found:
                only_g1.append(t1)

        only_g2 = [
            t
            for j, t in enumerate(transitions_g2)
            if j not in matched_g2
        ]
 
        if only_g1 or only_g2:
            differences.append({
                "node_g1": node_g1,
                "node_g2": node_g2,
                "label": nx_g1.nodes[node_g1].get("label"),
                "transitions_g1": transitions_g1,
                "transitions_g2": transitions_g2,
                "only_g1": only_g1,
                "only_g2": only_g2
            })

    return differences


def percentage_of_common_transitions(g1, g2):
    common, only_g1, only_g2 = compare_transition_sets(g1, g2)
    total_transitions = len(common) + len(only_g1) + len(only_g2)

    if total_transitions == 0:
        return 0.0

    return (2 * len(common) / total_transitions) * 100

def percentage_of_common_nodes(g1, g2):
    common_subgraph, mapping = get_maximum_common_subgraph(g1, g2)

    if common_subgraph is None:
        return 0.0

    total_nodes = len(g1["groups"])+1 + len(g2["groups"])+1  # +1 for terminal nodes
    common_nodes = len(common_subgraph.nodes())

    return (2 * common_nodes / total_nodes) * 100

def percentage_of_common_edges(g1, g2):
    common_subgraph, mapping = get_maximum_common_subgraph(g1, g2)

    if common_subgraph is None:
        return 0.0

    total_edges = len(g1["edges"]) + len(g2["edges"])
    common_edges = len(common_subgraph.edges())

    return (2 * common_edges / total_edges) * 100

def build_common_percentage_matrices(graph_dicts):
    n = len(graph_dicts)

    common_nodes_percentage = np.zeros((n, n))
    common_edges_percentage = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                common_nodes_percentage[i, j] = 100.0
                common_edges_percentage[i, j] = 100.0
            else:
                common_nodes_percentage[i, j] = percentage_of_common_nodes(
                    graph_dicts[i], graph_dicts[j]
                )
                common_edges_percentage[i, j] = percentage_of_common_edges(
                    graph_dicts[i], graph_dicts[j]
                )

    return common_nodes_percentage, common_edges_percentage

def percentage_of_g1_nodes_in_g2(g1, g2):
    common_subgraph, mapping = get_maximum_common_subgraph(g1, g2)

    if common_subgraph is None:
        return 0.0

    total_g1_nodes = graph_dict_to_nx(g1).number_of_nodes()

    common_nodes = len(common_subgraph.nodes())

    return (common_nodes / total_g1_nodes) * 100


def percentage_of_g1_edges_in_g2(g1, g2):
    common_subgraph, mapping = get_maximum_common_subgraph(g1, g2)

    if common_subgraph is None:
        return 0.0

    total_g1_edges = len(g1["edges"])
    if total_g1_edges == 0:
        return 0.0

    common_edges = len(common_subgraph.edges())

    return (common_edges / total_g1_edges) * 100


def percentage_table_g1_in_g2(graph_dicts):
    n = len(graph_dicts)

    g1_in_g2_nodes_percentage = np.zeros((n, n))
    g1_in_g2_edges_percentage = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                g1_in_g2_nodes_percentage[i, j] = 100.0
                g1_in_g2_edges_percentage[i, j] = 100.0
            else:
                g1_in_g2_nodes_percentage[i, j] = percentage_of_g1_nodes_in_g2(
                    graph_dicts[i], graph_dicts[j]
                )
                g1_in_g2_edges_percentage[i, j] = percentage_of_g1_edges_in_g2(
                    graph_dicts[i], graph_dicts[j]
                )

    return g1_in_g2_nodes_percentage, g1_in_g2_edges_percentage

def print_percentage_table(matrix, title=None):
    n = matrix.shape[0]

    if title:
        print(title)

    header = "     " + "   ".join(f"{j:>7}" for j in range(n))
    print(header)

    for i in range(n):
        row_vals = "   ".join(f"{matrix[i, j]:>6.2f}%" for j in range(n))
        print(f"{i:>3}  {row_vals}")