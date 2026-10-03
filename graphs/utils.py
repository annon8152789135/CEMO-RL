import re
import numpy as np
from grakel import Graph 
import re

def parse_transition_lines(lines):
    """
    lines: list[str] like:
      'Group 1 to Group 2 with p=1.0 and action 2'
    Returns:
      edges: dict[(int,int) -> float]   # transition probability
      edge_labels: dict[(int,int) -> str]  # action as string
      node_labels: dict[int -> str]     # WL vertex labels
    """
    edges = {}
    edge_labels = {}
    node_labels = {}

    pattern = re.compile(
        r"Group\s+(?P<src>\d+)\s+to\s+Group\s+(?P<dst>\d+)\s+with p=(?P<prob>[0-9eE\.\-]+)\s+and action\s+(?P<action>\d+)"
    )

    for line in lines:
        line = line.strip()
        if not line:
            continue
        m = pattern.match(line)
        if not m:
            raise ValueError(f"Cannot parse line: {line}")

        src = int(m.group("src"))
        dst = int(m.group("dst"))
        prob = float(m.group("prob"))
        action = m.group("action")   # keep as string label

        edges[(src, dst)] = prob
        edge_labels[(src, dst)] = action

        # Create simple categorical node labels for WL
        if src not in node_labels:
            node_labels[src] = f"G{src}"
        if dst not in node_labels:
            node_labels[dst] = f"G{dst}"

    return edges, edge_labels, node_labels

def parse_graph_dict(graph_dict, node_label_mode="translation", edge_label_mode="action"): 

    edges = {}
    edge_labels = {}
    node_labels = {}

    for group in graph_dict.get("groups", []):
        group_id = int(group["group"])

        if node_label_mode == "translation":
            label = group.get("translation", f"G{group_id}")

        elif node_label_mode == "important_features":
            features = group.get("important_features", [])
            label = "|".join(features) if features else "none"

        elif node_label_mode == "critical":
            label = f"critical={int(group.get('critical_value', 0))}"

        elif node_label_mode == "group_id":
            label = f"G{group_id}"

        else:
            raise ValueError(f"Unknown node_label_mode: {node_label_mode}")

        node_labels[group_id] = label

    for edge in graph_dict.get("edges", []):
        src = int(edge["from_group"])
        dst = int(edge["to_group"])
        prob = float(edge.get("probability", 1.0))
        action = edge.get("action", "unknown")

        edges[(src, dst)] = prob

        if edge_label_mode == "action":
            edge_labels[(src, dst)] = str(action)

        elif edge_label_mode == "probability_bin":
            edge_labels[(src, dst)] = probability_to_bin(prob)

        elif edge_label_mode == "action_probability_bin":
            edge_labels[(src, dst)] = f"a{action}_p{probability_to_bin(prob)}"

        elif edge_label_mode is None:
            pass

        else:
            raise ValueError(f"Unknown edge_label_mode: {edge_label_mode}")

        # Handle edges to groups that may not appear in graph_dict["groups"]
        if src not in node_labels:
            node_labels[src] = f"G{src}"
        if dst not in node_labels:
            node_labels[dst] = f"G{dst}"

    if edge_label_mode is None:
        return Graph(edges, node_labels=node_labels)

    return Graph(edges, node_labels=node_labels, edge_labels=edge_labels)

def probability_to_bin(prob):
    if prob < 0.25:
        return "low"
    elif prob < 0.75:
        return "medium"
    else:
        return "high"

def print_kernel_table(kernel_matrix):
    n = kernel_matrix.shape[0]

    # Print column header
    header = "     " + "   ".join(f"{j:>4}" for j in range(n))
    print(header)

    # Print each row
    for i in range(n):
        row_vals = "   ".join(f"{kernel_matrix[i,j]:.2f}" for j in range(n))
        print(f"{i:>3}  {row_vals}")

def select_most_similar_pair(kernel_matrix, exclude_identical=True):
    """
    Select the most similar pair of distinct graphs.

    If exclude_identical=True, pairs with similarity approximately
    equal to 1.0 are ignored, since they provide no useful contrast.
    """

    k = np.asarray(kernel_matrix, dtype=float).copy()
 
    np.fill_diagonal(k, -np.inf)

    if exclude_identical: 
        k[np.isclose(k, 1.0)] = -np.inf

    if np.all(np.isneginf(k)):
        return None, None

    row, col = np.unravel_index(
        np.argmax(k),
        k.shape
    )

    return row, col

def assign_cluster_to_state(clusters, state, attr_names):
    for cluster in clusters:
        boundaries = cluster["boundaries"]
        match = True

        for feature in attr_names:
            if feature not in state:
                continue  # skip derived/non-state features like State Value or Action

            if feature not in boundaries:
                continue

            low, high = boundaries[feature]
            value = state[feature]

            low, high = min(low, high), max(low, high)

            if value is None or value < low or value > high:
                match = False
                break

        if match:
            return cluster["group"]

    return None
def get_next_probable_action(graph_dict, current_group):
    # graph_dict is the graph representation of the policy, with "groups" and "edges"
    # current_group is the group id of the current state
    # This function returns the action with the highest transition probability from the current group

    best_action = None
    best_prob = -1.0
    best_next_group = None

    for edge in graph_dict.get("edges", []):
        if int(edge["from_group"]) == current_group:
            prob = float(edge.get("probability", 0.0))
            action = edge.get("action", None)
            if prob > best_prob:
                best_prob = prob
                best_action = action
                best_next_group = int(edge["to_group"])

    return best_action, best_next_group