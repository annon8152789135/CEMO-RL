import matplotlib.pyplot as plt
import networkx as nx
import os

def build_nx_graph(edges, edge_labels, node_labels):
    """
    Build a NetworkX DiGraph from:
      edges: dict[(src, dst) -> prob]
      edge_labels: dict[(src, dst) -> action (str)]
      node_labels: dict[node_id -> label (str)]
    """
    G = nx.DiGraph()

    for node_id, label in node_labels.items():
        G.add_node(node_id, label=label)

    for (src, dst), prob in edges.items():
        action = edge_labels.get((src, dst), "")
        G.add_edge(src, dst, prob=prob, action=action)

    return G


def plot_graph(G, node_labels, edges, edge_labels, save_path=None, title="Policy Graph"):
    """
    Plot the graph with node labels and edge labels "aX, p=Y".
    """

    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)

    nx.draw(
        G,
        pos,
        with_labels=True,
        labels=node_labels,
        node_size=800,
        font_size=8,
        arrows=True,
    )

    edge_label_dict = {}
    for (src, dst), prob in edges.items():
        action = edge_labels.get((src, dst), "")
        edge_label_dict[(src, dst)] = f"a{action}, p={prob:.2f}"

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_label_dict, font_size=6)

    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.savefig(f"./plots/{title.replace(' ', '_')}.png", dpi=300)



def build_nx_graph_from_apg(graph):
    """
    Build a NetworkX DiGraph directly from one APG dictionary.

    Expected structure:
        graph["groups"] -> list of group dictionaries
        graph["edges"]  -> list of transition dictionaries
    """

    G = nx.DiGraph()
 
    for group in graph["groups"]:
        group_id = group["group"]
        translation = group["translation"]

        G.add_node(
            group_id,
            label=translation,
            critical_value=group.get("critical_value"),
            entropy=group.get("entropy"),
            num_instances=group.get("num_instances"),
        )
 
    for edge in graph["edges"]:
        src = edge["from_group"]
        dst = edge["to_group"]
 
        if dst not in G:
            G.add_node(dst, label="Terminal")

        if src not in G:
            G.add_node(src, label="Terminal")

        G.add_edge(
            src,
            dst,
            probability=edge["probability"],
            action=edge["action"],
        )

    return G


def plot_apg(graph, save_path=None, title="Policy Graph"):
    """
    Plot one APG dictionary.
    """

    G = build_nx_graph_from_apg(graph)

    plt.figure(figsize=(12, 8))

    pos = nx.spring_layout(
        G,
        seed=42,
        k=1.2
    )
 
    node_labels = {
        node: f"{node}\n{data['label']}"
        for node, data in G.nodes(data=True)
    }

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=1800
    )

    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowsize=20
    )

    nx.draw_networkx_labels(
        G,
        pos,
        labels=node_labels,
        font_size=7
    )
 
    edge_labels = {
        (src, dst):
            f"a{data['action']}, p={data['probability']:.2f}"
        for src, dst, data in G.edges(data=True)
    }

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=6
    )

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    os.makedirs(save_path, exist_ok=True)

    filename = title.replace(" ", "_").replace("/", "_")

    plt.savefig(
        f"{save_path}/{filename}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()
