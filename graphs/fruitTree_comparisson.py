from grakel import Graph
from grakel.kernels import WeisfeilerLehman, VertexHistogram, SubgraphMatching
from utils import parse_transition_lines, print_kernel_table, FRUIT_TREE_GRAPHS
from sklearn.preprocessing import normalize

graphs = []
for idx, highwayGraph in enumerate(FRUIT_TREE_GRAPHS):
    edges, edge_labels, node_labels = parse_transition_lines(highwayGraph) 
    g = Graph(edges, node_labels=node_labels, edge_labels=edge_labels)
    graphs.append(g)

WLkernel = WeisfeilerLehman(base_graph_kernel=VertexHistogram,normalize=True)
SMkernel = SubgraphMatching(normalize=True)

K = WLkernel.fit_transform(graphs) 

print("Weisfeiler-Lehman Kernel similarity matrix:")
print(print_kernel_table(K))


K = SMkernel.fit_transform(graphs) 

print("Subgraph Kernel similarity matrix:")
print(print_kernel_table(K))

