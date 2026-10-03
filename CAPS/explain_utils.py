import matplotlib.pyplot as plt
import numpy as np
from CAPS.CLTree import CLTree
from CAPS.data import InstanceData
import torch


def cluster_data(translator, abstraction_helper, dataset, attr_names, alpha, num_actions, lmbda, k, max_height=20, model_path=None, env='grid'):
    num_feats__in_cluster = dataset.num_feats+2
    cluster_data = InstanceData(dataset.cluster_input, num_feats__in_cluster, attr_names)
    cltree = CLTree(cluster_data)     
    cltree.buildTree()

    #print('\n\n\n\n\n Tree built, starting pruning... \n')
    #print('Tree info: ', 
    #      'Nr of clusters: ', len(cltree.getClustersList(min_nr_instances=1)), '\n', 
    #      #'example of cluster: ', cltree.getClustersList(min_nr_instances=1)[0].getInstanceIds(),
    #      #'example of cluster boundaries: ', cltree.getClustersList(min_nr_instances=1)[0].getBoundaries(),
    #      '\n\n\n\n\n'
    #        )
                    

    height = max_height
    interactive_config = {'height': height}
    interactive = True

    cluster_scores = []
    lengths = []
    value_scores = []
    entropy_scores = []
    all_clusters = []

    test_alphas = np.arange(21)
    test_alphas = test_alphas / 100
    test_alpha_scores = []

    print('Starting graph generation...')
    for i in range(height):
        #print('Height: ', i+1)
        interactive_config = {'height': i+1}
        cltree.pruneTree(interactive, interactive_config)        
        clusters = cltree.getClustersList(min_nr_instances=1)
        all_clusters.append(clusters)

        #print('\n\n Tree info: ', 
        #  'Nr of clusters: ', len(clusters), '\n', 
        #    )
        if (env in ['MO_highway', 'traffic_junction']):
            for i in range(len(clusters)):
                attr_names = []
                for j in range(num_feats__in_cluster-2):
                    feat_name = f"Feature_{j}"
                    attr_names.append(feat_name)
                attr_names.append("State Value")
                attr_names.append("Action")
                #print(' cluster Nr: ', i, 'states: ', clusters[i].getInstanceIds(), ' \n boundaries: ')
                for j in range(num_feats__in_cluster):
                    feat_name = attr_names[j] if j < len(attr_names) else f"Feature_{j}"
                    #print('Feature {} boundaries: '.format(feat_name), clusters[i].get_bounds(j))
                #print('\n\n')
        else:
            for i in range(len(clusters)):
                #print(' cluster Nr: ', i, 'states: ', clusters[i].getInstanceIds(), ' \n boundaries: ')
                for j in range(num_feats__in_cluster):
                    feat_name = attr_names[j] if j < len(attr_names) else f"Feature_{j}"
                    #print('Feature {} boundaries: '.format(feat_name), clusters[i].get_bounds(j))
                #print('\n\n')
        #print('--------------------------------------------- \n\n\n\n')
        
        c = 0
        cluster_state_indices = []
        for i, node in enumerate(clusters):
            
            c += node.getNrInstancesInNode()
            
            cluster_state_indices.append(node.getInstanceIds())
        #print('Number of clusters: ', len(clusters))
        #print("Total instances clustered: ", c)
        #print("Percent instances clusered: ", c/dataset.num_entries)


        abstract_state_groups = []
        abstract_binary_state_groups = []
        cluster_values = []
        cluster_policies = []
        for cluster in cluster_state_indices:
            abs_t = []
            bin_t = []
            v = []
            a = np.zeros(num_actions)
            for idx in cluster:
                idx = int(idx)
                val = dataset.values[idx]
                v.append(val)
                a[dataset.actions[idx]] += 1
                abs_t.append((dataset.states[idx], dataset.actions[idx], dataset.next_states[idx], dataset.dones[idx], dataset.entropies[idx], dataset.rewards[idx]))
                binary = translator.state_to_binary(dataset.states[idx])
                bin_t.append((binary, dataset.actions[idx]))
            abstract_state_groups.append(abs_t)
            abstract_binary_state_groups.append(bin_t)
            cluster_values.append(sum(v)/len(v))
            a = a / np.sum(a)
            cluster_policies.append(a)
        
        cluster_values = np.array(cluster_values)
        pred_cluster_values = []
        abs_t = abstract_state_groups
        bin_t = abstract_binary_state_groups
        l, transitions, taken_actions = abstraction_helper.compute_graph_info(abs_t)
        cl_entropies = []


        
        for j, cluster in enumerate(clusters):
            transition_probs = np.array(transitions[j][:-1])
            pred_cluster_values.append(lmbda * sum(transition_probs * cluster_values))
            cl_pol = np.array(cluster_policies[j])
            cl_pol_nonzero = np.where(cl_pol != 0)[0]
            cl_pol_nonzero = cl_pol[cl_pol_nonzero]
            entr = -sum(cl_pol_nonzero * np.log2(cl_pol_nonzero))
            cl_entropies.append(entr)

        #Should value score be weighted according to number of states in a cluster?
        val_score = np.linalg.norm(cluster_values - pred_cluster_values) / np.linalg.norm(cluster_values)
        val_score = np.square(cluster_values - pred_cluster_values).mean()
        entropy_score = sum(cl_entropies) / len(cl_entropies)

        #print('Val score: ', val_score)
        #print('Entropy score: ', entropy_score)

        score = val_score + entropy_score

        a_score = score + alpha * len(clusters)

        """
        test_a_scores = []
        for test_a in test_alphas:
            t_a = score + test_a * len(clusters)
            test_a_scores.append(t_a)
        test_alpha_scores.append(test_a_scores)
        """
        
        #print('Cluster score (lower is better): ', a_score)

        value_scores.append(val_score)
        entropy_scores.append(entropy_score)
        cluster_scores.append(a_score)
        lengths.append(len(clusters))

    cluster_scores = np.array(cluster_scores) #shape num_graphs, num_alphas
    lengths = np.array(lengths)

    """
    best_scores_by_alpha = np.argmin(test_alpha_scores, axis=0)
    best_graph_size_by_alpha = lengths[best_scores_by_alpha]
    plot_data = np.array([best_graph_size_by_alpha, test_alphas])
    np.save('temp_plot_data/{}_num_nodes_vs_alpha.npy'.format(env), plot_data)
    """


    best_graph_idx = np.argsort(cluster_scores)
    best_graphs = best_graph_idx[:k]
    best_graphs = np.squeeze(best_graphs)
    best_graphs = np.array(best_graphs, dtype=np.int32)
    if best_graphs[0] == 0: #Don't want to include the low graph since its value score is always low
        best_graphs = best_graph_idx[1:k+1]
    
    return all_clusters, best_graphs, cluster_scores, value_scores, entropy_scores, lengths, cltree

def graph_scores(env, alpha, lengths, cluster_scores=None, value_scores=None, entropy_scores=None, fidelity_scores=None):
    plt.style.use('ggplot')
    if cluster_scores is not None:
        """
        plt.plot(lengths, cluster_scores, color='orange')
        plt.xlabel('Number of Graph Nodes')
        plt.ylabel('Heuristic Score')
        plt.title('Score vs Graph Size (alpha={0:.3f})'.format(alpha))
        plt.savefig('results/heuristic_graphs/{}_score_vs_size.png'.format(env))
        plt.clf()
        """
        plot_data = np.array([cluster_scores, lengths])
        np.save('temp_plot_data/{}_score_vs_size.npy'.format(env), plot_data)

        log_scores = np.log(cluster_scores)
        plot_data = np.array([log_scores, lengths])
        np.save('temp_plot_data/{}_log_score_vs_size.npy'.format(env), plot_data)

    if value_scores is not None:
        """
        plt.plot(lengths, value_scores, color='orange')
        plt.xlabel('Number of Graph Nodes')
        plt.ylabel('Value Score')
        plt.title('Value Score vs Graph Size')
        plt.savefig('results/heuristic_graphs/{}_val_score_vs_size.png'.format(env))
        plt.clf()
        """

        plot_data = np.array([value_scores, lengths])
        np.save('temp_plot_data/{}_value_vs_size.npy'.format(env), plot_data)

    if entropy_scores is not None:
        """
        plt.plot(lengths, entropy_scores, color='orange')
        plt.xlabel('Number of Graph Nodes')
        plt.ylabel('Graph Policy Entropy')
        plt.title('Entropy Score vs Graph Size')
        plt.savefig('results/heuristic_graphs/{}_entropy_score_vs_size.png'.format(env))
        plt.clf()
        """

        plot_data = np.array([entropy_scores, lengths])
        np.save('temp_plot_data/{}_entropy_vs_size.npy'.format(env), plot_data)

    if fidelity_scores is not None:
        
        plt.plot(lengths, fidelity_scores, color='orange')
        plt.xlabel('Number of Graph Nodes')
        plt.ylabel('Graph Policy Fidelity')
        plt.title('Fidelity vs Graph Size')
        plt.savefig('results/heuristic_graphs/{}_fidelity_vs_size.png'.format(env))
        plt.clf()

def cluster_data_with_boundaries(
    translator,
    abstraction_helper,
    dataset,
    attr_names,
    alpha,
    num_actions,
    lmbda,
    k,
    max_height=20,
    model_path=None,
    env='grid',
    cluster_features = False
):
    print(f"generating graph for model: {model_path}", flush=True)
    num_feats__in_cluster = dataset.num_feats + 2
    cluster_data_obj = InstanceData(dataset.cluster_input, num_feats__in_cluster, attr_names)
    cltree = CLTree(cluster_data_obj)
    cltree.buildTree()

    """print('\n\n Tree built, starting pruning... \n')
    print(
        'Tree info: ',
        'Nr of clusters: ', len(cltree.getClustersList(min_nr_instances=1)), '\n',
        '\n\n'
    )"""

    height = max_height
    interactive_config = {'height': height}
    interactive = True

    cluster_scores = []
    lengths = []
    value_scores = []
    entropy_scores = []
    all_clusters = []

    # Build feature names once
    if env in ['traffic_junction']:
        feature_names = [f"Feature_{j}" for j in range(num_feats__in_cluster - 2)]
        feature_names.append("State Value")
        feature_names.append("Action")
    else:
        feature_names = []
        for j in range(num_feats__in_cluster):
            if j < len(attr_names):
                feature_names.append(attr_names[j])
            else:
                feature_names.append(f"Feature_{j}")

    test_alphas = np.arange(21)
    test_alphas = test_alphas / 100
    test_alpha_scores = []

    print('Starting graph generation...')
    for h in range(height):
        #print('Height: ', h + 1)
        interactive_config = {'height': h + 1}
        cltree.pruneTree(interactive, interactive_config)
        clusters = cltree.getClustersList(min_nr_instances=1)
        all_clusters.append(clusters)

        """print('\n\n Tree info: ',
              'Nr of clusters: ', len(clusters), '\n')
        for c_idx in range(len(clusters)):
            print(' cluster Nr: ', c_idx, 'states: ', clusters[c_idx].getInstanceIds(), ' \n boundaries: ')
            for j in range(num_feats__in_cluster):
                print('Feature {} boundaries: '.format(feature_names[j]), clusters[c_idx].get_bounds(j))
            print('\n\n')
        print('--------------------------------------------- \n\n')"""

        c = 0
        cluster_state_indices = []
        for i, node in enumerate(clusters):
            c += node.getNrInstancesInNode()
            cluster_state_indices.append(node.getInstanceIds())

        abstract_state_groups = []
        abstract_binary_state_groups = []
        cluster_values = []
        cluster_policies = []

        for cluster in cluster_state_indices:
            abs_t = []
            #bin_t = []
            v = []
            a = np.zeros(num_actions)

            for idx in cluster:
                idx = int(idx)
                val = dataset.values[idx]
                v.append(val)

                action = int(dataset.actions[idx])

                if action < 0 or action >= num_actions:
                    raise ValueError(
                        f"Action {action} out of bounds for num_actions={num_actions}. "
                        f"Unique dataset actions: {np.unique(dataset.actions)}. idx={idx}"
                    )

                a[action] += 1
                abs_t.append((
                    dataset.states[idx],
                    action,
                    dataset.next_states[idx],
                    dataset.dones[idx],
                    dataset.entropies[idx],
                    dataset.rewards[idx]
                ))
                #binary = translator.state_to_binary(dataset.states[idx])
                #bin_t.append((binary, dataset.actions[idx]))

            abstract_state_groups.append(abs_t)
            #abstract_binary_state_groups.append(bin_t)
            cluster_values.append(sum(v) / len(v))
            a = a / np.sum(a)
            cluster_policies.append(a)

        cluster_values = np.array(cluster_values)
        pred_cluster_values = []
        abs_t = abstract_state_groups
        #bin_t = abstract_binary_state_groups
        #print("translator:", translator)
        #print("abstract_baseline:", abstraction_helper)
        #print("type(abstract_baseline):", type(abstraction_helper))
        l, transitions, taken_actions = abstraction_helper.compute_graph_info(abs_t)
        cl_entropies = []

        for j, cluster in enumerate(clusters):
            transition_probs = np.array(transitions[j][:-1])
            pred_cluster_values.append(lmbda * sum(transition_probs * cluster_values))

            cl_pol = np.array(cluster_policies[j])
            cl_pol_nonzero = np.where(cl_pol != 0)[0]
            cl_pol_nonzero = cl_pol[cl_pol_nonzero]
            entr = -sum(cl_pol_nonzero * np.log2(cl_pol_nonzero))
            cl_entropies.append(entr)

        val_score = np.square(cluster_values - pred_cluster_values).mean()
        entropy_score = sum(cl_entropies) / len(cl_entropies)
        score = val_score + entropy_score
        a_score = score + alpha * len(clusters)

        value_scores.append(val_score)
        entropy_scores.append(entropy_score)
        cluster_scores.append(a_score)
        lengths.append(len(clusters))

    cluster_scores = np.array(cluster_scores)
    lengths = np.array(lengths)

    best_graph_idx = np.argsort(cluster_scores)
    best_graphs = best_graph_idx[:k]
    best_graphs = np.squeeze(best_graphs)
    best_graphs = np.array(best_graphs, dtype=np.int32)

    if best_graphs[0] == 0:  # Don't want to include the low graph since its value score is always low
        best_graphs = best_graph_idx[1:k+1]

    # Collect boundaries only for the selected graphs/clusters
    """selected_cluster_boundaries = []
    for graph_idx in best_graphs:
        graph_clusters = all_clusters[graph_idx]
        graph_boundary_info = []

        for cluster in graph_clusters:
            cluster_boundary_dict = {
                feature_names[j]: cluster.get_bounds(j)
                for j in range(num_feats__in_cluster)
            }
            graph_boundary_info.append(cluster_boundary_dict)

        selected_cluster_boundaries.append(graph_boundary_info)
        
    return all_clusters, best_graphs, cluster_scores, value_scores, entropy_scores, lengths, selected_cluster_boundaries, feature_names"""
    
    ###################################################
    ###################################################

    selected_cluster_boundaries = []
    selected_cluster_features = [] if cluster_features else None

    for graph_idx in best_graphs:
        graph_clusters = all_clusters[graph_idx]
        graph_boundary_info = []
        graph_feature_info = [] if cluster_features else None
        #print(f"\n\nSelected graph index: {graph_idx}, number of clusters: {len(graph_clusters)}\n")
        for cluster in graph_clusters:
            cluster_boundary_dict = {
                feature_names[j]: cluster.get_bounds(j)
                for j in range(num_feats__in_cluster)
            }
            graph_boundary_info.append(cluster_boundary_dict)

            if cluster_features:
                #print(f"Getting cluster features for cluster with boundaries: {cluster_boundary_dict}")
                cluster_feature_list = cltree.get_cluster_features(cluster, attr_names=feature_names)
                graph_feature_info.append(cluster_feature_list)

        selected_cluster_boundaries.append(graph_boundary_info)

        if cluster_features:
            selected_cluster_features.append(graph_feature_info)

    
    if cluster_features:
        return all_clusters, best_graphs, cluster_scores, value_scores, entropy_scores, lengths, selected_cluster_boundaries, feature_names, selected_cluster_features, cltree
    else: 
        return all_clusters, best_graphs, cluster_scores, value_scores, entropy_scores, lengths, selected_cluster_boundaries, feature_names, cltree
    
    ###################################################
    ###################################################