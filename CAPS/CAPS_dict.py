import torch
import numpy as np
import matplotlib.pyplot as plt
from CAPS.explain_utils import graph_scores
from CAPS.explain_utils import cluster_data, cluster_data_with_boundaries
import shap


"""
Env specific info:
run episode function
predicate class
number of actions
number of features
value function
env name
alpha
max height
lambda
feature groups (include in predicate class)
"""
def explain(args, dataset, model_path, translator, num_feats, num_actions, fidelity_fn=None, apg_baseline=None, mode="PPO"):


    attr_names = translator.attr_names
    attr_names.append('State Value')
    attr_names.append('Action')
    
    num_runs = 1
    fidelities = []
    cluster_v_scores = []
    ls = []
    e_scores = []
    for run in range(num_runs):
        all_clusters, best_heights, cluster_scores, value_scores, entropy_scores, lengths, cltree = cluster_data(translator, 
                                                                                                        apg_baseline, 
                                                                                                        dataset,
                                                                                                        attr_names,
                                                                                                        args.alpha,
                                                                                                        num_actions=num_actions,
                                                                                                        lmbda=args.lmbda,
                                                                                                        k=args.k,
                                                                                                        max_height=args.max_height,
                                                                                                        model_path=model_path,
                                                                                                        env=args.env
                                                                                                    )

        """
        graph_scores('cart', alpha, lengths, 
                    cluster_scores=cluster_scores, 
                    value_scores=value_scores, 
                    entropy_scores=entropy_scores,
                    fidelity_scores=fidelity_scores)
        """

    all_clusters = np.array(all_clusters, dtype=object)
    # I'm changing the selection to only the last best cluster configuration for easier analysis and comparisson between graphs, 
    # but we could easily modify to produce explanations for all cluster configurations at different heights
    selected_height = best_heights[-1]
    clusters = all_clusters[selected_height]

    graph_info = {
        "tree": cltree,
        "height": selected_height + 1,
        "fidelity": None,
        "state_features": attr_names,
        "groups": [],
        "edges": []
    }

    print('***********************************************')
    print('Clusters at height {}'.format(selected_height + 1))

    cluster_state_indices = []
    for node in clusters:
        cluster_state_indices.append(node.getInstanceIds())

    if fidelity_fn is not None:
        assert model_path is not None
        fidelity = fidelity_fn(model_path, clusters, dataset, mode=mode)
        print('Fidelity: ', fidelity)

        fidelities.append(fidelity)
        fidelity_scores = [fidelity]

        graph_info["fidelity"] = fidelity

    cluster_v_scores.append(value_scores[selected_height])

    abstract_state_groups = []
    abstract_binary_state_groups = []

    for cluster in cluster_state_indices:
        abs_t = []
        bin_t = []

        for idx in cluster:
            idx = int(idx)

            abs_t.append((
                dataset.states[idx],
                dataset.actions[idx],
                dataset.next_states[idx],
                dataset.dones[idx],
                dataset.entropies[idx],
                dataset.rewards[idx]
            ))

            binary = translator.state_to_binary(dataset.states[idx])
            bin_t.append((binary, dataset.actions[idx]))

        abstract_state_groups.append(abs_t)
        abstract_binary_state_groups.append(bin_t)

    abs_t = abstract_state_groups
    bin_t = abstract_binary_state_groups

    critical_values, group_ent = apg_baseline.get_critical_groups(abs_t)

    l, transitions, taken_actions = apg_baseline.compute_graph_info(abs_t)

    for j in range(len(transitions)):
        nonzero_idx = np.where(np.array(transitions[j]) != 0)[0]

        for idx in nonzero_idx:
            action = int(np.mean(taken_actions[j][idx]))
            probability = transitions[j][idx]

            #print(
            #    'Group {} to Group {} with p={} and action {}'.format(
            #        j + 1,
            #        idx + 1,
            #        probability,
            #        action
            #    )
            #)

            graph_info["edges"].append({
                "from_group": j + 1,
                "to_group": idx + 1,
                "probability": float(probability),
                "action": action
            })

    if args.hayes_baseline:
        hayes_translations = translator.reduce_logic(bin_t)

    #print('----------------------------------------')

    translations = translator.my_translation_algo(bin_t)

    for j, t in enumerate(translations):
        #print('Group {}: {}'.format(j + 1, t))

        group_data = {
            "group": j + 1,
            "translation": t,
            "critical_value": critical_values[j],
            "entropy": float(group_ent[j]),
            "num_instances": len(cluster_state_indices[j])
        }

        if args.hayes_baseline:
            #print('(Hayes) Group {}: {}'.format(j + 1, hayes_translations[j]))
            group_data["hayes_translation"] = hayes_translations[j]

        #print(
        #    'Critical value: {}. Entropy: {:.2f}'.format(
        #        critical_values[j],
        #        group_ent[j]
        #    )
        #)
        
        #print(f"{group_data}")
        graph_info["groups"].append(group_data)

    #print('----------------------------------------')

def explain_auto_pred(args, dataset, model_path, translator, num_actions, attr_names, fidelity_fn=None, apg_baseline=None, mode="PPO", num_feats=0, shap_feature_selection=True,use_all_features=False):

    print(f"apg_baseline inside explain_auto_pred: {apg_baseline}", flush=True)
    #print("type(apg_baseline):", type(apg_baseline))
    ft_names = translator.feature_names
    attr_names = ft_names
    attr_names.append('State Value')
    attr_names.append('Action')
    
    num_runs = 1
    fidelities = []
    cluster_v_scores = []
    ls = []
    e_scores = []
    for run in range(num_runs):
        if shap_feature_selection:
            all_clusters, best_heights, cluster_scores, value_scores, entropy_scores, lengths, selected_cluster_boundaries, feature_names, cltree = cluster_data_with_boundaries(translator, 
                                                                                                        apg_baseline, 
                                                                                                        dataset,
                                                                                                        attr_names,
                                                                                                        args.alpha,
                                                                                                        num_actions=num_actions,
                                                                                                        lmbda=args.lmbda,
                                                                                                        k=args.k,
                                                                                                        max_height=args.max_height,
                                                                                                        model_path=model_path,
                                                                                                        env=args.env 
                                                                                                    )
        else:
            all_clusters, best_heights, cluster_scores, value_scores, entropy_scores, lengths, selected_cluster_boundaries, feature_names, selected_cluster_features, cltree = cluster_data_with_boundaries(translator, 
                                                                                                        apg_baseline, 
                                                                                                        dataset,
                                                                                                        attr_names,
                                                                                                        args.alpha,
                                                                                                        num_actions=num_actions,
                                                                                                        lmbda=args.lmbda,
                                                                                                        k=args.k,
                                                                                                        max_height=args.max_height,
                                                                                                        model_path=model_path,
                                                                                                        env=args.env,
                                                                                                        cluster_features=True
                                                                                                    ) 

        all_clusters = np.array(all_clusters, dtype=object)

        # Select only the last best cluster configuration
        selected_h_idx = len(best_heights) - 1
        selected_height = best_heights[selected_h_idx]
        clusters = all_clusters[selected_height]

        fidelity_scores = []
        cluster_v_scores.append(value_scores[selected_height])

        graph_info = {
            #"tree": cltree,
            "height": selected_height + 1,
            "fidelity": None,
            "state_features": attr_names,
            "groups": [],
            "edges": [],
            "feature_selection": {
                "method": "shap" if shap_feature_selection else "tree_path",
                "important_features": []
            }
        }

        #print('***********************************************')
        #print('Clusters at height {}'.format(selected_height + 1))

        cluster_state_indices = []

        for node in clusters:
            cluster_state_indices.append(node.getInstanceIds())

        if fidelity_fn is not None:
            assert model_path is not None
            fidelity = fidelity_fn(model_path, clusters, dataset, mode=mode)

            print('Fidelity: ', fidelity)

            fidelities.append(fidelity)
            fidelity_scores.append(fidelity)

            graph_info["fidelity"] = fidelity

        abstract_state_groups = []

        for cluster in cluster_state_indices:
            abs_t = []

            for idx in cluster:
                    idx = int(idx)

                    abs_t.append((
                        dataset.states[idx],
                        dataset.actions[idx],
                        dataset.next_states[idx],
                        dataset.dones[idx],
                        dataset.entropies[idx],
                        dataset.rewards[idx]
                    ))

            abstract_state_groups.append(abs_t)

        abs_t = abstract_state_groups

        critical_values, group_ent = apg_baseline.get_critical_groups(abs_t)

        l, transitions, taken_actions = apg_baseline.compute_graph_info(abs_t)

        for j in range(len(transitions)):
                nonzero_idx = np.where(np.array(transitions[j]) != 0)[0]

                for idx in nonzero_idx:
                    probability = transitions[j][idx]
                    action = int(np.mean(taken_actions[j][idx]))

                    #print(
                    #    'Group {} to Group {} with p={} and action {}'.format(
                    #        j + 1,
                    #        idx + 1,
                    #        probability,
                    #        action
                    #    )
                    #)

                    graph_info["edges"].append({
                        "from_group": j + 1,
                        "to_group": idx + 1,
                        "probability": float(probability),
                        "action": action
                    })

        #print('----------------------------------------')

        cluster_boundaries = selected_cluster_boundaries[selected_h_idx]

        ###################################################
        ###################################################

        if use_all_features:
            print("\nUsing all features for predicate generation...")

            # Every cluster gets all available features
            important_features = [
                list(translator.feature_names)
                for _ in cluster_state_indices
            ]

        elif shap_feature_selection:
                #number_of_features_per_cluster = 4 # top x features by shap value to use for predicate generation (per cluster)
                percentage_of_states_for_shap_training = 10 # sample x% of total dataset for SHAP training
                percentage_of_states_per_cluster_for_shap = 20  # sample x% of each cluster for SHAP explanation
                important_features = []
                number_of_states_of_training_percentage = int(len(dataset.states) * percentage_of_states_for_shap_training / 100)
                max_training_states = 100
                number_of_states_to_use_for_training = min(
                    max_training_states,
                    number_of_states_of_training_percentage
                )
                sample_train_states_array = np.array(
                    dataset.states[:number_of_states_to_use_for_training],
                    dtype=np.float32
                )
                number_of_states_of_cluster_percentage = int(len(dataset.states) * percentage_of_states_per_cluster_for_shap / 100)
                max_states_per_cluster = 100
                number_of_states_to_use_for_shap = min(
                    max_states_per_cluster,
                    number_of_states_of_cluster_percentage
                )

                shap_value_fn = make_shap_value_fn(apg_baseline.value_fn)

                #print("background:", sample_train_states_array.shape)
                #print("wrapper output:", shap_value_fn(sample_train_states_array).shape)

                shap_explainer = shap.KernelExplainer(
                    shap_value_fn,
                    sample_train_states_array
                )

                print("\n Computing SHAP values for clusters...")

                for cluster_id, cluster in enumerate(cluster_state_indices):
                    cluster = np.array(cluster, dtype=int)

                    if len(cluster) == 0:
                        important_features.append([])
                        continue

                    # sample some states from this cluster
                    n_cluster_states = max(
                        1,
                        number_of_states_to_use_for_shap
                    )
                    n_cluster_states = min(n_cluster_states, len(cluster))

                    sampled_cluster_indices = np.random.choice(
                        cluster,
                        size=n_cluster_states,
                        replace=False
                    )

                    cluster_states_array = np.array(
                        [dataset.states[idx] for idx in sampled_cluster_indices],
                        dtype=np.float32
                    )

                    shap_values = shap_explainer.shap_values(cluster_states_array, nsamples=n_cluster_states)
                    if shap_values.ndim == 3 and shap_values.shape[2] == 1:
                        shap_values = shap_values[:, :, 0]

                    # For a scalar output value_fn, shap_values is usually shape:
                    #   (n_samples, n_features)
                    # but depending on SHAP version it can sometimes be returned in a list/extra dimension
                    if isinstance(shap_values, list):
                        shap_values = np.array(shap_values)

                        # common case: list of length 1 for single-output model
                        if shap_values.ndim == 3 and shap_values.shape[0] == 1:
                            shap_values = shap_values[0]

                    shap_values = np.array(shap_values)

                    # ensure final shape is [n_samples, n_features]
                    if shap_values.ndim == 1:
                        shap_values = shap_values.reshape(1, -1)
                    elif shap_values.ndim > 2:
                        raise ValueError(f"Unexpected SHAP shape for cluster {cluster_id+1}: {shap_values.shape}")

                    #print(f"Cluster {cluster_id+1}: shap_values shape: {shap_values.shape}")

                    # aggregate feature importance across the sampled states in this cluster
                    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

                    # normalize to [0,1]
                    total_importance = np.sum(mean_abs_shap) + 1e-8
                    normalized_importance = mean_abs_shap / total_importance

                    # sort
                    sorted_indices = np.argsort(normalized_importance)[::-1]
                    sorted_importances = normalized_importance[sorted_indices]

                    #print(f"indices: {sorted_indices}")
                    #print(f"normalized importances: {sorted_importances}")

                    # relative drop
                    relative_diffs = (sorted_importances[:-1] - sorted_importances[1:]) / (sorted_importances[:-1] + 1e-8)
                    cutoff_idx = np.argmax(relative_diffs) + 1

                    # threshold
                    threshold = 0.05

                    threshold_indices = [
                        idx for idx in sorted_indices
                        if normalized_importance[idx] >= threshold
                    ]

                    cutoff_indices = sorted_indices[:cutoff_idx]

                    # combine
                    selected_indices = list(set(threshold_indices) & set(cutoff_indices))

                    if len(selected_indices) == 0:
                        selected_indices = cutoff_indices[:1]

                    top_feature_names = [feature_names[i] for i in selected_indices]

                    important_features.append(top_feature_names)

                    #print(f"Cluster {cluster_id+1}: top features = {top_feature_names}")
        else:
                # if not using SHAP-based feature selection, use tree path strategy for
                # predicate generation using features in cluster decision treepath
                print("\nUsing tree path feature selection for predicate generation...")
                #print(f"total states in dataset: {len(dataset.states)}")
                important_features = selected_cluster_features[h]
            
        graph_info["feature_selection"]["important_features"] = important_features


        ###################################################
        ###################################################

        # Predicate Generation  
        translations = translator.my_translation_algo(
                cluster_boundaries,
                feature_names_to_use=important_features
            )

        for j, t in enumerate(translations):
                #print('Group {}: {}'.format(j + 1, t))
                #print(
                #    'Critical value: {}. Entropy: {:.2f}'.format(
                #        critical_values[j],
                #        group_ent[j]
                #    )
                #)

                graph_info["groups"].append({
                    "group": j + 1,
                    "translation": t,
                    "critical_value": critical_values[j],
                    "entropy": float(group_ent[j]),
                    "num_instances": len(cluster_state_indices[j]),
                    "important_features": important_features[j],
                    "boundaries": cluster_boundaries[j]
                })

        #print('----------------------------------------')
    return graph_info

def make_shap_value_fn(value_fn):
    def shap_value_fn(x):
        x = np.asarray(x, dtype=np.float32)

        # If SHAP sends one flat observation, make it a batch of one.
        if x.ndim == 1:
            x = x[None, :]

        values = []
        for obs in x:
            v = value_fn(obs)
            values.append(float(np.asarray(v).squeeze()))

        return np.asarray(values, dtype=np.float32)

    return shap_value_fn