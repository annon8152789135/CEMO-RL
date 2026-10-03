import time
import os
import pickle
import csv
import glob
from config import argparser
from CAPS.CAPS_main import CAPS_main
from plot_policies import plot_policy_returns
from graphs.plot_graph import plot_apg
from graphs.compare_kernels import compare_explanation_graphs
from graphs.utils import print_kernel_table, select_most_similar_pair, assign_cluster_to_state, get_next_probable_action
from graphs.subgraph_search import get_maximum_common_subgraph, compare_transition_sets, get_transition_differences_at_common_nodes
from graphs.subgraph_search import build_common_percentage_matrices,percentage_table_g1_in_g2, print_percentage_table

from sample_states import test_states_ft, test_states_hw, test_states_dst
from model_paths import paths_ft, paths_hw, paths_dst, paths_hw2
from model_paths import dpmorl_output_dir_ft, dpmorl_output_dir_hw, dpmorl_output_dir_dst, dpmorl_output_dir_hw2
import time

def run_policies(paths, args, experiment_name=None):
    for pol_idx, model_path in enumerate(paths):
        args.path = model_path 
        print(f"\nRunning policy {pol_idx}: {model_path}")
        explanation = CAPS_main(args)
        print(f"Policy {pol_idx} explanation graph generated. {len(explanation['groups'])} nodes and {len(explanation['edges'])} edges.")
        print (f"sample label for policy {pol_idx}: {explanation['groups'][0]}  ")
        all_graphs[f"policy_{pol_idx}"] = explanation
        pkl_file_path = (
            f"outputs/graphs/{args.experiment_name}/"
            f"{args.env}_{args.num_episodes}_{args.lmbda}_{args.compare_criterion}/"
            f"policy_{pol_idx}_graph.pkl"
        )
        with open(pkl_file_path, 'wb') as f:
            pickle.dump(explanation, f)
    return all_graphs

def read_graphs_from_files(paths):
    for pol_idx, model_path in enumerate(paths):
        pkl_file_path = (
            f"outputs/graphs/{args.experiment_name}/"
            f"{args.env}_{args.num_episodes}_{args.lmbda}_{args.compare_criterion}/"
            f"policy_{pol_idx}_graph.pkl"
        )
        if os.path.exists(pkl_file_path):
            with open(pkl_file_path, 'rb') as f:
                explanation = pickle.load(f)
                all_graphs[f"policy_{pol_idx}"] = explanation
                print(f"Loaded graph for policy {pol_idx} from {pkl_file_path}.")
        else:
            print(f"Graph file for policy {pol_idx} not found at {pkl_file_path}. Please run the policies to generate the graphs.")
    return all_graphs
    

if __name__ == '__main__':

    """"
    This is the main script running the explanation generation algorithm for MO-Policies
    It works by first collecting the initial parameters and then running the policy graph
    computation algorithm to get the graph for each policy, and then collecting all the 
    graphs in a dictionary for later use.
    Then We compute graph comparisson metrics for all generated graphs, and select the best
    for comparisson to generate contrastive explanations. 
    Finally, we generate the contrastive explanations for the best graph and print them out/Log them.
    """

    # Step 1: Collect initial parameters 
    args = argparser()

    test_name= f"{args.env}_{args.num_episodes}_{args.lmbda}_{args.compare_criterion}"

    """
    if args.env == "MO_fruitTree":
        paths = paths_ft
        dpmorl_output_dir = dpmorl_output_dir_ft
    elif args.env == "MO_highway":
        paths = paths_hw2
        dpmorl_output_dir = dpmorl_output_dir_hw2
    elif args.env == "MO_deepSea":
        paths = paths_dst
        dpmorl_output_dir = dpmorl_output_dir_dst
    else:
        raise ValueError(f"Unknown environment: {args.env}")
        """

    if args.env not in ["MO_fruitTree", "MO_highway", "MO_deepSea"]:
        raise ValueError(f"Unknown environment: {args.env}. Please choose from 'MO_fruitTree', 'MO_highway', or 'MO_deepSea'.") 

    env_name = args.env.replace("MO_", "")  # Remove the "MO_" prefix to get the environment name
    if env_name == 'deepSea':
        env_name = 'DeepSeaTreasure'  # Adjust for the specific case of DeepSeaTreasure

    dpmorl_output_dir = f'CAPS/DPMORL/experiments/{args.experiment_name}/DPMORL.{env_name}.LossNormLamda_{args.lmbda}/'

    paths = glob.glob(os.path.join(dpmorl_output_dir, "policy-program-*.zip"))

    paths = [
        path[:-4] if path.endswith(".zip") else path
        for path in paths
    ]
    print(f"Found {len(paths)} policies in {dpmorl_output_dir}.")
    num_pols = len(paths)

    if args.plot_returns == True:
        plots_dir = f"outputs/plots/{args.experiment_name}/{test_name}"
        os.makedirs(plots_dir, exist_ok=True)
        plot_policy_returns(
            env_name=args.env,
            dpmorl_output_dir=dpmorl_output_dir,
            save_dir=plots_dir,
            batch_size=args.batch_size,
            final_episodes=args.final_episodes,
            reward_dims=args.reward_dims
        )

    # Step 2: Run the policy graph computation algorithm for each policy and collect the graphs

    all_graphs = {}

    time_graph_phase = time.time()
    
    if os.path.exists(f"outputs/graphs/{args.experiment_name}/{test_name}"):
        print(f"Directory {test_name} already exists.") 
        if args.use_existing:
            print(f"Using existing graphs from {test_name} as --use_existing flag is set to True. Checking for existing graphs...")
            if len(os.listdir(f"outputs/graphs/{args.experiment_name}/{test_name}")) < len(paths):
                raise ValueError(f"The directory {test_name} is empty or contains "+
                      f"fewer files than expected but the --use_existing flag is  set to True. "+
                      f"Please check if the directory contains the expected graph files.")
            else:
                print(f"""length of files in {test_name}: {len(os.listdir(f"outputs/graphs/{args.experiment_name}/{test_name}"))}, 
                expected: {len(paths)}.
                Re-running the policies to collect new graphs""")
                all_graphs = read_graphs_from_files(paths)
                print(f"Loaded existing graphs from {f'{test_name}'} . {len(all_graphs)} graphs in total.")
        else:
            print(f"Re-running the policies to collect new graphs as --use_existing flag is set to False.")
            all_graphs = run_policies(paths, args)
            print(f"All policies have been tested and graphs collected. {len(all_graphs)} graphs in total.")
    else:
        print(f"Directory {test_name} does not exist. Creating it and running the policies to collect new graphs.")
        os.makedirs(f"outputs/graphs/{args.experiment_name}/{test_name}", exist_ok=True)
        all_graphs = run_policies(paths, args)
        print(f"All policies have been tested and graphs collected. {len(all_graphs)} graphs in total.")

    print(f"All graphs collected:")
    for policy_name, graph in all_graphs.items():
        plot_apg(
            graph,
            title=f"{args.env}_{policy_name}",
            save_path=f"outputs/graphs/{args.experiment_name}/{test_name}"
        )

    time_graph_phase = time.time() - time_graph_phase

    # Step 3: Compute graph comparison metrics and select the best graph for contrastive explanation generation

    time_comparison_phase = time.time() 

    pol_names = list(all_graphs.keys())
    graph_dicts = list(all_graphs.values())

    K_wl, K_sm = compare_explanation_graphs(graph_dicts)
      
    avg_similarity = (K_wl + K_sm) / 2

    print("Weisfeiler-Lehman Kernel Matrix:")
    print_kernel_table(K_wl)

    print("Subgraph Matching Kernel Matrix:")
    print_kernel_table(K_sm)

    print("Avg. simillarity Matrix:")
    print_kernel_table(avg_similarity)
    
    # We need to select 2 graphs to compare for generating contrastive explanations, for that, we can use 
    # kernel similarity metrics to select the most similar graphs, as they are more likely to have interesting contrasts.
    # As selection criteria, we implement 3 options, selecting according to:
    # 1. The graph with the highest combined average similarity, for both WL and SM (the ones with the highest 
    #    average similarity between them)
    # 2. The graph with the highest similarity based on WL kernel (most similar in structure)
    # 3. The graph with the highest similarity based on SM kernel (most similar in subgraph patterns)

    # Select which similarity matrix will be used for pair selection
    if args.compare_criterion == "combined":
        selection_matrix = avg_similarity
        type_of_similarity = "combined average similarity"

    elif args.compare_criterion == "wl":
        selection_matrix = K_wl
        type_of_similarity = "WL kernel similarity"

    elif args.compare_criterion == "sm":
        selection_matrix = K_sm
        type_of_similarity = "SM kernel similarity"

    else:
        raise ValueError(
            f"Unknown comparison criterion: {args.compare_criterion}"
        )


    print(
        f"\nSelecting top {args.top_x_policies} policy pairs "
        f"based on {type_of_similarity}..."
    )

    # IMPORTANT:
    # Work on a copy so we do not modify the original kernel matrix.
    working_similarity = selection_matrix.copy()

    pairs = []

    for pair_idx in range(args.top_x_policies):

        selected_pair = select_most_similar_pair(working_similarity)

        # Depending on how your function is written, adapt this if necessary.
        if selected_pair is None:
            print(
                f"No more valid non-identical pairs available. "
                f"Selected {len(pairs)} pairs."
            )
            break

        id_graph1, id_graph2 = selected_pair

        if id_graph1 is None or id_graph2 is None:
            print(
                f"No more valid non-identical pairs available. "
                f"Selected {len(pairs)} pairs."
            )
            break

        pairs.append((id_graph1, id_graph2))

        print(
            f"Pair {pair_idx + 1}: "
            f"{pol_names[id_graph1]} - {pol_names[id_graph2]} "
            f"(similarity = {selection_matrix[id_graph1, id_graph2]:.4f})"
        )

        # Prevent this pair from being selected again
        working_similarity[id_graph1, id_graph2] = -1
        working_similarity[id_graph2, id_graph1] = -1
    
    time_comparison_phase = time.time() - time_comparison_phase

    # Step 4: Generate contrastive explanations for selected pairs

    if args.env == "MO_fruitTree":
        test_states = test_states_ft
    elif args.env == "MO_highway":
        test_states = test_states_hw
    elif args.env == "MO_deepSea":
        test_states = test_states_dst
    else:
        raise ValueError(f"Unknown environment: {args.env}")


    # Accumulate times across all selected pairs
    time_action_explanation_phase = 0
    time_edge_explanation_phase = 0
    time_subgraph_explanation_phase = 0
    time_transition_difference_phase = 0


    for pair_idx, (id_graph1, id_graph2) in enumerate(pairs, start=1):

        policy1 = pol_names[id_graph1]
        policy2 = pol_names[id_graph2]

        print(
            f"\n\n=================================================="
            f"\nPAIR {pair_idx}/{len(pairs)}"
            f"\n{policy1} vs {policy2}"
            f"\n=================================================="
        )

        print(
            f"Selection similarity "
            f"({type_of_similarity}): "
            f"{selection_matrix[id_graph1, id_graph2]:.4f}"
        )


        # --------------------------------------------------
        # ACTION EXPLANATIONS
        # --------------------------------------------------

        """start_time = time.time()

        output_file = (
            f"outputs/action_differences/"
            f"{args.experiment_name}/{test_name}/"
            f"cf_{id_graph1}_{id_graph2}.csv"
        )

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow([
                "test_state",
                f"abstract_state_{policy1}",
                f"next_action_{policy1}",
                f"abstract_state_{policy2}",
                f"next_action_{policy2}"
            ])

            for state_idx, test_state in enumerate(test_states):

                print(
                    f"\n--- Test state {state_idx}: "
                    f"{test_state} ---"
                )

                group_for_state_graph1 = assign_cluster_to_state(
                    graph_dicts[id_graph1]["groups"],
                    test_state,
                    graph_dicts[id_graph1]["state_features"]
                )

                group_for_state_graph2 = assign_cluster_to_state(
                    graph_dicts[id_graph2]["groups"],
                    test_state,
                    graph_dicts[id_graph2]["state_features"]
                )

                next_action_policy_1, next_abstract_state_policy_1 = (
                    get_next_probable_action(
                        graph_dicts[id_graph1],
                        group_for_state_graph1
                    )
                )

                next_action_policy_2, next_abstract_state_policy_2 = (
                    get_next_probable_action(
                        graph_dicts[id_graph2],
                        group_for_state_graph2
                    )
                )

                print(
                    f"State analysis for {policy1}: "
                    f"{group_for_state_graph1}"
                )

                print(
                    f"State analysis for {policy2}: "
                    f"{group_for_state_graph2}"
                )

                print(
                    f"Next probable action for {policy1}: "
                    f"{next_action_policy_1}"
                )

                print(
                    f"Next probable action for {policy2}: "
                    f"{next_action_policy_2}"
                )

                writer.writerow([
                    test_state,
                    group_for_state_graph1,
                    next_action_policy_1,
                    group_for_state_graph2,
                    next_action_policy_2
                ])

        time_action_explanation_phase += time.time() - start_time"""


        # --------------------------------------------------
        # EDGE / TRANSITION SET EXPLANATION
        # --------------------------------------------------

        start_time = time.time()

        common, only_g1, only_g2 = compare_transition_sets(
            graph_dicts[id_graph1],
            graph_dicts[id_graph2]
        )

        print(
            f"\n--- Transition comparison: "
            f"{policy1} vs {policy2} ---"
        )

        print(f"Common transitions: {common}")
        print(f"Only in {policy1}: {only_g1}")
        print(f"Only in {policy2}: {only_g2}")

        time_edge_explanation_phase += time.time() - start_time


        # --------------------------------------------------
        # MAXIMUM COMMON SUBGRAPH
        # --------------------------------------------------

        start_time = time.time()

        common_subgraph, mapping = get_maximum_common_subgraph(
            graph_dicts[id_graph1],
            graph_dicts[id_graph2]
        )

        print(
            f"\n--- Maximum common subgraph: "
            f"{policy1} vs {policy2} ---"
        )

        print("MCS mapping:", mapping)

        if common_subgraph is not None:
            print(
                "Common nodes:",
                common_subgraph.nodes(data=True)
            )
            print(
                "Common edges:",
                common_subgraph.edges(data=True)
            )
        else:
            print("No common subgraph found.")

        time_subgraph_explanation_phase += time.time() - start_time


        # --------------------------------------------------
        # ACTION DIFFERENCES AT COMMON NODES
        # --------------------------------------------------

        start_time = time.time()

        # Only run this if a valid mapping exists
        if mapping:

            action_differences = (
                get_transition_differences_at_common_nodes(
                    graph_dicts[id_graph1],
                    graph_dicts[id_graph2],
                    mapping
                )
            )

        else:
            action_differences = []

        print(
            f"\n--- Action differences at common nodes between "
            f"{policy1} and {policy2} ---"
        )

        if not action_differences:

            print(
                "No action differences found at common nodes.\n"
                "The selected policies may be behaviorally identical "
                "under the current graph representation."
            )

        else:

            print(
                f"Differences: {len(action_differences)}\n"
                f"{policy1} nodes: "
                f"{len(graph_dicts[id_graph1]['groups'])}\n"
                f"{policy2} nodes: "
                f"{len(graph_dicts[id_graph2]['groups'])}"
            )

            for diff in action_differences:

                print(
                    f"Node id in {policy1}: "
                    f"{diff['node_g1']}"
                )

                print(
                    f"Node in {policy1}: "
                    f"{graph_dicts[id_graph1]['groups'][diff['node_g1'] - 1]}"
                )

                print(
                    f"Node id in {policy2}: "
                    f"{diff['node_g2']}"
                )

                print(
                    f"Node in {policy2}: "
                    f"{graph_dicts[id_graph2]['groups'][diff['node_g2'] - 1]}"
                )

                print(f"State: {diff['label']}")
                print(f"Only in {policy1}: {diff['only_g1']}")
                print(f"Only in {policy2}: {diff['only_g2']}\n")


        # Save one action-difference CSV per pair
        differences_output_file = (
            f"outputs/action_differences/"
            f"{args.experiment_name}/{test_name}/"
            f"action_differences_{id_graph1}_{id_graph2}.csv"
        )

        with open(
            differences_output_file,
            "w",
            newline=""
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                "node_id_g1",
                "node_g1",
                "node_id_g2",
                "node_g2",
                "state",
                "only_g1",
                "only_g2",
                "all_transitions_g1",
                "all_transitions_g2"
            ])

            for diff in action_differences:

                writer.writerow([
                    diff["node_g1"],
                    graph_dicts[id_graph1]["groups"][
                        diff["node_g1"] - 1
                    ],
                    diff["node_g2"],
                    graph_dicts[id_graph2]["groups"][
                        diff["node_g2"] - 1
                    ],
                    diff["label"],
                    diff["only_g1"],
                    diff["only_g2"],
                    diff["transitions_g1"],
                    diff["transitions_g2"]
                ])

        time_transition_difference_phase += (
            time.time() - start_time
        )

    # --------------------------------------------------
    # GLOBAL GRAPH COMPARISON STATISTICS
    # --------------------------------------------------

    start_time = time.time()

    print(
        "\n\n--- Shared subgraph percentage "
        "between all policy graphs ---"
    )

    print("Weisfeiler-Lehman Kernel Matrix:")
    print_kernel_table(K_wl)

    print("Subgraph Matching Kernel Matrix:")
    print_kernel_table(K_sm)

    print("Average Similarity Matrix:")
    print_kernel_table(avg_similarity)


    common_nodes_percentage, common_edges_percentage = (
        build_common_percentage_matrices(graph_dicts)
    )

    print("\nCommon nodes:")
    print_percentage_table(common_nodes_percentage)

    print("\nCommon edges:")
    print_percentage_table(common_edges_percentage)


    g1_in_g2_nodes_percentage, g1_in_g2_edges_percentage = (
        percentage_table_g1_in_g2(graph_dicts)
    )

    print("\nPercentage of G1 nodes contained in G2:")
    print_percentage_table(g1_in_g2_nodes_percentage)

    print("\nPercentage of G1 edges contained in G2:")
    print_percentage_table(g1_in_g2_edges_percentage)


    time_subgraph_percentage_explanation_phase = (
        time.time() - start_time
    )

    # Calculate total time for explanation generation phase

    selected_pairs_text = "\n".join(
        [
            f"        {idx + 1}. "
            f"{pol_names[g1]} vs {pol_names[g2]} "
            f"(similarity: {selection_matrix[g1, g2]:.4f})"
            for idx, (g1, g2) in enumerate(pairs)
        ]
    )

    time_action_explanation_phase = 0
    
    total_explanation_phase_time = (
        time_action_explanation_phase
        + time_edge_explanation_phase
        + time_subgraph_explanation_phase
        + time_transition_difference_phase
        + time_subgraph_percentage_explanation_phase
    )

    print(f"""
    --- Summary of execution ---

    -- args used --

    Environment: {args.env}
    Number of episodes: {args.num_episodes}
    Lambda value: {args.lmbda}
    Graph comparison criterion: {args.compare_criterion}
    Requested top pairs: {args.top_x_policies}
    Actual pairs analysed: {len(pairs)}
    Used stored graphs: {args.use_existing}

    -- Selected pairs --

    {selected_pairs_text}

    -- Times for each phase --

    Graph generation phase:
        {time_graph_phase:.2f} seconds

    Graph comparison phase:
        {time_comparison_phase:.2f} seconds

    Action explanation generation phase:
        {time_action_explanation_phase:.2f} seconds

    Edge explanation generation phase:
        {time_edge_explanation_phase:.2f} seconds

    Subgraph explanation generation phase:
        {time_subgraph_explanation_phase:.2f} seconds

    Transition difference phase:
        {time_transition_difference_phase:.2f} seconds

    Subgraph percentage phase:
        {time_subgraph_percentage_explanation_phase:.2f} seconds

    Total explanation generation phase:
        {total_explanation_phase_time:.2f} seconds

    --- End of execution ---
    """)

