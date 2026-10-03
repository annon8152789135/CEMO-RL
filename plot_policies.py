import os
import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def plot_policy_returns(
    env_name,
    dpmorl_output_dir,
    save_dir,
    batch_size=1,
    final_episodes=100,
    reward_dims=None
):
    """
    Plot the return distributions of the trained DPMORL policies.

    Supports environments with either 2 or 3 objectives.

    Parameters
    ----------
    env_name : str
        Environment name used for the plot title.

    dpmorl_output_dir : str
        Directory containing the DPMORL MORL*.npz files.

    save_dir : str
        Directory where the plot should be saved.

    batch_size : int
        Number of episodes to average into one plotted point.

    final_episodes : int
        Number of final episodes from each policy to plot.
    """

    os.makedirs(save_dir, exist_ok=True)

    file_paths = sorted(
        glob.glob(os.path.join(dpmorl_output_dir, "MORL*.npz"))
    )

    if len(file_paths) == 0:
        print(
            f"No MORL .npz files found in:\n"
            f"{dpmorl_output_dir}"
        )
        return

    print(
        f"\nPlotting return distributions for "
        f"{len(file_paths)} policies..."
    )


    # Detect total number of reward objectives in the data.
    total_objectives = None

    for file_path in file_paths:

        data = np.load(file_path)

        if "episode_vec_returns" in data:
            episode_vec_returns = data["episode_vec_returns"]

            if episode_vec_returns.ndim != 2:
                continue

            total_objectives = episode_vec_returns.shape[1]
            break

    if total_objectives is None:
        print("No valid 'episode_vec_returns' arrays found.")
        return


    # Decide which reward dimensions to plot.
    if reward_dims is None:

        if total_objectives not in [2, 3]:
            print(
                f"Environment has {total_objectives} reward dimensions. "
                f"Please specify 2 or 3 dimensions using reward_dims."
            )
            return

        reward_dims = list(range(total_objectives))

    else:

        if len(reward_dims) not in [2, 3]:
            print(
                "reward_dims must contain exactly 2 or 3 dimensions."
            )
            return

        if any(
            dim < 0 or dim >= total_objectives
            for dim in reward_dims
        ):
            print(
                f"Invalid reward_dims {reward_dims}. "
                f"Available dimensions are "
                f"0-{total_objectives - 1}."
            )
            return

        if len(set(reward_dims)) != len(reward_dims):
            print(
                "reward_dims must contain unique dimensions."
            )
            return


    num_objectives = len(reward_dims)

    print(
        f"Detected {total_objectives} reward dimensions. "
        f"Plotting dimensions {reward_dims}."
    )

    colors = matplotlib.colormaps["gist_rainbow"](
        np.linspace(0, 1, len(file_paths))
    )

    markers = ["o", "v", "^", "s", "p", "*", "h", "D", "X", "<", ">"]

    fig = plt.figure(figsize=(10, 6))
    
    if num_objectives == 3:
        ax = fig.add_subplot(111, projection="3d")
    else:
        ax = fig.add_subplot(111)

    handles = []
    labels = []
    return_stats = []

    for policy_index, file_path in enumerate(file_paths):

        data = np.load(file_path)

        if "episode_vec_returns" not in data:
            print(
                f"Skipping {file_path}: "
                "'episode_vec_returns' not found."
            )
            continue

        episode_vec_returns = data["episode_vec_returns"]

        # Only keep the objectives we are plotting.
        episode_vec_returns = episode_vec_returns[
            :, reward_dims
        ]

        # Take only the final episodes.
        final_returns = episode_vec_returns[
            -final_episodes * batch_size:
        ]
        
        episode_batches = [
            np.mean(
                final_returns[i:i + batch_size],
                axis=0
            )
            for i in range(
                0,
                len(final_returns),
                batch_size
            )
        ]

        episode_batches = np.asarray(episode_batches)

        if len(episode_batches) == 0:
            continue

        mean_returns = np.mean(
            episode_batches,
            axis=0
        )

        std_returns = np.std(
            episode_batches,
            axis=0
        )

        return_stats.append({
            "policy": policy_index,
            "mean": mean_returns,
            "std": std_returns
        })

        marker = markers[
            (policy_index // 10) % len(markers)
        ]

        print(
            f"Policy {policy_index}: {len(episode_batches)} episode batches"
        )

        if num_objectives == 3:
            scatter = ax.scatter(
                episode_batches[:, 0],
                episode_batches[:, 1],
                episode_batches[:, 2],
                alpha=0.6,
                marker=marker,
                color=colors[policy_index],
                s=40,
                edgecolors="black",
                linewidths=0.2
            )
        else:

            scatter = ax.scatter(
                episode_batches[:, 0],
                episode_batches[:, 1],
                alpha=0.6,
                marker=marker,
                color=colors[policy_index],
                s=40,
                edgecolors="black",
                linewidths=0.2
            )

        handles.append(scatter)
        labels.append(f"Policy {policy_index}")

    ax.set_title(
        f"{env_name} - Final Policy Return Distribution"
    )

    ax.set_xlabel(f"Return {reward_dims[0]}", labelpad=15, fontsize=12)
    ax.set_ylabel(f"Return {reward_dims[1]}", labelpad=15, fontsize=12)

    if num_objectives == 3:
        ax.set_zlabel(f"Return {reward_dims[2]}", labelpad=15, fontsize=12)

    ax.legend(
        handles,
        labels,
        title="Policies",
        loc="center left",
        bbox_to_anchor=(1.18, 0.5),
        borderaxespad=0.0,
        fontsize=10
    )

    output_path = os.path.join(
        save_dir,
        f"{env_name}_policy_returns.png"
    )

    plt.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight"
    )

    plt.close(fig)

    stats_output_path = os.path.join(
        save_dir,
        f"{env_name}_policy_return_stats.txt"
    )

    with open(stats_output_path, "w") as f:

        # Header
        header = f"{'Policy':<10}"

        for dim in reward_dims:
            header += (
                f"{f'Return {dim} Mean':>18}"
                f"{f'Return {dim} Std':>18}"
            )

        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        # Policy rows
        for stats in return_stats:

            policy_name = f"Policy {stats['policy']}"
            row = f"{policy_name:<10}"

            for objective_index in range(num_objectives):

                row += (
                    f"{stats['mean'][objective_index]:>18.4f}"
                    f"{stats['std'][objective_index]:>18.4f}"
                )

            f.write(row + "\n")

    print(f"Policy return plot saved to: {output_path}")