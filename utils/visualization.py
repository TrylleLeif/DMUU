import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional

# ============================================
# || Plot results following template        ||
# ============================================

def plot_results(wind_trajectory: np.ndarray, 
                price_trajectory: np.ndarray, 
                results: Dict[str, List[float]], 
                data: Dict[str, Any], 
                title: str) -> None:
    times = range(data['num_timeslots'])
    
    plt.figure(figsize=(14, 10))
    
    plt.subplot(8, 1, 1)
    plt.plot(times, wind_trajectory, label="Wind Power", color="blue")
    plt.ylabel("Wind Power")
    plt.legend()
    plt.title(title)
    
    plt.subplot(8, 1, 2)
    plt.plot(times, data['demand_schedule'], label="Demand Schedule", color="orange")
    plt.ylabel("Demand")
    plt.legend()
    
    plt.subplot(8, 1, 3)
    plt.step(times, results['electrolyzer_status'], label="Electrolyzer Status", color="red", where="post")
    plt.ylabel("El. Status")
    plt.legend()
    
    plt.subplot(8, 1, 4)
    plt.plot(times, results['hydrogen_storage_level'], label="Hydrogen Level", color="green")
    plt.ylabel("Hydr. Level")
    plt.legend()
    
    plt.subplot(8, 1, 5)
    plt.plot(times, results['power_to_hydrogen'], label="p2h", color="orange")
    plt.ylabel("p2h")
    plt.legend()
    
    plt.subplot(8, 1, 6)
    plt.plot(times, results['hydrogen_to_power'], label="h2p", color="blue")
    plt.ylabel("h2p")
    plt.legend()
    
    plt.subplot(8, 1, 7)
    plt.plot(times, results['grid_power'], label="Grid Power", color="green")
    plt.ylabel("Grid Power")
    plt.legend()
    
    plt.subplot(8, 1, 8)
    plt.plot(times, price_trajectory, label="price", color="red")
    plt.ylabel("Price")
    plt.xlabel("Time")
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def plot_experiment_results(
    results: Dict[str, Any],
    experiment_index: int = 0,
    data: Dict[str, Any] = None
) -> None:
    
    e = experiment_index
    num_timeslots = results['wind_trajectories'].shape[1]
    times = range(num_timeslots)
    
    plt.figure(figsize=(14, 12))
    
    # Plot wind power
    plt.subplot(7, 1, 1)
    plt.plot(times, results['wind_trajectories'][e], label="Wind Power", color="blue")
    plt.ylabel("Wind Power")
    plt.legend()
    plt.title(f"Experiment {e} Results")
    
    # Plot demand schedule
    plt.subplot(7, 1, 2)
    if data:
        plt.plot(times, data['demand_schedule'], label="Demand Schedule", color="orange")
    plt.ylabel("Demand")
    plt.legend()
    
    # Plot electrolyzer status
    plt.subplot(7, 1, 3)
    plt.step(times, results['electrolyzer_status'][e, :-1], label="Electrolyzer Status", color="red", where="post")
    plt.ylabel("El. Status")
    plt.legend()
    
    # Plot hydrogen storage level
    plt.subplot(7, 1, 4)
    plt.plot(range(num_timeslots+1), results['hydrogen_storage'][e], label="Hydrogen Level", color="green")
    plt.ylabel("Hydr. Level")
    plt.legend()
    
    # Plot power to hydrogen
    plt.subplot(7, 1, 5)
    plt.plot(times, results['p_p2h'][e], label="p2h", color="orange")
    plt.ylabel("p2h")
    plt.legend()
    
    # Plot hydrogen to power
    plt.subplot(7, 1, 6)
    plt.plot(times, results['p_h2p'][e], label="h2p", color="blue")
    plt.ylabel("h2p")
    plt.legend()
    
    # Plot grid power
    plt.subplot(7, 1, 7)
    plt.plot(times, results['p_grid'][e], label="Grid Power", color="purple")
    plt.ylabel("Grid Power")
    plt.xlabel("Time")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_cost_histogram(
    results: Dict[str, Any], 
    policy_name: str = "Policy"
) -> None:
    """
    Plots a histogram of the total costs across experiments and saves to file.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(results['total_costs'], bins=10, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(np.mean(results['total_costs']), color='red', linestyle='dashed', linewidth=2, 
               label=f'Mean: {np.mean(results["total_costs"]):.2f}')
    plt.title(f'Total Cost Distribution for {policy_name}')
    plt.xlabel('Total Cost')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# def compare_policies(
#     policy_results: Dict[str, Dict[str, Any]],
#     bins: int = 20,
#     figsize: tuple = (10, 12),
#     colors: List[str] = None,
# ) -> None:
#     """
#     Compares the costs of multiple policies using histograms in a vertical layout.
    
#     Args:
#         policy_results: Dictionary mapping policy names to their results dictionaries
#         bins: Number of bins for histograms
#         figsize: Figure size (width, height)
#         colors: List of colors for each policy (will use default if None)
#     """
#     import matplotlib.pyplot as plt
#     import numpy as np
    
#     n_policies = len(policy_results)
    
#     if colors is None:
#         # Default colormap choices if colors not specified
#         cmap = plt.cm.get_cmap('tab10', n_policies)
#         colors = [cmap(i) for i in range(n_policies)]
    
#     # Create subplots - one for each policy
#     fig, axes = plt.subplots(n_policies, 1, figsize=figsize, sharex=True)
#     plt.subplots_adjust(hspace=0.4)  # Add some space between subplots
    
#     # Find global min and max for consistent x-axis
#     all_costs = np.concatenate([results['total_costs'] for results in policy_results.values()])
#     x_min = max(np.min(all_costs) - 10, 0)  # Avoid negative values for costs
#     x_max = np.max(all_costs) + 10
    
#     # Plot each policy
#     for i, (policy_name, results) in enumerate(policy_results.items()):
#         ax = axes[i] if n_policies > 1 else axes
#         costs = results['total_costs']
#         mean_cost = np.mean(costs)
#         std_cost = np.std(costs)
        
#         # Create histogram
#         ax.hist(costs, bins=bins, alpha=0.7, color=colors[i], edgecolor='black')
        
#         # Add mean line
#         ax.axvline(mean_cost, color='darkred', linestyle='--', linewidth=2)
        
#         # Add legend with statistics
#         ax.legend([
#             f"{policy_name}",
#             f"Mean: {mean_cost:.2f}"
#         ], loc='upper right')
        
#         # Set y-axis label only for middle plot
#         if i == 0:
#             ax.set_title('Comparison of the different models objective values')
#         if i == n_policies // 2:
#             ax.set_ylabel('Frequency')
        
#         # Set x and y limits
#         ax.set_xlim(x_min, x_max)
        
#         # Grid
#         ax.grid(True, alpha=0.3)
        
#     # Set common x-axis label for the bottom subplot
#     axes[-1].set_xlabel('Objective Value')
    
#     plt.tight_layout()
#     plt.show()
    
#     # Also create a box plot for direct comparison
#     plt.figure(figsize=(10, 6))
    
#     data = []
#     names = []
    
#     for name, results in policy_results.items():
#         data.append(results['total_costs'])
#         names.append(f"{name}\nMean: {np.mean(results['total_costs']):.2f}")
    
#     plt.boxplot(data, labels=names, patch_artist=True, 
#                 boxprops=dict(facecolor='skyblue', color='black'),
#                 medianprops=dict(color='red'))
    
#     plt.title('Cost Distribution Comparison Between Policies')
#     plt.ylabel('Total Cost')
#     plt.grid(True, alpha=0.3, axis='y')
#     plt.xticks(rotation=15)
    
#     plt.tight_layout()
#     plt.show()

def compare_policies(
    policy_results: Dict[str, Dict[str, Any]],
    bucket_ranges: List[tuple] = None,
    figsize: tuple = (10, 12),
    colors: List[str] = None,
) -> None:
    """
    Compares the costs of multiple policies using histograms in a vertical layout with defined buckets.
    
    Args:
        policy_results: Dictionary mapping policy names to their results dictionaries
        bucket_ranges: List of tuples defining bucket ranges [(min_val, max_val), ...]. 
                       If None, auto-generates ranges.
        figsize: Figure size (width, height)
        colors: List of colors for each policy (will use default if None)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    
    # Set the Seaborn style
    sns.set_theme(style="whitegrid")
    
    n_policies = len(policy_results)
    if colors is None:
        # Use Seaborn's color palette
        colors = sns.color_palette("husl", n_policies)
    
    # Create subplots - one for each policy
    fig, axes = plt.subplots(n_policies, 1, figsize=figsize, sharex=True)
    plt.subplots_adjust(hspace=0.4)  # Add some space between subplots
    
    # Find global min and max for consistent x-axis
    all_costs = np.concatenate([results['total_costs'] for results in policy_results.values()])
    global_min = max(np.min(all_costs), 0)  # Avoid negative values for costs
    global_max = np.max(all_costs)
    
    # Define bucket ranges if not provided
    if bucket_ranges is None:
        # Create default buckets: [min-100], [100-300], [300-500], [500-750], [750-1000], [1000-max]
        min_val = int(global_min)
        max_val = int(global_max)
        
        if min_val < 100:
            bucket_ranges = [(min_val, 100)]
        else:
            bucket_ranges = []
            
        if max_val > 1000:
            bucket_ranges.extend([(100, 300), (300, 500), (500, 750), (750, 1000), (1000, max_val)])
        else:
            # Adjust buckets if max is less than 1000
            remaining = max_val - max(min_val, 100)
            step = remaining / 4
            
            current = max(min_val, 100)
            while current < max_val:
                next_val = min(current + step, max_val)
                bucket_ranges.append((int(current), int(next_val)))
                current = next_val
    
    # Create bin edges from bucket ranges
    bin_edges = [range[0] for range in bucket_ranges] + [bucket_ranges[-1][1]]
    
    # Plot each policy
    for i, (policy_name, results) in enumerate(policy_results.items()):
        ax = axes[i] if n_policies > 1 else axes
        costs = results['total_costs']
        mean_cost = np.mean(costs)
        std_cost = np.std(costs)
        
        # Create histogram with Seaborn
        sns.histplot(
            costs, 
            bins=bin_edges, 
            ax=ax, 
            color=colors[i], 
            edgecolor='black',
            alpha=0.7,
            kde=True  # Add density curve
        )
        
        # Add mean line
        ax.axvline(mean_cost, color='darkred', linestyle='--', linewidth=2)
        
        # Add bucket labels
        bucket_labels = [f"[{r[0]}-{r[1]}]" for r in bucket_ranges]
        tick_positions = [(r[0] + r[1]) / 2 for r in bucket_ranges]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(bucket_labels, rotation=45)
        
        # Add legend with statistics
        ax.legend([
            f"{policy_name}",
            f"Mean: {mean_cost:.2f}, Std: {std_cost:.2f}"
        ], loc='upper right')
        
        # Set titles and labels
        if i == 0:
            ax.set_title('Comparison of Policy Objective Values')
        if i == n_policies // 2:
            ax.set_ylabel('Frequency')
        
        # Set x limits
        buffer = (global_max - global_min) * 0.05
        ax.set_xlim(global_min - buffer, global_max + buffer)
    
    # Set common x-axis label for the bottom subplot
    axes[-1].set_xlabel('Objective Value')
    plt.tight_layout()
    
    # Also create a box plot for direct comparison using Seaborn
    plt.figure(figsize=(10, 6))
    
    # Prepare data for boxplot
    data_list = []
    for name, results in policy_results.items():
        for cost in results['total_costs']:
            data_list.append({
                'Policy': name,
                'Cost': cost,
                'Mean': np.mean(results['total_costs'])
            })
    
    import pandas as pd
    boxplot_data = pd.DataFrame(data_list)
    
    # Create boxplot with Seaborn
    sns.boxplot(
        x='Policy', 
        y='Cost', 
        data=boxplot_data,
        palette=colors
    )
    
    # Add text annotations for means
    for i, name in enumerate(policy_results.keys()):
        mean_val = np.mean(policy_results[name]['total_costs'])
        plt.text(
            i, 
            mean_val, 
            f"Mean: {mean_val:.2f}", 
            ha='center', 
            va='bottom',
            fontweight='bold'
        )
    
    plt.title('Cost Distribution Comparison Between Policies')
    plt.ylabel('Total Cost')
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=15)
    plt.tight_layout()