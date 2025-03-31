import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List

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

def compare_policies(
    policy_results: Dict[str, Dict[str, Any]],
) -> None:
    """
    Compares the costs of multiple policies using boxplots.
    
    Args:
        policy_results: Dictionary mapping policy names to their results
    """
    plt.figure(figsize=(12, 6))
    
    data = []
    names = []
    
    for name, results in policy_results.items():
        data.append(results['total_costs'])
        names.append(name)
    
    plt.boxplot(data, labels=names, patch_artist=True, 
                boxprops=dict(facecolor='skyblue', color='black'),
                medianprops=dict(color='red'))
    
    plt.title('Cost Distribution Comparison Between Policies')
    plt.xlabel('Policy')
    plt.ylabel('Total Cost')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.show()
