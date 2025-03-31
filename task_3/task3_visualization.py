import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any


def plot_experiment_results(results, experiment_index, data):
    """Plot results for a single experiment"""
    fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    
    # Time points
    t = np.arange(data['num_timeslots'])
    
    # Plot 1: Prices and wind power
    ax1 = axs[0]
    ax1.plot(t, results['price_trajectories'][experiment_index], 'r-', label='Price')
    ax1.set_ylabel('Price')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(t, results['wind_trajectories'][experiment_index], 'b-', label='Wind')
    ax1_twin.set_ylabel('Wind Power')
    ax1.set_title('Price and Wind Trajectories')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    
    # Plot 2: Hydrogen Storage Level
    ax2 = axs[1]
    ax2.plot(t, results['hydrogen_storage'][experiment_index, :-1], 'g-', label='H2 Storage')
    ax2.set_ylabel('Hydrogen Level')
    ax2.set_title('Hydrogen Storage Level')
    ax2.legend()
    
    # Plot 3: Electrolyzer Status and Operations
    ax3 = axs[2]
    ax3.plot(t, results['electrolyzer_status'][experiment_index, :-1], 'k-', label='Electrolyzer Status')
    ax3.plot(t, results['p_grid'][experiment_index], 'c-', label='Grid Power')
    ax3.plot(t, results['p_p2h'][experiment_index], 'm-', label='P2H')
    ax3.plot(t, results['p_h2p'][experiment_index], 'y-', label='H2P')
    ax3.set_ylabel('Power / Status')
    ax3.set_title('Electrolyzer Operations')
    ax3.legend()
    
    # Plot 4: Costs
    ax4 = axs[3]
    ax4.plot(t, results['policy_cost'][experiment_index], 'r-', label='Period Cost')
    ax4.plot(t, np.cumsum(results['policy_cost'][experiment_index]), 'b-', label='Cumulative Cost')
    ax4.set_ylabel('Cost')
    ax4.set_title('Period and Cumulative Costs')
    ax4.set_xlabel('Time Period')
    ax4.legend()
    
    plt.tight_layout()
    #plt.savefig(f'{experiment_index}_results.png')
    #plt.close()


def plot_cost_histogram(results, policy_name):
    """Plot histogram of costs across all experiments for a policy"""
    plt.figure(figsize=(10, 6))
    plt.hist(results['total_costs'], bins=10, alpha=0.7, color='blue')
    plt.axvline(np.mean(results['total_costs']), color='red', linestyle='dashed', linewidth=2)
    plt.title(f'Cost Distribution for {policy_name}')
    plt.xlabel('Total Cost')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    #plt.savefig(f'{policy_name.replace(" ", "_").lower()}_cost_histogram.png')
    #plt.close()


def compare_policies(policy_results):
    """Compare the performance of all policies"""
    plt.figure(figsize=(12, 8))
    
    # Calculate average costs
    avg_costs = {}
    std_costs = {}
    
    for policy_name, results in policy_results.items():
        avg_costs[policy_name] = np.mean(results['total_costs'])
        std_costs[policy_name] = np.std(results['total_costs'])
    
    # Sort policies by average cost (ascending)
    sorted_policies = sorted(avg_costs.items(), key=lambda x: x[1])
    policy_names = [p[0] for p in sorted_policies]
    costs = [p[1] for p in sorted_policies]
    stds = [std_costs[p[0]] for p in sorted_policies]
    
    # Create bar chart
    x = np.arange(len(policy_names))
    plt.bar(x, costs, yerr=stds, align='center', alpha=0.7, ecolor='black', capsize=10)
    plt.xticks(x, policy_names, rotation=45, ha='right')
    plt.ylabel('Average Total Cost')
    plt.title('Policy Comparison: Average Total Cost (Lower is Better)')
    plt.tight_layout()
    plt.grid(True, axis='y', alpha=0.3)
    #plt.savefig('policy_comparison.png')
    #plt.close()
    
    # Print comparison table
    print("\n----- Policy Comparison -----")
    print(f"{'Policy':<30} {'Avg Cost':<10} {'Std Dev':<10} {'Min':<10} {'Max':<10}")
    print("-" * 70)
    
    for name in policy_names:
        avg = avg_costs[name]
        std = std_costs[name]
        min_cost = np.min(policy_results[name]['total_costs'])
        max_cost = np.max(policy_results[name]['total_costs'])
        print(f"{name:<30} {avg:<10.2f} {std:<10.2f} {min_cost:<10.2f} {max_cost:<10.2f}")


def plot_feature_importance(theta):
    """Plot the importance of each feature in the value function"""
    # Feature names
    feature_names = ['Bias', 'Price', 'Wind', 'Hydrogen', 'Electrolyzer Status', 'H2 × Price']
    
    # Calculate absolute values for importance
    importance = np.abs(theta)
    
    # Normalize
    normalized_importance = importance / np.sum(importance)
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, normalized_importance, color='skyblue')
    plt.ylabel('Normalized Importance')
    plt.title('Feature Importance in Value Function Approximation')
    plt.xticks(rotation=45)
    plt.tight_layout()
    #plt.savefig('feature_importance.png')
    #plt.close()
    
    # Print numerical values
    print("\n----- Feature Importance -----")
    for name, imp in zip(feature_names, normalized_importance):
        print(f"{name}: {imp:.4f}")