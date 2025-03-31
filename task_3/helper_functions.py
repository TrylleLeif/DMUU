import numpy as np
from typing import Dict, List, Tuple, Any
import sys, os, random
import matplotlib.pyplot as plt
# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)


from utils.data import get_fixed_data
from utils.WindProcess import wind_model
from utils.PriceProcess import price_model


def generate_trajectories(data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a single trajectory of wind and price for policy execution.
    
    Args:
        data: Dictionary containing problem parameters
        
    Returns:
        Tuple: (wind_trajectory, price_trajectory)
    """
    num_timeslots = data['num_timeslots']
    
    # Create arrays for trajectories
    wind_trajectory = np.zeros(num_timeslots)
    price_trajectory = np.zeros(num_timeslots)
    
    # Initialize first two values
    wind_trajectory[0] = data['target_mean_wind']
    wind_trajectory[1] = data['target_mean_wind']
    price_trajectory[0] = data['mean_price']
    price_trajectory[1] = data['mean_price']
    
    # Generate the rest of the trajectory
    for t in range(2, num_timeslots):
        wind_trajectory[t] = wind_model(
            wind_trajectory[t-1],
            wind_trajectory[t-2],
            data
        )
        price_trajectory[t] = price_model(
            price_trajectory[t-1],
            price_trajectory[t-2],
            wind_trajectory[t],
            data
        )
    
    return wind_trajectory, price_trajectory


def generate_next_states(
    current_wind: float, 
    current_price: float, 
    data: Dict[str, Any], 
    num_samples: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate sample next states from current wind and price values.
    
    Args:
        current_wind: Current wind power
        current_price: Current electricity price
        data: Problem data dictionary
        num_samples: Number of samples to generate
        
    Returns:
        Tuple of arrays: (next_wind_samples, next_price_samples)
    """
    next_wind_samples = np.zeros(num_samples)
    next_price_samples = np.zeros(num_samples)
    
    for i in range(num_samples):
        next_wind = wind_model(current_wind, current_wind, data)
        next_price = price_model(current_price, current_price, next_wind, data)
        
        next_wind_samples[i] = next_wind
        next_price_samples[i] = next_price
    
    return next_wind_samples, next_price_samples


def sample_state_space(
    data: Dict[str, Any], 
    num_samples: int = 10
) -> List[Tuple[float, float, float, int]]:
    """
    Sample representative states from the state space for VFA training.
    
    Args:
        data: Problem data dictionary
        num_samples: Number of state samples to generate
        
    Returns:
        List of state tuples: [(price, wind, hydrogen, electrolyzer_status), ...]
    """
    states = []
    
    # Sample price, wind, hydrogen and electrolyzer status
    for _ in range(num_samples):
        price = np.random.uniform(data['price_floor'], data['price_cap'])
        wind = np.random.uniform(0, 10)  # Assuming wind range is 0-10
        hydrogen = np.random.uniform(0, data['hydrogen_capacity'])
        status = np.random.choice([0, 1])
        
        states.append((price, wind, hydrogen, status))
    
    return states


def visualize_policy_decisions(results: Dict[str, Any], experiment_index: int = 0):
    """
    Visualize the decisions made by a policy for a single experiment.
    
    Args:
        results: Results dictionary from policy evaluation
        experiment_index: Index of the experiment to visualize
    """
    # Create subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Time axis
    t = np.arange(len(results['hydrogen_storage'][experiment_index]) - 1)
    
    # Plot 1: Hydrogen storage and electrolyzer status
    axes[0].plot(t, results['hydrogen_storage'][experiment_index, :-1], 'b-', label='Hydrogen Storage')
    axes[0].set_ylabel('Hydrogen Level')
    ax2 = axes[0].twinx()
    ax2.plot(t, results['electrolyzer_status'][experiment_index, :-1], 'r-', label='Electrolyzer Status')
    ax2.set_ylabel('Status (0/1)')
    axes[0].set_title('Hydrogen Storage and Electrolyzer Status')
    axes[0].legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    # Plot 2: Decision variables
    axes[1].plot(t, results['p_grid'][experiment_index], 'g-', label='Grid Power')
    axes[1].plot(t, results['p_p2h'][experiment_index], 'c-', label='Power to Hydrogen')
    axes[1].plot(t, results['p_h2p'][experiment_index], 'm-', label='Hydrogen to Power')
    axes[1].set_ylabel('Power')
    axes[1].set_title('Decision Variables')
    axes[1].legend()
    
    # Plot 3: Wind power, price and cost
    axes[2].plot(t, results['wind_trajectories'][experiment_index], 'b-', label='Wind Power')
    axes[2].set_ylabel('Wind Power')
    axes[2].set_xlabel('Time Period')
    ax3 = axes[2].twinx()
    ax3.plot(t, results['price_trajectories'][experiment_index], 'r-', label='Price')
    ax3.plot(t, results['policy_cost'][experiment_index], 'k--', label='Period Cost')
    ax3.set_ylabel('Price / Cost')
    axes[2].set_title('Wind Power, Price, and Cost')
    axes[2].legend(loc='upper left')
    ax3.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('policy_decisions.png')
    plt.show()


def calculate_feature_importance(theta: np.ndarray):
    """
    Calculate and visualize the importance of each feature in the value function.
    
    Args:
        theta: Trained parameter vector for value function approximation
    """
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
    plt.savefig('feature_importance.png')
    plt.show()
    
    # Print numerical values
    print("Feature Importance:")
    for name, imp in zip(feature_names, normalized_importance):
        print(f"{name}: {imp:.4f}")


def generate_value_function_heatmap(theta: np.ndarray, data: Dict[str, Any]):
    """
    Generate a heatmap visualization of the value function for different 
    combinations of hydrogen storage and price, with fixed wind and electrolyzer status.
    
    Args:
        theta: Trained parameter vector for value function approximation
        data: Dictionary containing problem parameters
    """
    from matplotlib import cm
    
    # Fixed values for wind and electrolyzer status
    wind = data['target_mean_wind']
    status = 1  # Electrolyzer on
    
    # Grid for hydrogen and price
    hydrogen_levels = np.linspace(0, data['hydrogen_capacity'], 20)
    prices = np.linspace(data['price_floor'], data['price_cap'], 20)
    
    # Create meshgrid
    H, P = np.meshgrid(hydrogen_levels, prices)
    
    # Calculate value function for each point
    V = np.zeros_like(H)
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            state = (P[i, j], wind, H[i, j], status)
            # Extract features for the value function
            features = np.array([
                1.0,                # Bias
                P[i, j],            # Price
                wind,               # Wind
                H[i, j],            # Hydrogen
                float(status),      # Electrolyzer status
                H[i, j] * P[i, j]   # Interaction term
            ])
            V[i, j] = -np.dot(theta, features)  # Negative because we're minimizing cost
    
    # Create the heatmap
    plt.figure(figsize=(12, 10))
    contour = plt.contourf(H, P, V, 20, cmap=cm.viridis)
    plt.colorbar(contour, label='Value Function')
    plt.xlabel('Hydrogen Storage Level')
    plt.ylabel('Electricity Price')
    plt.title('Value Function Heatmap (Wind = {:.2f}, Electrolyzer Status = {})'.format(wind, status))
    plt.tight_layout()
    plt.savefig('value_function_heatmap.png')
    plt.show()


def compare_adp_with_stochastic_programming(adp_results: Dict[str, Any], sp_results: Dict[str, Any]):
    """
    Compare the performance of ADP with Stochastic Programming approaches.
    
    Args:
        adp_results: Results dictionary from ADP policy evaluation
        sp_results: Results dictionary from SP policy evaluation
    """
    # Calculate statistics
    adp_mean = np.mean(adp_results['total_costs'])
    adp_std = np.std(adp_results['total_costs'])
    adp_min = np.min(adp_results['total_costs'])
    adp_max = np.max(adp_results['total_costs'])
    
    sp_mean = np.mean(sp_results['total_costs'])
    sp_std = np.std(sp_results['total_costs'])
    sp_min = np.min(sp_results['total_costs'])
    sp_max = np.max(sp_results['total_costs'])
    
    # Create bar plot for mean costs
    policies = ['ADP Policy', 'SP Policy']
    means = [adp_mean, sp_mean]
    stds = [adp_std, sp_std]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(policies, means, yerr=stds, capsize=10, color=['blue', 'orange'])
    plt.ylabel('Average Cost')
    plt.title('Policy Performance Comparison')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                 f'{height:.2f}', ha='center', va='bottom')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('adp_vs_sp_comparison.png')
    plt.show()
    
    # Print detailed comparison
    print("\n--- Policy Comparison ---")
    print(f"{'Policy':<15} {'Mean Cost':<12} {'Std Dev':<12} {'Min Cost':<12} {'Max Cost':<12}")
    print("-" * 65)
    print(f"{'ADP Policy':<15} {adp_mean:<12.2f} {adp_std:<12.2f} {adp_min:<12.2f} {adp_max:<12.2f}")
    print(f"{'SP Policy':<15} {sp_mean:<12.2f} {sp_std:<12.2f} {sp_min:<12.2f} {sp_max:<12.2f}")
    print("-" * 65)
    
    # Calculate improvement percentage
    if sp_mean > 0:
        improvement = (sp_mean - adp_mean) / sp_mean * 100
        if improvement > 0:
            print(f"ADP outperforms SP by {improvement:.2f}%")
        else:
            print(f"SP outperforms ADP by {-improvement:.2f}%")


def analyze_decision_patterns(results: Dict[str, Any], experiment_index: int = 0):
    """
    Analyze the decision patterns of the policy over time.
    
    Args:
        results: Results dictionary from policy evaluation
        experiment_index: Index of the experiment to analyze
    """
    # Time periods
    num_periods = len(results['hydrogen_storage'][experiment_index]) - 1
    periods = np.arange(num_periods)
    
    # Extract decision variables
    electrolyzer_status = results['electrolyzer_status'][experiment_index, :-1]
    p_grid = results['p_grid'][experiment_index]
    p_p2h = results['p_p2h'][experiment_index]
    p_h2p = results['p_h2p'][experiment_index]
    
    # Extract exogenous variables
    wind = results['wind_trajectories'][experiment_index]
    price = results['price_trajectories'][experiment_index]
    
    # Calculate correlations
    corr_price_grid = np.corrcoef(price, p_grid)[0, 1]
    corr_wind_grid = np.corrcoef(wind, p_grid)[0, 1]
    corr_price_p2h = np.corrcoef(price, p_p2h)[0, 1]
    corr_wind_p2h = np.corrcoef(wind, p_p2h)[0, 1]
    
    # Print correlations
    print("\n--- Decision Pattern Analysis ---")
    print(f"Correlation between price and grid power: {corr_price_grid:.4f}")
    print(f"Correlation between wind and grid power: {corr_wind_grid:.4f}")
    print(f"Correlation between price and P2H: {corr_price_p2h:.4f}")
    print(f"Correlation between wind and P2H: {corr_wind_p2h:.4f}")
    
    # Plot the relationship between price, wind and decisions
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Price vs decisions
    ax1 = axes[0]
    ax1.scatter(price, p_grid, alpha=0.7, c='blue', label='Grid Power')
    ax1.scatter(price, p_p2h, alpha=0.7, c='green', label='P2H')
    ax1.scatter(price, p_h2p, alpha=0.7, c='red', label='H2P')
    ax1.set_xlabel('Price')
    ax1.set_ylabel('Power')
    ax1.set_title('Decisions vs Price')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Wind vs decisions
    ax2 = axes[1]
    ax2.scatter(wind, p_grid, alpha=0.7, c='blue', label='Grid Power')
    ax2.scatter(wind, p_p2h, alpha=0.7, c='green', label='P2H')
    ax2.scatter(wind, p_h2p, alpha=0.7, c='red', label='H2P')
    ax2.set_xlabel('Wind Power')
    ax2.set_ylabel('Power')
    ax2.set_title('Decisions vs Wind')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('decision_pattern_analysis.png')
    plt.show()


def evaluate_theta_performance(data: Dict[str, Any], theta: np.ndarray) -> Dict[str, Any]:
    """
    Evaluate the performance of a given theta parameter.
    
    Args:
        data: Dictionary containing problem parameters
        theta: Parameter vector for value function approximation
        
    Returns:
        Dict: Performance metrics
    """
    from adp_implementation import predict_value
    
    # Generate test states
    num_test_states = 1000
    test_states = []
    for _ in range(num_test_states):
        price = np.random.uniform(data['price_floor'], data['price_cap'])
        wind = np.random.uniform(0, 10)
        hydrogen = np.random.uniform(0, data['hydrogen_capacity'])
        status = np.random.choice([0, 1])
        test_states.append((price, wind, hydrogen, status))
    
    # Evaluate value function on test states
    values = [predict_value(state, theta) for state in test_states]
    
    # Calculate statistics
    mean_value = np.mean(values)
    std_value = np.std(values)
    min_value = np.min(values)
    max_value = np.max(values)
    
    # Return metrics
    return {
        'mean_value': mean_value,
        'std_value': std_value,
        'min_value': min_value,
        'max_value': max_value,
        'num_test_states': num_test_states
    }