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

# CHECK
def check_feasibility(electrolyzer_status, electrolyzer_on, electrolyzer_off, p_grid, p_p2h, p_h2p, hydrogen_level, wind_power, demand, data):
 
    tolerance = 1e-9
    
    # Get conversion rates and capacities from data
    r_p2h = data['conversion_p2h']  # Efficiency of converting power to hydrogen
    r_h2p = data['conversion_h2p']  # Efficiency of converting hydrogen to power
    p2h_max_rate = data['p2h_max_rate']  # Maximum P2H rate
    h2p_max_rate = data['h2p_max_rate']  # Maximum H2P rate
    hydrogen_capacity = data['hydrogen_capacity']  # Maximum hydrogen storage
    
    # Calculate next electrolyzer status based on switching decisions
    next_status = electrolyzer_status + electrolyzer_on - electrolyzer_off
    
    # Check power balance: wind + grid + h2p - p2h >= demand
    if wind_power + p_grid + r_h2p * p_h2p - p_p2h < demand - tolerance:
        return False, f"Power balance constraint violated: {wind_power + p_grid + r_h2p * p_h2p - p_p2h} < {demand}"
    
    # Check power-to-hydrogen limit: p_p2h <= P2H * next_status
    if p_p2h > p2h_max_rate * next_status:
        return False, f"P2H limit constraint violated: {p_p2h} > {p2h_max_rate * next_status}"
    
    # Check hydrogen-to-power limit: p_h2p <= H2P
    if p_h2p > h2p_max_rate:
        return False, f"H2P limit constraint violated: {p_h2p} > {h2p_max_rate}"
    
    # Check hydrogen level doesn't exceed capacity
    next_hydrogen = calculate_next_hydrogen(hydrogen_level, p_h2p, p_p2h, data)
    if next_hydrogen > hydrogen_capacity:
        return False, f"Hydrogen capacity constraint violated: {next_hydrogen} > {hydrogen_capacity}"
    
    # Check if there's enough hydrogen for conversion to power
    if p_h2p > hydrogen_level:
        return False, f"Hydrogen availability constraint violated: {p_h2p} > {hydrogen_level}"
    
    # Check that at most one switching action happens
    if electrolyzer_on + electrolyzer_off > 1:
        return False, "Multiple switching actions constraint violated"
    
    # Check that you can only switch ON if it's currently OFF
    if electrolyzer_on == 1 and electrolyzer_status == 1:
        return False, "Invalid ON action: electrolyzer is already ON"
    
    # Check that you can only switch OFF if it's currently ON
    if electrolyzer_off == 1 and electrolyzer_status == 0:
        return False, "Invalid OFF action: electrolyzer is already OFF"
    
    # All constraints satisfied
    return True, ""

def sample_representative_states(data, num_samples=50): # CHECK
    states = []
    
    # Stratify by price levels
    price_ranges = [
        (data['price_floor'], data['mean_price'] * 0.8),                    # Low prices
        (data['mean_price'] * 0.8, data['mean_price'] * 1.2),               # Medium prices
        (data['mean_price'] * 1.2, data['price_cap'])                       # High prices
    ]
    
    # Stratify by wind levels
    wind_ranges = [
        (0, data['target_mean_wind'] * 0.5),                                # Low wind
        (data['target_mean_wind'] * 0.5, data['target_mean_wind'] * 1.5),   # Medium wind
        (data['target_mean_wind'] * 1.5, 10)                                # High wind
    ]
    
    # Stratify by hydrogen storage levels
    hydrogen_ranges = [
        (0, data['hydrogen_capacity'] * 0.3),                               # Low storage
        (data['hydrogen_capacity'] * 0.3, data['hydrogen_capacity'] * 0.7), # Medium storage
        (data['hydrogen_capacity'] * 0.7, data['hydrogen_capacity'])        # High storage
    ]
    
    # Allocate samples across all strata
    samples_per_combination = max(1, num_samples // (len(price_ranges) * len(wind_ranges) * len(hydrogen_ranges) * 2))
    
    for price_range in price_ranges:
        for wind_range in wind_ranges:
            for hydrogen_range in hydrogen_ranges:
                for status in [0, 1]:  # Electrolyzer status
                    for _ in range(samples_per_combination):
                        price = np.random.uniform(price_range[0], price_range[1])
                        wind = np.random.uniform(wind_range[0], wind_range[1])
                        hydrogen = np.random.uniform(hydrogen_range[0], hydrogen_range[1])
                        
                        states.append((price, wind, hydrogen, status))
    
    # If we need more samples to reach the target number, add them randomly
    while len(states) < num_samples:
        price = np.random.uniform(data['price_floor'], data['price_cap'])
        wind = np.random.uniform(0, 10)
        hydrogen = np.random.uniform(0, data['hydrogen_capacity'])
        status = np.random.choice([0, 1])
        states.append((price, wind, hydrogen, status))
    
    return states

def add_critical_region_samples(states, data, num_extra=20): # CHECK
    # Add extra samples for price arbitrage opportunities (low prices with empty storage)
    for _ in range(num_extra // 4):
        price = np.random.uniform(data['price_floor'], data['mean_price'] * 0.6)  # Low price
        wind = np.random.uniform(data['target_mean_wind'] * 0.8, data['target_mean_wind'] * 1.5)  # Medium-high wind
        hydrogen = np.random.uniform(0, data['hydrogen_capacity'] * 0.3)  # Low hydrogen
        status = 1  # Electrolyzer on
        states.append((price, wind, hydrogen, status))
    
    # Add extra samples for hydrogen utilization opportunities (high prices with full storage)
    for _ in range(num_extra // 4):
        price = np.random.uniform(data['mean_price'] * 1.4, data['price_cap'])  # High price
        wind = np.random.uniform(0, data['target_mean_wind'] * 0.7)  # Low wind
        hydrogen = np.random.uniform(data['hydrogen_capacity'] * 0.7, data['hydrogen_capacity'])  # High hydrogen
        status = np.random.choice([0, 1])  # Either status
        states.append((price, wind, hydrogen, status))
    
    # Add extra samples for electrolyzer switching decisions (medium hydrogen, medium price)
    for _ in range(num_extra // 2):
        price = np.random.uniform(data['mean_price'] * 0.9, data['mean_price'] * 1.1)  # Medium price
        wind = np.random.uniform(data['target_mean_wind'] * 0.7, data['target_mean_wind'] * 1.3)  # Medium wind
        hydrogen = np.random.uniform(data['hydrogen_capacity'] * 0.4, data['hydrogen_capacity'] * 0.6)  # Medium hydrogen
        status = np.random.choice([0, 1])  # Either status
        states.append((price, wind, hydrogen, status))
    
    return states


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
    #plt.savefig('policy_decisions.png')
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
    #plt.savefig('feature_importance.png')
    plt.show()
    
    # Print numerical values
    print("Feature Importance:")
    for name, imp in zip(feature_names, normalized_importance):
        print(f"{name}: {imp:.4f}")

def evaluate_theta_performance(data: Dict[str, Any], theta: np.ndarray) -> Dict[str, Any]:
    """
    Evaluate the performance of a given theta parameter.
    
    Args:
        data: Dictionary containing problem parameters
        theta: Parameter vector for value function approximation
        
    Returns:
        Dict: Performance metrics
    """
    from task_3.task3_6 import predict_value
    
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