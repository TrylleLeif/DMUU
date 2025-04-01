import os, sys, random
# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import numpy as np
from utils.data import get_fixed_data
from utils.WindProcess import wind_model
from utils.PriceProcess import price_model
from task_3.helper_functions import (
    check_feasibility,
    sample_representative_state_pairs,
    compute_target_value,
    sample_exogenous_states,
    predict_value,
    extract_features,
    update_theta_parameters,
    calculate_next_hydrogen
)

# Global variable to store trained parameters
TRAINED_THETA = None

# setting a seed for reproducibility
np.random.seed(44)

def train_vfa():
    """Train the Value Function Approximation"""
    data = get_fixed_data()
    num_timeslots = data['num_timeslots']
    
    # Initialize theta with feature dimensionality
    feature_dim = len(predict_value(state, theta))
    theta = np.zeros(feature_dim)
    K_val = 20  # Number of samples for future cost estimation
    
    print("Starting VFA training...")
    
    # Backward Value Function Approximation
    for t in range(num_timeslots-2, -1, -1):
        if t % 5 == 0:
            print(f"Training time period {t}")
        
        # Step 1.1: Sample representative state pairs
        states = sample_representative_state_pairs(data, num_samples=60)
        
        # For each sample
        for state in states:
            price, wind, hydrogen, status = state
            
            # Get demand for this time period
            t_mod = t % len(data['demand_schedule'])
            demand = data['demand_schedule'][t_mod]
            
            # Step 1.2: Compute target value for this state
            V_target = compute_target_value(state, theta, t, demand, data, K=K_val)
            
            # Step 1.3: Update theta parameters
            theta = update_theta_parameters(theta, state, V_target)
    
    print("VFA training completed.")
    print(f"Final theta: {theta}")
    
    return theta

def compute_optimal_action(state, theta, demand, data, next_states, K_val=20):
    """
    Step 2.2: Compute optimal action for the current state.
    """
    price, wind, hydrogen, status = state
    
    # Define decision options
    grid_powers = [0, 2, 4, 6, 8, 10]
    h2p_values = [0, 1, 2] if hydrogen >= 2 else ([0, 1] if hydrogen >= 1 else [0])
    p2h_values = [0, 2, 4] if status == 1 else [0]
    
    # Define electrolyzer options
    if status == 0:
        elec_options = [(0, 0), (1, 0)]  # Stay off or turn on
    else:
        elec_options = [(0, 0), (0, 1)]  # Stay on or turn off
    
    # Track best decision and its value
    best_value = float('inf')  # Start with infinity for minimization
    best_decision = None
    
    # Try all possible decisions
    for grid_power in grid_powers:
        for h2p in h2p_values:
            for p2h in p2h_values:
                for elec_on, elec_off in elec_options:
                    # Calculate next electrolyzer status
                    next_status = status + elec_on - elec_off
                    
                    # Skip invalid combinations
                    if p2h > 0 and next_status == 0:
                        continue
                    
                    # Check feasibility
                    feasible = check_feasibility(
                        electrolyzer_status=status,
                        electrolyzer_on=elec_on,
                        electrolyzer_off=elec_off,
                        p_grid=grid_power,
                        p_p2h=p2h,
                        p_h2p=h2p,
                        hydrogen_level=hydrogen,
                        wind_power=wind,
                        demand=demand,
                        p2h_max_rate=data['p2h_max_rate'],
                        h2p_max_rate=data['h2p_max_rate'],
                        r_p2h=data['conversion_p2h'],
                        r_h2p=data['conversion_h2p'],
                        hydrogen_capacity=data['hydrogen_capacity']
                    )
                    
                    if not feasible:
                        continue
                    
                    # Calculate immediate cost
                    immediate_cost = grid_power * price + data['electrolyzer_cost'] * next_status
                    
                    # Calculate next hydrogen level
                    next_hydrogen = hydrogen - h2p + p2h * data['conversion_p2h']
                    next_hydrogen = max(0, min(next_hydrogen, data['hydrogen_capacity']))
                    
                    # Calculate future cost across K samples
                    future_cost = 0
                    discount_factor = 0.95
                    
                    # Use the pre-sampled next states
                    for next_price, next_wind in next_states:
                        next_state = (next_price, next_wind, next_hydrogen, next_status)
                        # Value function is negative of cost
                        value = predict_value(next_state, theta)
                        future_cost -= value  # Convert value to cost
                    
                    # Average future cost
                    future_cost = future_cost / len(next_states)
                    
                    # Total cost
                    total_cost = immediate_cost + discount_factor * future_cost
                    
                    # Update best decision if better
                    if total_cost < best_value:
                        best_value = total_cost
                        best_decision = (elec_on, elec_off, grid_power, p2h, h2p)
    
    # Return fallback decision if no feasible option found
    if best_decision is None:
        # Meet demand with grid power, no hydrogen operations
        fallback_grid = max(0, demand - wind)
        fallback_decision = (0, 0, fallback_grid, 0, 0)
        return fallback_decision
    
    return best_decision

def adp_policy(t, electrolyzer_status, hydrogen_level, wind_power, grid_price, demand, data):
    """
    Step 2: Main policy function - provides decisions for energy hub management
    
    Args:
        t: Current time period
        electrolyzer_status: Current electrolyzer status (0=off, 1=on)
        hydrogen_level: Current hydrogen storage level
        wind_power: Current wind power generation
        grid_price: Current grid electricity price
        demand: Current power demand
        data: Dictionary containing problem parameters
        
    Returns:
        Tuple of decision variables (elec_on, elec_off, grid, p2h, h2p)
    """
    global TRAINED_THETA
    K_val = 20  # Number of samples for future cost estimation
    
    # Initialize and train theta if not already trained
    if TRAINED_THETA is None:
        TRAINED_THETA = train_vfa()
    
    # Current state
    state = (grid_price, wind_power, hydrogen_level, electrolyzer_status)
    
    # Step 2.1: Sample next exogenous states
    next_states = sample_exogenous_states(grid_price, wind_power, data, K_val)
    
    # Step 2.2: Compute optimal action
    best_decision = compute_optimal_action(state, TRAINED_THETA, demand, data, next_states, K_val)
    
    # Ensure returned decision is feasible
    elec_on, elec_off, grid, p2h, h2p = best_decision
    
    # Final feasibility check
    feasible = check_feasibility(
        electrolyzer_status=electrolyzer_status,
        electrolyzer_on=elec_on,
        electrolyzer_off=elec_off,
        p_grid=grid,
        p_p2h=p2h,
        p_h2p=h2p,
        hydrogen_level=hydrogen_level,
        wind_power=wind_power,
        demand=demand,
        p2h_max_rate=data['p2h_max_rate'],
        h2p_max_rate=data['h2p_max_rate'],
        r_p2h=data['conversion_p2h'],
        r_h2p=data['conversion_h2p'],
        hydrogen_capacity=data['hydrogen_capacity']
    )
    
    if not feasible:
        # Emergency fallback if somehow final decision is not feasible
        grid = max(0, demand - wind_power)
        return 0, 0, grid, 0, 0
    
    return elec_on, elec_off, grid, p2h, h2p

def create_adp_policy():
    """
    Creates and returns the ADP policy function.
    Returns the policy function directly with no parameters required.
    """
    return adp_policy