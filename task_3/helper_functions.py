
import numpy as np
import sys, os, random
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
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

def check_feasibility(
        electrolyzer_status: int,
        electrolyzer_on: int,
        electrolyzer_off: int,
        p_grid: float,
        p_p2h: float,
        p_h2p: float,
        hydrogen_level: float,
        wind_power: float,
        demand: float,
        p2h_max_rate: float,
        h2p_max_rate: float,
        r_p2h: float,
        r_h2p: float,
        hydrogen_capacity: float
    ) -> bool:
    # Check power balance: wind + grid + h2p - p2h >= demand
    tolerance = 1e-9
    if wind_power + p_grid + r_h2p * p_h2p - p_p2h < demand - tolerance:
        print(f"Power balance constraint violated: {wind_power + p_grid + r_h2p * p_h2p - p_p2h} < {demand}")
        return False
    
    # Check power-to-hydrogen limit: p_p2h <= P2H * x
    if p_p2h > p2h_max_rate * electrolyzer_status:
        print(f"P2H limit constraint violated: {p_p2h} > {p2h_max_rate * electrolyzer_status}")
        return False
    
    # Check hydrogen-to-power limit: p_h2p <= H2P
    if p_h2p > h2p_max_rate :
        print(f"H2P limit constraint violated: {p_h2p} > {h2p_max_rate}")
        return False
    
    # Check hydrogen level doesn't exceed capacity
    if hydrogen_level > hydrogen_capacity:
        print(f"Hydrogen capacity constraint violated: {hydrogen_level} > {hydrogen_capacity}")
        return False
    
    # Check if there's enough hydrogen for conversion to power
    if p_h2p > hydrogen_level:
        print(f"Hydrogen availability constraint violated: {p_h2p} > {hydrogen_level}")
        return False
    
    # Check that at most one switching action happens
    if electrolyzer_on + electrolyzer_off > 1:
        return False
        
    # Check that you can only switch ON if it's currently OFF
    if electrolyzer_on == 1 and electrolyzer_status == 1:
        return False
        
    # Check that you can only switch OFF if it's currently ON
    if electrolyzer_off == 1 and electrolyzer_status == 0:
        return False


    # All constraints satisfied
    return True

def sample_representative_state_pairs(data, num_samples=50):
    """
    Step 1.1: Sample representative state pairs (z_t^i, y_t^i) for VFA training.
    
    Args:
        data: Dictionary containing problem parameters
        num_samples: Number of state pairs to generate
        
    Returns:
        List of state pairs, where each state is (price, wind, hydrogen, electrolyzer_status)
    """
    
    state_pairs = []
    
    # Extract data parameters needed for sampling and feasibility checks
    price_floor = data['price_floor']
    price_cap = data['price_cap']
    mean_price = data['mean_price']
    target_mean_wind = data['target_mean_wind']
    hydrogen_capacity = data['hydrogen_capacity']
    p2h_max_rate = data['p2h_max_rate']
    h2p_max_rate = data['h2p_max_rate']
    r_p2h = data['conversion_p2h']
    r_h2p = data['conversion_h2p']
    demand = data['demand_schedule'][0]  # Using first demand value for simplicity
    
    # Counter for attempts to avoid infinite loops
    attempts = 0
    max_attempts = num_samples * 10
    
    while len(state_pairs) < num_samples and attempts < max_attempts:
        attempts += 1
        
        # Sample exogenous variables z_t (price, wind)
        price = random.uniform(price_floor, price_cap)
        wind = random.uniform(0, target_mean_wind * 2)  # 0 to twice the mean
        
        # Sample endogenous variables y_t (hydrogen, electrolyzer_status)
        hydrogen = random.uniform(0, hydrogen_capacity)
        electrolyzer_status = random.choice([0, 1])
        
        # Create the state
        state = (price, wind, hydrogen, electrolyzer_status)
        
        # Check if this state has at least one feasible decision
        # We'll try a few basic decisions to see if any are feasible
        has_feasible_decision = False
        
        # Try some simple decision combinations
        for p_grid in [0, demand/2, demand]:
            for p_h2p in [0, min(1, hydrogen)]:
                for p_p2h in [0] if electrolyzer_status == 0 else [0, 2]:
                    # No switching for simplicity
                    electrolyzer_on, electrolyzer_off = 0, 0
                    
                    # Check if this decision is feasible
                    feasible = check_feasibility(
                        electrolyzer_status=electrolyzer_status,
                        electrolyzer_on=electrolyzer_on,
                        electrolyzer_off=electrolyzer_off,
                        p_grid=p_grid,
                        p_p2h=p_p2h,
                        p_h2p=p_h2p,
                        hydrogen_level=hydrogen,
                        wind_power=wind,
                        demand=demand,
                        p2h_max_rate=p2h_max_rate,
                        h2p_max_rate=h2p_max_rate,
                        r_p2h=r_p2h,
                        r_h2p=r_h2p,
                        hydrogen_capacity=hydrogen_capacity
                    )
                    
                    if feasible:
                        has_feasible_decision = True
                        break
                
                if has_feasible_decision:
                    break
            
            if has_feasible_decision:
                break
        
        # Only add states that have at least one feasible decision
        if has_feasible_decision:
            state_pairs.append(state)
    
    # Add a few critical region samples (20% of total samples)
    critical_samples = min(int(num_samples * 0.2), max(1, num_samples - len(state_pairs)))
    
    for _ in range(critical_samples):
        # Choose a critical region scenario (low price or high price)
        scenario = random.choice(['low_price', 'high_price'])
        
        if scenario == 'low_price':
            # Low price, opportunity to produce hydrogen
            price = random.uniform(price_floor, mean_price * 0.7)
            wind = random.uniform(target_mean_wind * 0.5, target_mean_wind * 2)
            hydrogen = random.uniform(0, hydrogen_capacity * 0.5)
            electrolyzer_status = 1  # On for producing hydrogen
        else:
            # High price, opportunity to use hydrogen
            price = random.uniform(mean_price * 1.3, price_cap)
            wind = random.uniform(0, target_mean_wind)
            hydrogen = random.uniform(hydrogen_capacity * 0.5, hydrogen_capacity)
            electrolyzer_status = random.choice([0, 1])
        
        # Add this critical state if it has feasible decisions (same check as above)
        state = (price, wind, hydrogen, electrolyzer_status)
        state_pairs.append(state)
    
    return state_pairs[:num_samples]  # Return exactly num_samples states

def compute_target_value(state, theta, t, demand, data, K=20):

    price, wind, hydrogen, status = state
    
    # Define possible values for decision variables
    grid_powers = [0,1,2,3,4,5,6,7,8,9,10]  # Grid power options
    h2p_values = [0,1,2,3] if hydrogen >= 2 else ([0, 1] if hydrogen >= 1 else [0])
    p2h_values = [0,1,2,3,4,5] if status == 1 else [0]
    
    # Electrolyzer control options
    if status == 0:
        elec_options = [(0, 0), (1, 0)]  # Stay off or turn on
    else:
        elec_options = [(0, 0), (0, 1)]  # Stay on or turn off
    
    # Track best decision and its value
    best_value = float('-inf')
    
    # Step 1.2.a: Sample K next exogenous states
    next_exogenous_states = sample_exogenous_states(price, wind, data, K)
    
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
                    
                    # Calculate immediate reward (negative cost)
                    immediate_cost = grid_power * price + data['electrolyzer_cost'] * next_status
                    immediate_reward = -immediate_cost
                    
                    # Calculate next hydrogen level
                    next_hydrogen = hydrogen - h2p + p2h * data['conversion_p2h']
                    next_hydrogen = max(0, min(next_hydrogen, data['hydrogen_capacity']))
                    
                    # Calculate future value across K samples
                    future_value = 0
                    for next_price, next_wind in next_exogenous_states:
                        next_state = (next_price, next_wind, next_hydrogen, next_status)
                        state_value = predict_value(next_state, theta)
                        future_value += state_value
                    
                    # Average future value and apply discount
                    future_value = future_value / K
                    discount_factor = 0.95
                    
                    # Total value for this decision
                    total_value = immediate_reward + discount_factor * future_value
                    
                    # Update best value
                    if total_value > best_value:
                        best_value = total_value
    
    # Return the maximum value found (or default if no feasible decision)
    return best_value if best_value > float('-inf') else 0


def sample_exogenous_states(price, wind, data, num_samples):

    samples = []
    prev_wind = wind
    prev_price = price
    
    for _ in range(num_samples):
        # Use transition models to generate next states
        next_wind = wind_model(wind, prev_wind, data)
        next_price = price_model(price, prev_price, next_wind, data)
        samples.append((next_price, next_wind))
        
        # Update previous values for the next sample
        prev_wind, prev_price = next_wind, next_price
    
    return samples


def predict_value(state, theta):

    # Extract features
    price, wind, hydrogen, status = state
    features = [
        1.0,              # Constant term
        price,            # Price
        wind,             # Wind
        hydrogen,         # Hydrogen storage
        float(status),    # Electrolyzer status
    ]
    
    # Linear value function approximation
    return sum(f * t for f, t in zip(features, theta))

def extract_features(state): # CHECK    
    """Extract features for the value function approximation"""
    price, wind, hydrogen, status = state
    return np.array([
        1.0,              # Constant term
        price,            # Price
        wind,             # Wind
        hydrogen,         # Hydrogen storage
        float(status),   # Electrolyzer status
    ])

def update_theta_parameters(theta, state, V_target, num_candidates=20):
    """
    Step 1.3: Update the theta parameters using random search to minimize
    the squared error between predicted value and target value.
    
    Args:
        theta: Current theta parameters
        state: Current state (price, wind, hydrogen, status)
        V_target: Target value computed from Step 1.2
        num_candidates: Number of candidate thetas to generate
        
    Returns:
        Updated theta parameters
    """
    # Generate candidate theta vectors
    thetas = []
    errors = []
    
    for _ in range(num_candidates):
        # Generate a random perturbation of the current theta
        theta_var = theta + np.random.normal(0, 0.1, size=len(theta))
        thetas.append(theta_var)
        
        # Calculate predicted value
        state_features = extract_features(state)
        V_tilde = np.dot(theta_var, state_features)
        
        # Calculate squared error
        error = (V_tilde - V_target) ** 2
        errors.append(error)
    
    # Update theta to the one with minimum error
    best_theta = thetas[np.argmin(errors)]
    
    return best_theta