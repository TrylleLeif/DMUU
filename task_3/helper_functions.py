
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

def sample_representative_state_pairs(num_experiments, data):

    trajectories = []
    
    for e in range(num_experiments):
        trajectory = []
        
        # Initialize starting values
        prev_wind = 4
        curr_wind = 5
        prev_price = 28
        curr_price = 30
        
        # Store initial values as state pairs with random hydrogen and status
        for t in range(2):
            wind_val = 4 if t == 0 else 5
            price_val = 28 if t == 0 else 30
            # Sample endogenous state
            h_storage = round(float(np.random.uniform(0, data['hydrogen_capacity'])), 2)
            elzr_status = int(np.random.choice([0, 1]))
            
            # Create state pair
            z_t = (wind_val, price_val)
            y_t = (h_storage, elzr_status)
            trajectory.append((z_t, y_t))
        
        # Generate trajectory for remaining time slots
        for t in range(2, data['num_timeslots']):
            # Get next wind and price values
            next_wind = float(wind_model(curr_wind, prev_wind, data))
            next_price = float(price_model(curr_price, prev_price, next_wind, data))
            
            # Round to 2 decimal places for cleaner output
            next_wind = round(next_wind, 2)
            next_price = round(next_price, 2)
            
            # Sample endogenous state
            h_storage = round(float(np.random.uniform(0, data['hydrogen_capacity'])), 2)
            elzr_status = int(np.random.choice([0, 1]))
            
            # Create state pair
            z_t = (next_wind, next_price)
            y_t = (h_storage, elzr_status)
            trajectory.append((z_t, y_t))
            
            # Update values for next iteration
            prev_wind = curr_wind
            curr_wind = next_wind
            prev_price = curr_price
            curr_price = next_price
        
        trajectories.append(trajectory)

    return trajectories

def sample_K_next_exogenous_states(z_t, K, data):

    curr_wind, curr_price = z_t
    prev_wind = curr_wind + np.random.normal()
    prev_price = curr_price + np.random.normal()
    
    z_plus_one_samples = []
    
    for _ in range(K):
        # Calculate next wind and price
        next_wind = float(wind_model(curr_wind, prev_wind, data))
        next_price = float(price_model(curr_price, prev_price, next_wind, data))
        
        # Round to 2 decimal places for cleaner output
        next_wind = round(next_wind, 2)
        next_price = round(next_price, 2)
        
        # Sample random demand from demand schedule
        t_mod = np.random.randint(0, len(data["demand_schedule"]))
        next_demand = round(data["demand_schedule"][t_mod], 2)
        
        z_plus_one_samples.append((next_wind, next_price, next_demand))
        
        # Update values for next iteration
        prev_wind = curr_wind
        curr_wind = next_wind
        prev_price = curr_price
        curr_price = next_price
        
    return z_plus_one_samples

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
        #print(f"Power balance constraint violated: {wind_power + p_grid + r_h2p * p_h2p - p_p2h} < {demand}")
        return False
    
    # Check power-to-hydrogen limit: p_p2h <= P2H * x
    if p_p2h > p2h_max_rate * electrolyzer_status:
        #print(f"P2H limit constraint violated: {p_p2h} > {p2h_max_rate * electrolyzer_status}")
        return False
    
    # Check hydrogen-to-power limit: p_h2p <= H2P
    if p_h2p > h2p_max_rate :
        #print(f"H2P limit constraint violated: {p_h2p} > {h2p_max_rate}")
        return False
    
    # Check hydrogen level doesn't exceed capacity
    if hydrogen_level > hydrogen_capacity:
        #print(f"Hydrogen capacity constraint violated: {hydrogen_level} > {hydrogen_capacity}")
        return False
    
    # Check if there's enough hydrogen for conversion to power
    if p_h2p > hydrogen_level:
        #print(f"Hydrogen availability constraint violated: {p_h2p} > {hydrogen_level}")
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

def compute_target_value(state, theta, t, demand, data, K=20):

    price, wind, hydrogen, status = state
    
    # Define possible values for decision variables
    grid_powers = [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5,6,6.5,7,7.5,8]
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
        int(status),    # Electrolyzer status
        price * hydrogen, # Interaction term, price and hydrogen
        wind * hydrogen,  # Interaction term, wind and hydrogen
        price * status, # Interaction term, price and status         
        wind * status,  # Interaction term, wind and status
        hydrogen * status   # Interaction term, hydrogen and status
    ]
    
    # Linear value function approximation
    return np.dot(features, theta)

def extract_features(state): # CHECK    
    """Extract features for the value function approximation"""
    price, wind, hydrogen, status = state
    return np.array([
        1.0,              # Constant term
        price,            # Price
        wind,             # Wind
        hydrogen,         # Hydrogen storage
        int(status),   # Electrolyzer status
        price * hydrogen, # Interaction term, price and hydrogen
        wind * hydrogen,  # Interaction term, wind and hydrogen
        price * status, # Interaction term, price and status         
        wind * status,  # Interaction term, wind and status
        hydrogen * status   # Interaction term, hydrogen and status
    ])

#####


def update_theta_parameters(theta, state, V_target, num_candidates=20):

    """
    Step 1.3: Update the theta parameters using random search to minimize
    the squared error between predicted value and target value.
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

def calculate_next_hydrogen(hydrogen, h2p, p2h, data): # CHECK
    """Calculate the next hydrogen level"""
    h_used = h2p
    h_produced = p2h * data['conversion_p2h']
    next_hydrogen = hydrogen - h_used + h_produced
    next_hydrogen = max(0, min(next_hydrogen, data['hydrogen_capacity']))
    return next_hydrogen