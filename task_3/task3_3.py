import numpy as np
import time
from typing import Dict, Any, Tuple, List
import os, sys, random

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import required modules - assuming these are available in your project
from utils.data import get_fixed_data
from utils.WindProcess import wind_model
from utils.PriceProcess import price_model


def extract_features(state: Tuple) -> np.ndarray:   # CHECK

    price, wind, hydrogen, status = state

    return np.array([
        1.0,              # Constant term (bias)
        price,            # Price 
        wind,             # Wind 
        hydrogen,         # Hydrogen storage 
        float(status),    # Electrolyzer status 
        hydrogen * price  # Interaction term
    ])


def predict_value(state: Tuple, theta: np.ndarray) -> float:   # CHECK

    features = extract_features(state)

    return np.dot(theta, features)


def calculate_immediate_cost(                   # CHECK           
                        price: float, 
                        wind: float, 
                        hydrogen: float, 
                        grid_power: float, 
                        h2p: float, 
                        p2h: float, 
                        elec_status: int, 
                        data: Dict[str, Any], 
                        time_period: int
                            ) -> float:

    # Get demand for current time period (cycling through demand schedule)
    t_mod = time_period % len(data['demand_schedule'])
    demand = data['demand_schedule'][t_mod]
    
    # Constraints check
    power_supply = wind + grid_power + h2p * data['conversion_h2p']
    power_demand = demand + p2h
    
    # Penalty for infeasible solution
    if power_supply < power_demand:
        return 1000  # Large penalty for infeasible solution
    
    # Calculate costs
    grid_cost = price * grid_power
    electrolyzer_cost = data['electrolyzer_cost'] * elec_status
    
    return grid_cost + electrolyzer_cost

def calculate_next_hydrogen(
                        hydrogen: float, 
                        h2p: float, 
                        p2h: float, 
                        data: Dict[str, Any]
                            ) -> float:

    # Hydrogen used
    h_used = h2p
    
    # Hydrogen produced
    h_produced = p2h * data['conversion_p2h']
    
    # Next hydrogen level
    next_hydrogen = hydrogen - h_used + h_produced
    
    # Apply storage limits
    next_hydrogen = max(0, min(next_hydrogen, data['hydrogen_capacity']))
    
    return next_hydrogen


def train_vfa(
        data: Dict[str, Any], 
        num_state_samples: int = 5, 
        num_next_states: int = 3, 
        discount_factor: float = 0.95, 
        num_theta_candidates: int = 6,
        verbose: bool = True
            ) -> np.ndarray:

    num_timeslots = data['num_timeslots']
    
    # Init theta as nothing
    theta = np.zeros(6)
    
    if verbose:
        print("Starting VFA training...")
        print(f"Number of time periods: {num_timeslots}")
        print(f"Number of state samples per period: {num_state_samples}")
        print(f"Number of next state samples: {num_next_states}")
    
    # Step 1: Backward Value Function Approximation
    # For t = T, T-1, ..., τ:
    for t in range(num_timeslots-2, -1, -1):
        if verbose:
            print(f"Time period {t}")
        
        # Step 1: Sample representative state pairs
        states = []
        for _ in range(num_state_samples):
            # Sample z_t (wind and price)
            wind = np.random.uniform(0, 10)
            price = np.random.uniform(data['price_floor'], data['price_cap'])
            
            # Sample y_t (hydrogen storage and electrolyzer status)
            hydrogen = np.random.uniform(0, data['hydrogen_capacity'])
            status = np.random.choice([0, 1])
            
            # Store the state
            states.append((price, wind, hydrogen, status))
        
        # Step 2: For each sample
        for i, state in enumerate(states):
            price, wind, hydrogen, status = state
            
            if verbose and i == 0:
                print(f"  Processing sampled state: {state}")
            
            # Generate all possible combinations of decision variables
            grid_powers = [0, 2, 4]  # power from grid
            h2p_values = [0, 1, 2] if hydrogen > 0 else [0]  # hydrogen to power
            p2h_values = [0, 2, 4] if status == 1 else [0]  # power to hydrogen
            elec_statuses = [0, 1]  # electrolyzer status
            
            target_values = []
            
            # Go through all combinations
            for grid_power in grid_powers:
                for h2p in h2p_values:
                    for p2h in p2h_values:
                        for elec in elec_statuses:
                            # Skip invalid combinations
                            if p2h > 0 and elec == 0:
                                continue  # Can't convert power to hydrogen if electrolyzer is off
                            
                            # Calculate reward (negative cost)
                            immediate_cost = calculate_immediate_cost(
                                price, wind, hydrogen, grid_power, h2p, p2h, elec, data, t
                            )
                            reward = -immediate_cost
                            
                            # Skip infeasible actions
                            if immediate_cost >= 1000:
                                continue
                            
                            # If last time period, target value is just the reward
                            if t == num_timeslots - 1:
                                target_values.append(reward)
                            else:
                                # Sample K next exogenous states
                                value_functions = []
                                
                                for _ in range(num_next_states):
                                    # Sample z_{t+1} (next wind and price)
                                    next_wind = wind_model(wind, wind, data)
                                    next_price = price_model(price, price, next_wind, data)
                                    
                                    # Calculate next hydrogen level based on current decisions
                                    next_hydrogen = calculate_next_hydrogen(hydrogen, h2p, p2h, data)
                                    
                                    # Next state
                                    next_state = (next_price, next_wind, next_hydrogen, elec)
                                    
                                    # Calculate value function
                                    value = predict_value(next_state, theta)
                                    value_functions.append(value)
                                
                                # Target value = reward + (gamma/K) * sum(value functions)
                                target = reward + (discount_factor / num_next_states) * sum(value_functions)
                                target_values.append(target)
            
            # Calculate V_target as max of target values
            if target_values:
                V_target = max(target_values)
                
                # Step 3: Update theta
                # Generate variations of theta
                thetas = []
                errors = []
                
                for _ in range(num_theta_candidates):
                    theta_var = theta + np.random.normal(0, 0.1, size=6)
                    thetas.append(theta_var)
                    
                    # Calculate V_tilde
                    state_features = extract_features(state)
                    V_tilde = np.dot(theta_var, state_features)
                    
                    # Calculate error
                    error = (V_tilde - V_target) ** 2
                    errors.append(error)
                
                # Update theta to the one with minimum error
                theta = thetas[np.argmin(errors)]
                
                if verbose and i == 0:
                    print(f"  Updated theta: {theta}")
    
    if verbose:
        print("VFA training completed.")
        print(f"Final theta: {theta}")
    
    return theta


def adp_policy(
            t: int, 
            electrolyzer_status: int, 
            hydrogen_level: float, 
            wind_power: float, 
            grid_price: float, 
            demand: float, 
            data: Dict[str, Any], 
            theta: np.ndarray = None,
            num_next_samples: int = 3,
            discount_factor: float = 0.95
            ) -> Tuple[int, int, float, float, float]:
   
    # If theta is not provided, train it (this should be avoided in practice)
    if theta is None:
        theta = train_vfa(data, verbose=False)
    
    # Current state
    state = (grid_price, wind_power, hydrogen_level, electrolyzer_status)
    
    # Generate decision options
    grid_powers = [0, 2, 4]
    h2p_values = [0, 1, 2] if hydrogen_level > 0 else [0]
    p2h_values = [0, 2, 4] if electrolyzer_status == 1 else [0]
    
    # Evaluate possible decisions for electrolyzer
    best_value = float('inf')
    best_grid = 0
    best_h2p = 0
    best_p2h = 0
    best_elec_on = 0
    best_elec_off = 0
    
    # Sample next exogenous states
    next_states = []
    for _ in range(num_next_samples):
        next_wind = wind_model(wind_power, wind_power, data)
        next_price = price_model(grid_price, grid_price, next_wind, data)
        next_states.append((next_price, next_wind))
    
    # For each possible combination of decisions
    for grid_power in grid_powers:
        for h2p in h2p_values:
            for p2h in p2h_values:
                # Try keeping electrolyzer status the same
                elec = electrolyzer_status
                elec_on = 0
                elec_off = 0
                
                # Skip invalid combinations
                if p2h > 0 and elec == 0:
                    continue  # Can't convert power to hydrogen if electrolyzer is off
                
                # Calculate immediate cost
                immediate_cost = calculate_immediate_cost(
                    grid_price, wind_power, hydrogen_level, grid_power, h2p, p2h, elec, data, t
                )
                
                # Skip infeasible solutions
                if immediate_cost >= 1000:
                    continue
                
                # Calculate expected future cost
                future_cost = 0.0
                next_hydrogen = calculate_next_hydrogen(hydrogen_level, h2p, p2h, data)
                
                for next_price, next_wind in next_states:
                    next_state = (next_price, next_wind, next_hydrogen, elec)
                    future_cost += predict_value(next_state, theta)
                
                future_cost /= len(next_states)
                
                # Total cost (immediate + discounted future)
                total_cost = immediate_cost + discount_factor * future_cost
                
                # Update best decision if better
                if total_cost < best_value:
                    best_value = total_cost
                    best_grid = grid_power
                    best_h2p = h2p
                    best_p2h = p2h
                    best_elec_on = 0
                    best_elec_off = 0
                
                # Try changing electrolyzer status (turn on)
                if electrolyzer_status == 0:
                    elec = 1
                    elec_on = 1
                    elec_off = 0
                    
                    # Calculate immediate cost with new status
                    immediate_cost = calculate_immediate_cost(
                        grid_price, wind_power, hydrogen_level, grid_power, h2p, p2h, elec, data, t
                    )
                    
                    # Skip infeasible solutions
                    if immediate_cost >= 1000:
                        continue
                    
                    # Calculate expected future cost
                    future_cost = 0.0
                    next_hydrogen = calculate_next_hydrogen(hydrogen_level, h2p, p2h, data)
                    
                    for next_price, next_wind in next_states:
                        next_state = (next_price, next_wind, next_hydrogen, elec)
                        future_cost += predict_value(next_state, theta)
                    
                    future_cost /= len(next_states)
                    
                    # Total cost
                    total_cost = immediate_cost + discount_factor * future_cost
                    
                    # Update best decision if better
                    if total_cost < best_value:
                        best_value = total_cost
                        best_grid = grid_power
                        best_h2p = h2p
                        best_p2h = p2h
                        best_elec_on = 1
                        best_elec_off = 0
                
                # Try changing electrolyzer status (turn off)
                if electrolyzer_status == 1:
                    elec = 0
                    elec_on = 0
                    elec_off = 1
                    
                    # With electrolyzer off, can't use p2h
                    if p2h > 0:
                        continue
                    
                    # Calculate immediate cost with new status
                    immediate_cost = calculate_immediate_cost(
                        grid_price, wind_power, hydrogen_level, grid_power, h2p, 0, elec, data, t
                    )
                    
                    # Skip infeasible solutions
                    if immediate_cost >= 1000:
                        continue
                    
                    # Calculate expected future cost
                    future_cost = 0.0
                    next_hydrogen = calculate_next_hydrogen(hydrogen_level, h2p, 0, data)
                    
                    for next_price, next_wind in next_states:
                        next_state = (next_price, next_wind, next_hydrogen, elec)
                        future_cost += predict_value(next_state, theta)
                    
                    future_cost /= len(next_states)
                    
                    # Total cost
                    total_cost = immediate_cost + discount_factor * future_cost
                    
                    # Update best decision if better
                    if total_cost < best_value:
                        best_value = total_cost
                        best_grid = grid_power
                        best_h2p = h2p
                        best_p2h = 0  # Reset to 0 since electrolyzer is off
                        best_elec_on = 0
                        best_elec_off = 1
    
    return best_elec_on, best_elec_off, best_grid, best_p2h, best_h2p


def create_adp_policy(theta: np.ndarray):
    """
    Create a policy function that uses the trained theta parameter.
    
    Args:
        theta: Trained parameter vector for value function approximation
        
    Returns:
        function: Policy function compatible with the evaluation framework
    """
    def policy_function(t, electrolyzer_status, hydrogen_level, wind_power, 
                        grid_price, demand, data):
        return adp_policy(
            t, electrolyzer_status, hydrogen_level, wind_power, grid_price, 
            demand, data, theta=theta
        )
    
    return policy_function