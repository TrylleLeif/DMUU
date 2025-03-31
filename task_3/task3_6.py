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
    sample_representative_states,
    add_critical_region_samples)

# Global variable to store trained parameters
TRAINED_THETA = None
K_val = None 

def extract_features(state): # CHECK    
    """Extract features for the value function approximation"""
    price, wind, hydrogen, status = state
    return np.array([
        1.0,              # Constant term
        price,            # Price
        wind,             # Wind
        hydrogen,         # Hydrogen storage
        float(status),    # Electrolyzer status
        #hydrogen * price, # Interaction term - helps with hydrogen valuation
        #wind * price,     # Wind-price interaction - helps with grid vs. wind decisions
    ])

def predict_value(state, theta): # CHECK
    features = extract_features(state)
    return np.dot(theta, features)

def calculate_immediate_cost(price, wind, hydrogen, grid_power, h2p, p2h, elec_status, data, time_period): # CHECK
    t_mod = time_period % len(data['demand_schedule'])
    demand = data['demand_schedule'][t_mod]
    
    # Power balance check - this should be guaranteed by the feasibility check
    power_supply = wind + grid_power + h2p * data['conversion_h2p']
    power_demand = demand + p2h
    
    # Calculate costs
    grid_cost = price * grid_power
    electrolyzer_cost = data['electrolyzer_cost'] * elec_status
    
    # Add a small penalty for having electrolyzer on but not using it
    # idle_penalty = 0.05 * data['electrolyzer_cost'] * (elec_status * (p2h == 0)) # Removed for simplicity
    
    return grid_cost + electrolyzer_cost # + idle_penalty

def calculate_next_hydrogen(hydrogen, h2p, p2h, data): # CHECK
    """Calculate the next hydrogen level"""
    h_used = h2p
    h_produced = p2h * data['conversion_p2h']
    next_hydrogen = hydrogen - h_used + h_produced
    next_hydrogen = max(0, min(next_hydrogen, data['hydrogen_capacity']))
    return next_hydrogen

def train_vfa():
    """Train the Value Function Approximation"""
    data = get_fixed_data()
    num_timeslots = data['num_timeslots']
    
    # Initialize theta with feature dimensionality
    theta = np.zeros(4) # ADJUST TO MATCH FEATURE DIMENSIONALITY
    
    print("Starting VFA training...")
    
    # Backward Value Function Approximation
    for t in range(num_timeslots-2, -1, -1):
        if t % 5 == 0:
            print(f"Training time period {t}")
        
        # Sample representative state pairs
        states = sample_representative_states(data, num_samples=40)
        states = add_critical_region_samples(states, data, num_extra=20)
        
        # For each sample
        for state in states:
            price, wind, hydrogen, status = state
            
            # Get demand for this time period
            t_mod = t % len(data['demand_schedule'])
            demand = data['demand_schedule'][t_mod]
            
            # Generate all possible combinations of decision variables
            grid_powers = [0, 2, 4, 6, 8]  # More grid power options
            h2p_values = [0, 1, 2] if hydrogen >= 2 else ([0, 1] if hydrogen >= 1 else [0])
            p2h_values = [0, 2, 4] if status == 1 else [0]
            
            # If electrolyzer is off, consider turning it on
            if status == 0:
                elec_options = [(1, 0)]  # Turn on
            else:
                elec_options = [(0, 1)]  # Turn off
            
            # Also consider no change
            elec_options.append((0, 0))
            
            target_values = []
            
            # Go through all combinations
            for grid_power in grid_powers:
                for h2p in h2p_values:
                    for p2h in p2h_values:
                        for elec_on, elec_off in elec_options:
                            # Skip invalid combinations (e.g., turning electrolyzer on when it's already on)
                            if (elec_on == 1 and status == 1) or (elec_off == 1 and status == 0):
                                continue
                            
                            # Skip combinations where p2h > 0 but electrolyzer would be off
                            next_status = status + elec_on - elec_off
                            if p2h > 0 and next_status == 0:
                                continue
                            
                            # Check feasibility with new comprehensive function
                            feasible, _ = check_feasibility(
                                status, elec_on, elec_off, grid_power, p2h, h2p,
                                hydrogen, wind, demand, data
                            )
                            
                            if not feasible:
                                continue
                            
                            # Calculate reward (negative cost)
                            immediate_cost = calculate_immediate_cost(
                                price, wind, hydrogen, grid_power, h2p, p2h, next_status, data, t
                            )
                            reward = -immediate_cost
                            
                            # If last time period, target value is just the reward
                            if t == num_timeslots - 1:
                                target_values.append(reward)
                            else:

                                # Sample K next exogenous states
                                value_functions = []
                                prev_wind = wind  # Start with current wind
                                prev_price = price  # Start with current price
                                next_status = status + elec_on - elec_off  # Define the next electrolyzer status

                                for _ in range(K_val):
                                    next_wind = wind_model(wind, prev_wind, data)
                                    next_price = price_model(price, prev_price, next_wind, data)
                                    next_hydrogen = calculate_next_hydrogen(hydrogen, h2p, p2h, data)
                                    next_state = (next_price, next_wind, next_hydrogen, next_status)
                                    value = predict_value(next_state, theta)
                                    value_functions.append(value)
                                    
                                    # Update previous values for next sample
                                    prev_wind, prev_price = next_wind, next_price

                                # Target value = reward + (gamma/K) * sum(value functions)
                                discount_factor = 0.95
                                target = reward + (discount_factor / K_val) * sum(value_functions)
                                target_values.append(target)
            
            # Calculate V_target as max of target values
            if target_values:
                V_target = max(target_values)
                
                # Update theta
                # Generate candidate theta vectors
                thetas = []
                errors = []
                
                for _ in range(20):  # 20 candidate thetas
                    theta_var = theta + np.random.normal(0, 0.1, size=len(theta))
                    thetas.append(theta_var)
                    
                    # Calculate V_tilde (predicted value)
                    state_features = extract_features(state)
                    V_tilde = np.dot(theta_var, state_features)
                    
                    # Calculate error
                    error = (V_tilde - V_target) ** 2
                    errors.append(error)
                
                # Update theta to the one with minimum error
                theta = thetas[np.argmin(errors)]
    
    print("VFA training completed.")
    print(f"Final theta: {theta}")
    
    return theta

def make_decision(t, electrolyzer_status, hydrogen_level, wind_power, grid_price, demand, data, theta):
    """Make a decision using the trained VFA"""
    # Current state
    state = (grid_price, wind_power, hydrogen_level, electrolyzer_status)
    
    # Generate decision options
    grid_powers = [0, 2, 4, 6, 8]  # More grid power options
    h2p_values = [0, 1, 2] if hydrogen_level >= 2 else ([0, 1] if hydrogen_level >= 1 else [0])
    p2h_values = [0, 2, 4] if electrolyzer_status == 1 else [0]
    
    # Define electrolyzer options
    # If electrolyzer is off, consider turning it on
    if electrolyzer_status == 0:
        elec_options = [(1, 0)]  # Turn on
    else:
        elec_options = [(0, 1)]  # Turn off
    
    # Also consider no change
    elec_options.append((0, 0))
    
    # Evaluate possible decisions
    best_value = float('inf')
    best_grid = None
    best_h2p = None
    best_p2h = None
    best_elec_on = None
    best_elec_off = None
    
    # Sample next exogenous states
    next_states = []
    prev_wind = wind_power  # Start with current wind
    prev_price = grid_price  # Start with current price

    for _ in range(8):  # Set to 8 now for better expectation
        next_wind = wind_model(wind_power, prev_wind, data)
        next_price = price_model(grid_price, prev_price, next_wind, data)
        next_states.append((next_price, next_wind))
    
    # Update previous values for next sample
    prev_wind, prev_price = next_wind, next_price
    
    # For each possible decision
    for grid_power in grid_powers:
        for h2p in h2p_values:
            for p2h in p2h_values:
                for elec_on, elec_off in elec_options:
                    # Skip invalid combinations
                    if (elec_on == 1 and electrolyzer_status == 1) or (elec_off == 1 and electrolyzer_status == 0):
                        continue
                    
                    # Calculate next electrolyzer status
                    next_status = electrolyzer_status + elec_on - elec_off
                    
                    # Skip combinations where p2h > 0 but electrolyzer would be off
                    if p2h > 0 and next_status == 0:
                        continue
                    
                    # Check feasibility with new comprehensive function
                    feasible, _ = check_feasibility(
                        electrolyzer_status, elec_on, elec_off, grid_power, p2h, h2p,
                        hydrogen_level, wind_power, demand, data
                    )
                    
                    if not feasible:
                        continue
                    
                    # Calculate immediate cost
                    immediate_cost = calculate_immediate_cost(
                        grid_price, wind_power, hydrogen_level, grid_power, h2p, p2h, next_status, data, t
                    )
                    
                    # Calculate expected future cost
               # Calculate expected future cost
                    future_cost = 0.0
                    next_hydrogen = calculate_next_hydrogen(hydrogen_level, h2p, p2h, data)
                    next_status = electrolyzer_status + elec_on - elec_off

                    prev_wind = wind_power
                    prev_price = grid_price

                    for _ in range(K_val):
                        next_wind = wind_model(wind_power, prev_wind, data)
                        next_price = price_model(grid_price, prev_price, next_wind, data)
                        next_state = (next_price, next_wind, next_hydrogen, next_status)
                        future_cost += predict_value(next_state, theta)
                        
                        prev_wind, prev_price = next_wind, next_price
                    
                    future_cost /= len(next_states)
                    
                    # Total cost
                    discount_factor = 0.95
                    total_cost = immediate_cost + discount_factor * future_cost
                    
                    # Update best decision if better
                    if total_cost < best_value:
                        best_value = total_cost
                        best_grid = grid_power
                        best_h2p = h2p
                        best_p2h = p2h
                        best_elec_on = elec_on
                        best_elec_off = elec_off
    
    # Check if we found a feasible solution
    if best_grid is None:
        # Use fallback strategy - meet demand with grid power, no hydrogen operations
        best_grid = max(0, demand - wind_power)
        best_h2p = 0
        best_p2h = 0
        
        # Only change electrolyzer state if necessary for feasibility
        if electrolyzer_status == 0:
            best_elec_on = 0
            best_elec_off = 0
        else:
            # If electrolyzer is on but not used, consider turning it off
            best_elec_on = 0
            best_elec_off = 1
            
            # Verify this change is feasible
            feasible, _ = check_feasibility(
                electrolyzer_status, best_elec_on, best_elec_off, best_grid, best_p2h, best_h2p,
                hydrogen_level, wind_power, demand, data
            )
            
            if not feasible:
                # If turning off isn't feasible, keep it on
                best_elec_off = 0
    
    # Add a special case: if electrolyzer is on but not being used for p2h, consider turning it off
    if best_elec_on == 0 and best_elec_off == 0 and electrolyzer_status == 1 and best_p2h == 0:
        # Calculate cost of turning off
        off_cost = calculate_immediate_cost(
            grid_price, wind_power, hydrogen_level, best_grid, best_h2p, 0, 0, data, t
        )
        
        # Calculate cost of keeping on
        on_cost = calculate_immediate_cost(
            grid_price, wind_power, hydrogen_level, best_grid, best_h2p, 0, 1, data, t
        )
        
        # If it's cheaper to turn off and it's feasible, do it
        if off_cost < on_cost:
            test_feasible, _ = check_feasibility(
                electrolyzer_status, 0, 1, best_grid, 0, best_h2p,
                hydrogen_level, wind_power, demand, data
            )
            
            if test_feasible:
                best_elec_off = 1
                best_p2h = 0  # Ensure p2h is 0 when turning off
    
    return best_elec_on, best_elec_off, best_grid, best_p2h, best_h2p

def adp_policy(t, electrolyzer_status, hydrogen_level, wind_power, grid_price, demand, data):
    """Main policy function - provides decisions for energy hub management"""
    global TRAINED_THETA
    global K_val
    K_val = 20  # Number of samples for future cost estimation
    # Train theta if not already trained
    if TRAINED_THETA is None:
        TRAINED_THETA = train_vfa()
    
    # Make decision using trained theta
    elec_on, elec_off, grid, p2h, h2p = make_decision(
        t, electrolyzer_status, hydrogen_level, wind_power, 
        grid_price, demand, data, TRAINED_THETA
    )
    
    # Final feasibility check before returning
    feasible, message = check_feasibility(
        electrolyzer_status, elec_on, elec_off, grid, p2h, h2p,
        hydrogen_level, wind_power, demand, data
    )
    
    if not feasible:
        # Fallback to a simple feasible solution - grid power to meet demand
        grid = max(0, demand - wind_power)
        h2p = 0
        p2h = 0
        
        # Don't change electrolyzer state in the fallback
        elec_on = 0
        elec_off = 0
        
        # If electrolyzer is on but not being used, try to turn it off
        if electrolyzer_status == 1 and p2h == 0:
            test_feasible, _ = check_feasibility(
                electrolyzer_status, 0, 1, grid, 0, 0,
                hydrogen_level, wind_power, demand, data
            )
            
            if test_feasible:
                elec_off = 1
    
    return elec_on, elec_off, grid, p2h, h2p

def create_adp_policy():
    """
    Creates and returns the ADP policy function.
    Returns the policy function directly with no parameters required.
    """
    return adp_policy