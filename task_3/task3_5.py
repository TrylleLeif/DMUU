import os, sys, random
# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import numpy as np
from utils.data import get_fixed_data
from utils.WindProcess import wind_model
from utils.PriceProcess import price_model

# Global variable to store trained parameters
TRAINED_THETA = None

def extract_features(state):
    """Extract features for the value function approximation"""
    price, wind, hydrogen, status = state
    return np.array([
        1.0,              # Constant term
        price,            # Price
        wind,             # Wind
        hydrogen,         # Hydrogen storage
        float(status),    # Electrolyzer status
        # hydrogen * price,  # Interaction term
        # min(hydrogen, 2), # Capped hydrogen (relevant for decisions)
        # wind * price,     # Wind-price interaction
        # price * price,    # Quadratic price term
        # hydrogen * hydrogen # Quadratic hydrogen term
    ])

def predict_value(state, theta):
    """Predict the value of a state using the approximation"""
    features = extract_features(state)
    return np.dot(theta, features)

def calculate_immediate_cost(price, wind, hydrogen, grid_power, h2p, p2h, elec_status, data, time_period):
    """Calculate the immediate cost for decisions"""
    t_mod = time_period % len(data['demand_schedule'])
    demand = data['demand_schedule'][t_mod]
    
    # Power balance check
    power_supply = wind + grid_power + h2p * data['conversion_h2p']
    power_demand = demand + p2h
    
    if power_supply < power_demand:
        return 1000  # Penalty for infeasible solution
    
    # Calculate costs
    grid_cost = price * grid_power
    electrolyzer_cost = data['electrolyzer_cost'] * elec_status
    
    return grid_cost + electrolyzer_cost

def calculate_next_hydrogen(hydrogen, h2p, p2h, data):
    """Calculate the next hydrogen level"""
    h_used = h2p
    h_produced = p2h * data['conversion_p2h']
    next_hydrogen = hydrogen - h_used + h_produced
    next_hydrogen = max(0, min(next_hydrogen, data['hydrogen_capacity']))
    return next_hydrogen

def check_constraints(wind, hydrogen, grid_power, h2p, p2h, elec_status, data, time_period):
    """Check if the decision satisfies all constraints"""
    t_mod = time_period % len(data['demand_schedule'])
    demand = data['demand_schedule'][t_mod]
    
    # Power balance constraint
    power_supply = wind + grid_power + h2p * data['conversion_h2p']
    power_demand = demand + p2h
    
    if power_supply < power_demand:
        return False, f"Power balance constraint violated: {power_supply} < {power_demand}"
    
    # Hydrogen availability constraint
    if h2p > hydrogen:
        return False, f"Hydrogen availability constraint violated: {h2p} > {hydrogen}"
    
    # Electrolyzer status constraint
    if p2h > 0 and elec_status == 0:
        return False, f"Electrolyzer status constraint violated: Cannot convert to hydrogen when electrolyzer is off"
    
    # P2H rate constraint
    if p2h > data['p2h_max_rate'] and elec_status == 1:
        return False, f"P2H rate constraint violated: {p2h} > {data['p2h_max_rate']}"
    
    # H2P rate constraint
    if h2p > data['h2p_max_rate']:
        return False, f"H2P rate constraint violated: {h2p} > {data['h2p_max_rate']}"
    
    return True, ""

def train_vfa():
    """Train the Value Function Approximation"""
    data = get_fixed_data()
    num_timeslots = data['num_timeslots']
    
    # Initialize theta with feature dimensionality
    theta = np.zeros(10)
    
    print("Starting VFA training...")
    
    # Backward Value Function Approximation
    for t in range(num_timeslots-2, -1, -1):
        if t % 5 == 0:
            print(f"Training time period {t}")
        
        # Sample representative state pairs
        states = []
        for _ in range(10):  # 10 state samples per time period (increased from 5)
            wind = np.random.uniform(0, 10)
            price = np.random.uniform(data['price_floor'], data['price_cap'])
            hydrogen = np.random.uniform(0, data['hydrogen_capacity'])
            status = np.random.choice([0, 1])
            states.append((price, wind, hydrogen, status))
        
        # For each sample
        for state in states:
            price, wind, hydrogen, status = state
            
            # Generate all possible combinations of decision variables                          # WHAT IS THIS?
            grid_powers = [0, 2, 4]  # Added more grid power options
            h2p_values = [0, 1, 2] if hydrogen >= 2 else ([0, 1] if hydrogen >= 1 else [0])
            p2h_values = [0, 2, 4] if status == 1 else [0]
            elec_statuses = [0, 1]
            
            target_values = []
            
            # Go through all combinations
            for grid_power in grid_powers:
                for h2p in h2p_values:
                    for p2h in p2h_values:
                        for elec in elec_statuses:
                            # Skip invalid combinations
                            if p2h > 0 and elec == 0:
                                continue
                            
                            # Check constraints
                            feasible, _ = check_constraints(wind, hydrogen, grid_power, h2p, p2h, elec, data, t)
                            if not feasible:
                                continue
                            
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
                                
                                for _ in range(5):  # 5 next states for expectation (increased from 3)
                                    next_wind = wind_model(wind, wind, data)
                                    next_price = price_model(price, price, next_wind, data)
                                    next_hydrogen = calculate_next_hydrogen(hydrogen, h2p, p2h, data)
                                    next_state = (next_price, next_wind, next_hydrogen, elec)
                                    value = predict_value(next_state, theta)
                                    value_functions.append(value)
                                
                                # Target value = reward + (gamma/K) * sum(value functions)
                                discount_factor = 0.95
                                target = reward + (discount_factor / 5) * sum(value_functions)
                                target_values.append(target)
            
            # Calculate V_target as max of target values
            if target_values:
                V_target = max(target_values)
                
                # Update theta
                # Generate candidate theta vectors
                thetas = []
                errors = []
                
                for _ in range(20):  # 20 candidate thetas (increased from 10)
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
    grid_powers = [0, 2, 4, 6, 8]  # Added more grid power options
    h2p_values = [0, 1, 2] if hydrogen_level >= 2 else ([0, 1] if hydrogen_level >= 1 else [0])
    p2h_values = [0, 2, 4] if electrolyzer_status == 1 else [0]
    
    # Evaluate possible decisions
    best_value = float('inf')
    best_grid = 0
    best_h2p = 0
    best_p2h = 0
    best_elec_on = 0
    best_elec_off = 0
    
    # Sample next exogenous states
    next_states = []
    for _ in range(5):  # 5 next states (increased from 3)
        next_wind = wind_model(wind_power, wind_power, data)
        next_price = price_model(grid_price, grid_price, next_wind, data)
        next_states.append((next_price, next_wind))
    
    # For each possible decision
    for grid_power in grid_powers:
        for h2p in h2p_values:
            for p2h in p2h_values:
                # Try keeping electrolyzer status the same
                elec = electrolyzer_status
                elec_on = 0
                elec_off = 0
                
                # Skip invalid combinations
                if p2h > 0 and elec == 0:
                    continue
                
                # Check constraints directly
                feasible, _ = check_constraints(wind_power, hydrogen_level, grid_power, h2p, p2h, elec, data, t)
                if not feasible:
                    continue
                
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
                
                # Total cost
                discount_factor = 0.95
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
                    
                    # Check constraints with new status
                    feasible, _ = check_constraints(wind_power, hydrogen_level, grid_power, h2p, p2h, elec, data, t)
                    if not feasible:
                        continue
                    
                    immediate_cost = calculate_immediate_cost(
                        grid_price, wind_power, hydrogen_level, grid_power, h2p, p2h, elec, data, t
                    )
                    
                    if immediate_cost >= 1000:
                        continue
                    
                    future_cost = 0.0
                    next_hydrogen = calculate_next_hydrogen(hydrogen_level, h2p, p2h, data)
                    
                    for next_price, next_wind in next_states:
                        next_state = (next_price, next_wind, next_hydrogen, elec)
                        future_cost += predict_value(next_state, theta)
                    
                    future_cost /= len(next_states)
                    total_cost = immediate_cost + discount_factor * future_cost
                    
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
                    
                    if p2h > 0:
                        continue
                    
                    # Check constraints with new status
                    feasible, _ = check_constraints(wind_power, hydrogen_level, grid_power, h2p, 0, elec, data, t)
                    if not feasible:
                        continue
                    
                    immediate_cost = calculate_immediate_cost(
                        grid_price, wind_power, hydrogen_level, grid_power, h2p, 0, elec, data, t
                    )
                    
                    if immediate_cost >= 1000:
                        continue
                    
                    future_cost = 0.0
                    next_hydrogen = calculate_next_hydrogen(hydrogen_level, h2p, 0, data)
                    
                    for next_price, next_wind in next_states:
                        next_state = (next_price, next_wind, next_hydrogen, elec)
                        future_cost += predict_value(next_state, theta)
                    
                    future_cost /= len(next_states)
                    total_cost = immediate_cost + discount_factor * future_cost
                    
                    if total_cost < best_value:
                        best_value = total_cost
                        best_grid = grid_power
                        best_h2p = h2p
                        best_p2h = 0
                        best_elec_on = 0
                        best_elec_off = 1
    
    # Double-check the final decision for feasibility
    if best_value == float('inf'):
        # If no feasible solution found, use a safe fallback
        t_mod = t % len(data['demand_schedule'])
        best_demand = data['demand_schedule'][t_mod]
        best_grid = max(0, best_demand - wind_power)
        best_h2p = 0
        best_p2h = 0
        best_elec_on = 0
        best_elec_off = 0
    
    return best_elec_on, best_elec_off, best_grid, best_p2h, best_h2p

def adp_policy(t, electrolyzer_status, hydrogen_level, wind_power, grid_price, demand, data):
    """Main policy function - provides decisions for energy hub management"""
    global TRAINED_THETA
    
    # Train theta if not already trained
    if TRAINED_THETA is None:
        TRAINED_THETA = train_vfa()
    
    # Make decision using trained theta
    elec_on, elec_off, grid, p2h, h2p = make_decision(
        t, electrolyzer_status, hydrogen_level, wind_power, 
        grid_price, demand, data, TRAINED_THETA
    )
    
    # Final feasibility check before returning
    feasible, message = check_constraints(
        wind_power, hydrogen_level, grid, h2p, p2h,
        electrolyzer_status + elec_on - elec_off, data, t
    )
    
    if not feasible:
        # Fallback to a simple feasible solution
        t_mod = t % len(data['demand_schedule'])
        current_demand = data['demand_schedule'][t_mod]
        
        # Try to meet demand with wind and grid, no hydrogen conversion
        grid = max(0, current_demand - wind_power)
        h2p = 0
        p2h = 0
        elec_on = 0
        elec_off = 0
    
    return elec_on, elec_off, grid, p2h, h2p

def create_adp_policy():
    """
    Creates and returns the ADP policy function.
    Returns the policy function directly with no parameters required.
    """
    return adp_policy