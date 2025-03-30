import numpy as np
import os
import sys

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import required modules
from utils.data import get_fixed_data
from utils.WindProcess import wind_model
from utils.PriceProcess import price_model
from task_3.helper_functions import generate_trajectories


def train_vfa():
    """Train the Value Function Approximation using backward value function approximation"""
    data = get_fixed_data()
    num_timeslots = data['num_timeslots']
    
    # Initialize theta
    theta = np.zeros(6)
    
    # Step 1: Backward Value Function Approximation
    # For t = T, T-1, ..., τ:
    for t in range(num_timeslots-2, -1, -1):
        print(f"Time period {t}")
        
        # Step 1: Sample representative state pairs
        states = []
        for _ in range(5):
            # Sample z_t (wind and price)                                           # MÅSKE SKAL DER ÆNDRES HER
            wind = np.random.uniform(0, 10)
            price = np.random.uniform(data['price_floor'], data['price_cap'])
            
            # Sample y_t (hydrogen storage)
            hydrogen = np.random.uniform(0, data['hydrogen_capacity'])
            status = np.random.choice([0, 1])
            # Store the state                                                      # DEN HER RANDOM SAMPLING VIRKER MYSTISK 
            states.append((price, wind, hydrogen, status))
        
        # Step 2: For each sample
        for i, state in enumerate(states):
            price, wind, hydrogen, status = state
            print(f"  Sampled state: {state}")                                    # SAMME HER, MEN FRAMEWORKET VIRKER OK
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
                            immediate_cost = calculate_immediate_cost(price, wind, hydrogen, grid_power, h2p, p2h, elec, data, t)
                            reward = -immediate_cost
                            
                            # If last time period, target value is just the reward
                            if t == num_timeslots - 1:
                                target_values.append(reward)
                            else:
                                # Sample K next exogenous states
                                K = 3
                                value_functions = []
                                
                                for _ in range(K):
                                    # Sample z_{t+1} (next wind and price)
                                    next_wind = wind_model(wind, wind, data)                    # WIND WIND???? IKKE PREV WIND?
                                    next_price = price_model(price, price, next_wind, data)     # PRICE PRICE???? IKKE PREV PRICE?
                                    
                                    # Calculate next hydrogen level based on current decisions
                                    next_hydrogen = calculate_next_hydrogen(hydrogen, h2p, p2h, data)
                                    
                                    # Next state
                                    next_state = (next_price, next_wind, next_hydrogen, elec)
                                    
                                    # Calculate value function
                                    value = predict_value(next_state, theta)
                                    value_functions.append(value)
                                
                                # Target value = reward + (gamma/K) * sum(value functions)
                                gamma = 0.95
                                target = reward + (gamma / K) * sum(value_functions)
                                target_values.append(target)
            
            # Calculate V_target as max of target values
            if target_values:
                V_target = max(target_values)
                states[i] = (price, wind, hydrogen, status)  # Ensure state is updated
                
                # Step 3: Update theta
                # Generate 10 variations of theta
                thetas = []
                errors = []
                
                for _ in range(10):
                    theta_var = theta + np.random.normal(0, 0.1, size=6)
                    thetas.append(theta_var)
                    
                    # Calculate V_tilde
                    state_features = extract_features(states[i])
                    V_tilde = np.dot(theta_var, state_features)
                    
                    # Calculate error
                    error = (V_tilde - V_target) ** 2
                    errors.append(error)
                
                # Update theta to the one with minimum error
                theta = thetas[np.argmin(errors)]
                print(f"  Updated theta: {theta}")
    
    return theta


def extract_features(state):
    """Extract features for the value function approximation"""
    price, wind, hydrogen, status = state
    return np.array([
        1.0,              # Constant term (θ₀)
        price,            # Price (θ₁)
        wind,             # Wind (θ₂)
        hydrogen,         # Hydrogen storage (θ₃)
        float(status),    # Electrolyzer status (θ₄)
        hydrogen * price  # Interaction term (θ₅)      # SKAL VI HAVE DET HER MED? HELT NYT OG MÅSKE UNØDVENDIGT
    ])


def predict_value(state, theta):
    """Predict the value of a state using the current approximation"""
    features = extract_features(state)
    return np.dot(theta, features)


def calculate_immediate_cost(price, wind, hydrogen, grid_power, h2p, p2h, elec_status, data, time_period):
    """Calculate the immediate cost for a given combination of decisions"""
    t_mod = time_period % len(data['demand_schedule'])
    demand = data['demand_schedule'][t_mod]
    
    # Power balance check
    power_supply = wind + grid_power + h2p * data['conversion_h2p']
    power_demand = demand + p2h
    
    # If power balance is not satisfied, add a large penalty
    if power_supply < power_demand:
        return 1000  # Large penalty for infeasible solution
    
    # Calculate costs
    grid_cost = price * grid_power
    electrolyzer_cost = data['electrolyzer_cost'] * elec_status
    
    return grid_cost + electrolyzer_cost


def calculate_next_hydrogen(hydrogen, h2p, p2h, data):
    """Calculate the next hydrogen level based on decisions"""
    # Hydrogen used
    h_used = h2p
    
    # Hydrogen produced
    h_produced = p2h * data['conversion_p2h']
    
    # Next hydrogen level
    next_hydrogen = hydrogen - h_used + h_produced
    
    # Apply storage limits
    next_hydrogen = max(0, min(next_hydrogen, data['hydrogen_capacity']))
    
    return next_hydrogen


def make_decision(state, time_period, theta, data):
    """Make a decision for the current state using the trained VFA"""
    price, wind, hydrogen, status = state
    
    best_action = 0
    best_value = float('inf')
    best_grid = 0
    best_h2p = 0
    best_p2h = 0
    
    # Generate decision options                     # DET HER VIRKER OGSÅ MYSTISK
    grid_powers = [0, 2, 4]
    h2p_values = [0, 1, 2] if hydrogen > 0 else [0]
    p2h_values = [0, 2, 4] if status == 1 else [0]
    elec_statuses = [0, 1]
    
    # Sample next exogenous states
    next_states = []
    for _ in range(3):
        next_wind = wind_model(wind, wind, data)
        next_price = price_model(price, price, next_wind, data)
        next_states.append((next_price, next_wind))
    
    # For each possible action
    for grid_power in grid_powers:
        for h2p in h2p_values:
            for p2h in p2h_values:
                for elec in elec_statuses:
                    # Skip invalid combinations
                    if p2h > 0 and elec == 0:
                        continue
                    
                    # Calculate immediate cost
                    immediate_cost = calculate_immediate_cost(
                        price, wind, hydrogen, grid_power, h2p, p2h, elec, data, time_period
                    )
                    
                    # If not feasible, skip
                    if immediate_cost >= 1000:
                        continue
                    
                    # Calculate expected future cost
                    future_cost = 0.0
                    next_hydrogen = calculate_next_hydrogen(hydrogen, h2p, p2h, data)
                    
                    for next_price, next_wind in next_states:
                        next_state = (next_price, next_wind, next_hydrogen, elec)
                        future_cost += predict_value(next_state, theta)
                    
                    future_cost /= len(next_states)
                    
                    # Total cost
                    total_cost = immediate_cost + 0.95 * future_cost
                    
                    # Update best action
                    if total_cost < best_value:
                        best_value = total_cost
                        best_action = elec
                        best_grid = grid_power
                        best_h2p = h2p
                        best_p2h = p2h
    
    return best_action, best_grid, best_h2p, best_p2h


def execute_policy(theta):
    """Execute the VFA policy"""
    data = get_fixed_data()
    
    # Generate trajectories
    wind_trajectory, price_trajectory = generate_trajectories(data)
    
    # Initial state
    state = (price_trajectory[0], wind_trajectory[0], 0.0, 0)
    
    # Store results
    results = {
        'hydrogen_levels': [0.0],
        'electrolyzer_status': [0],
        'grid_power': [0],
        'h2p': [0],
        'p2h': [0]
    }
    
    print("\nExecuting policy:")
    
    # Step 2: Policy Execution
    for t in range(1, data['num_timeslots']):
        price, wind, hydrogen, status = state
        
        # Step 2.1: Compute optimal action
        action, grid, h2p, p2h = make_decision(state, t, theta, data)
        
        print(f"Time {t}: Price={price:.2f}, Wind={wind:.2f}, Storage={hydrogen:.2f}")
        print(f"  Decision: Electrolyzer={action}, Grid={grid}, H2P={h2p}, P2H={p2h}")
        
        # Calculate next hydrogen level
        next_hydrogen = calculate_next_hydrogen(hydrogen, h2p, p2h, data)
        
        # Update state
        state = (price_trajectory[t], wind_trajectory[t], next_hydrogen, action)
        
        # Store results
        results['hydrogen_levels'].append(next_hydrogen)
        results['electrolyzer_status'].append(action)
        results['grid_power'].append(grid)
        results['h2p'].append(h2p)
        results['p2h'].append(p2h)
    
    return results


if __name__ == "__main__":
    print("Training Value Function Approximation...")
    theta = train_vfa()
    
    print("\nFinal theta:")
    print(theta)
    
    print("\nExecuting policy...")
    results = execute_policy(theta)
    
    print("\nDone!")