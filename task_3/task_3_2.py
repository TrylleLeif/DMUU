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
        
        # Step 1.1: Sample representative state pairs
        states = []
        for _ in range(5):
            price = np.random.uniform(data['price_floor'], data['price_cap'])
            wind = np.random.uniform(0, 10)
            hydrogen = np.random.uniform(0, data['hydrogen_capacity'])
            status = np.random.choice([0, 1])
            states.append((price, wind, hydrogen, status))
        
        # Step 1.2: For each sample
        target_values = []
        for state in states:
            # Step 1.2.1: Sample K next exogenous states
            target = compute_target_value(state, t, theta, data)
            target_values.append(target)
        
        # Step 1.3: Update theta
        update_theta(theta, states, target_values)
    
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
        hydrogen * price  # Interaction term (θ₅)
    ])


def predict_value(state, theta):
    """Predict the value of a state using the current approximation"""
    features = extract_features(state)
    return np.dot(theta, features)


def update_theta(theta, states, target_values, learning_rate=0.001):
    """Update theta to minimize the squared error"""
    for state, target in zip(states, target_values):
        features = extract_features(state)
        prediction = np.dot(theta, features)
        error = target - prediction
        
        # Apply a small learning rate and clip gradients
        gradient = -2 * error * features
        gradient = np.clip(gradient, -1, 1)
        theta -= learning_rate * gradient
    
    # Clip theta values to prevent numerical issues
    theta = np.clip(theta, -10, 10)
    
    print(f"Updated theta: {theta}")


def compute_target_value(state, time_period, theta, data):
    """Compute the target value for a state by solving a small optimization problem"""
    price, wind, hydrogen, status = state
    
    # For each possible action (electrolyzer on/off)
    best_value = float('inf')
    
    for next_status in [0, 1]:
        # Sample next exogenous states
        total_cost = 0.0
        num_scenarios = 3
        
        for _ in range(num_scenarios):
            # Generate next exogenous state
            next_wind = wind_model(wind, wind, data)
            next_price = price_model(price, price, next_wind, data)
            
            # Calculate immediate cost and next hydrogen level
            immediate_cost, next_hydrogen = calculate_step(
                price, wind, hydrogen, next_status, time_period, data
            )
            
            # Next state
            next_state = (next_price, next_wind, next_hydrogen, next_status)
            
            # Estimate future value
            future_cost = predict_value(next_state, theta)
            
            # Total cost
            total_cost += immediate_cost + 0.95 * future_cost
        
        # Average cost
        avg_cost = total_cost / num_scenarios
        
        # Update best value
        if avg_cost < best_value:
            best_value = avg_cost
    
    return best_value


def calculate_step(price, wind, hydrogen, status, time_period, data):
    """Calculate immediate cost and next hydrogen level for an action"""
    t_mod = time_period % len(data['demand_schedule'])
    demand = data['demand_schedule'][t_mod]
    
    deficit = max(0, demand - wind)
    excess = max(0, wind - demand)
    
    if status == 1:  # Electrolyzer on
        p_to_h = min(excess, data['p2h_max_rate'])
        grid_power = deficit
        next_hydrogen = min(
            hydrogen + p_to_h * data['conversion_p2h'], 
            data['hydrogen_capacity']
        )
        electrolyzer_cost = data['electrolyzer_cost']
    else:  # Electrolyzer off
        h_to_p = min(hydrogen, deficit / max(data['conversion_h2p'], 0.001))
        grid_power = max(0, deficit - h_to_p * data['conversion_h2p'])
        next_hydrogen = hydrogen - h_to_p
        electrolyzer_cost = 0
    
    immediate_cost = electrolyzer_cost + grid_power * price
    
    return immediate_cost, next_hydrogen


def make_decision(state, time_period, theta, data):
    """Make a decision for the current state using the trained VFA"""
    price, wind, hydrogen, status = state
    
    best_action = 0
    best_value = float('inf')
    
    # Sample next exogenous states
    next_states = []
    for _ in range(3):
        next_wind = wind_model(wind, wind, data)
        next_price = price_model(price, price, next_wind, data)
        next_states.append((next_price, next_wind))
    
    # For each possible action
    for next_status in [0, 1]:
        immediate_cost, next_hydrogen = calculate_step(
            price, wind, hydrogen, next_status, time_period, data
        )
        
        # Calculate expected future cost
        future_cost = 0.0
        for next_price, next_wind in next_states:
            next_state = (next_price, next_wind, next_hydrogen, next_status)
            future_cost += predict_value(next_state, theta)
        
        future_cost /= len(next_states)
        
        # Total cost
        total_cost = immediate_cost + 0.95 * future_cost
        
        # Update best action
        if total_cost < best_value:
            best_value = total_cost
            best_action = next_status
    
    return best_action


def execute_policy(theta):
    """Execute the VFA policy"""
    data = get_fixed_data()
    
    # Generate trajectories
    wind_trajectory, price_trajectory = generate_trajectories(data)
    
    # Initial state
    state = (price_trajectory[0], wind_trajectory[0], 0.0, 0)
    
    # Store results
    hydrogen_levels = [0.0]
    electrolyzer_status = [0]
    
    print("\nExecuting policy:")
    
    # Step 2: Policy Execution
    for t in range(1, data['num_timeslots']):
        price, wind, hydrogen, status = state
        
        # Step 2.1: Compute optimal action
        action = make_decision(state, t, theta, data)
        
        print(f"Time {t}: Price={price:.2f}, Wind={wind:.2f}, H2={hydrogen:.2f}, Action={action}")
        
        # Calculate next state
        _, next_hydrogen = calculate_step(
            price, wind, hydrogen, action, t, data
        )
        
        # Update state
        state = (price_trajectory[t], wind_trajectory[t], next_hydrogen, action)
        
        # Store results
        hydrogen_levels.append(next_hydrogen)
        electrolyzer_status.append(action)
    
    return {
        'hydrogen_levels': hydrogen_levels,
        'electrolyzer_status': electrolyzer_status
    }


if __name__ == "__main__":
    print("Training Value Function Approximation...")
    theta = train_vfa()
    
    print("\nFinal theta:")
    print(theta)
    
    print("\nExecuting policy...")
    results = execute_policy(theta)
    
    print("\nDone!")