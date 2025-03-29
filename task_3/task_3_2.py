import numpy as np
import matplotlib.pyplot as plt
from utils.data import get_fixed_data
from utils.WindProcess import wind_model
from utils.PriceProcess import price_model
from helper_functions import generate_trajectories, solve_milp


def train_value_function(num_iterations=10, discount_factor=0.95):
    """
    Train the Value Function Approximation (VFA) for the electrolyzer problem.
    
    The VFA has the form:
    Ṽ(price, wind, hydrogen, status; θ) = θ₀ + θ₁·price + θ₂·wind + θ₃·hydrogen + θ₄·status + θ₅·(hydrogen·price)
    
    Args:
        num_iterations: Number of training iterations
        discount_factor: Discount factor for future costs
        
    Returns:
        numpy.ndarray: Trained parameters theta
    """
    # Get fixed data
    data = get_fixed_data()
    num_timeslots = data['num_timeslots']
    
    # Initialize parameters
    theta = np.zeros(6)
    
    # Training history
    theta_history = [theta.copy()]
    errors = []
    
    # Step 1: Backward Value Function Approximation
    print("Starting backward value function approximation...")
    
    for iteration in range(num_iterations):
        print(f"Iteration {iteration+1}/{num_iterations}")
        
        # For t = T, T-1, ..., τ:
        for t in range(num_timeslots-2, -1, -1):
            # Step 1.1: Sample representative state pairs
            states = []
            for _ in range(20):  # Sample 20 states
                price = np.random.uniform(data['price_floor'], data['price_cap'])
                wind = np.random.uniform(0, 15)
                hydrogen = np.random.uniform(0, data['hydrogen_capacity'])
                status = np.random.choice([0, 1])
                states.append((price, wind, hydrogen, status))
            
            # Step 1.2: For each sample
            target_values = []
            
            for state in states:
                # Step 1.2.1: Sample K next exogenous states
                price, wind, hydrogen, status = state
                target_value = compute_target_value(state, t, theta, data, discount_factor)
                target_values.append(target_value)
            
            # Step 1.3: Update the parameter θₜ by minimizing squared error
            update_parameters(theta, states, target_values)
            
            # Store history
            theta_history.append(theta.copy())
            
            # Calculate error
            predictions = [predict_value(state, theta) for state in states]
            mse = np.mean([(p - t)**2 for p, t in zip(predictions, target_values)])
            errors.append(mse)
            
            print(f"  Time {t}: MSE = {mse:.4f}")
        
        print(f"  Parameters: {theta}")
    
    # Plot training progress
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(errors)
    plt.yscale('log')
    plt.title('Training Error')
    plt.xlabel('Updates')
    plt.ylabel('Mean Squared Error')
    
    plt.subplot(1, 2, 2)
    theta_history = np.array(theta_history)
    for i in range(6):
        plt.plot(theta_history[:, i], label=f'θ{i}')
    plt.title('Parameter Convergence')
    plt.xlabel('Updates')
    plt.ylabel('Parameter Value')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('vfa_training.png')
    
    return theta


def extract_features(state):
    """
    Extract features for the value function approximation.
    
    Args:
        state: Tuple of (price, wind, hydrogen, status)
        
    Returns:
        numpy.ndarray: Feature vector
    """
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
    """
    Predict the value of a state using the current approximation.
    
    Args:
        state: Tuple of (price, wind, hydrogen, status)
        theta: VFA parameters
        
    Returns:
        float: Predicted value
    """
    features = extract_features(state)
    return np.dot(theta, features)


def update_parameters(theta, states, target_values, learning_rate=0.01):
    """
    Update the VFA parameters to minimize squared error.
    
    Args:
        theta: Current parameters
        states: List of state tuples
        target_values: List of target values
        learning_rate: Learning rate for gradient descent
    """
    for state, target in zip(states, target_values):
        features = extract_features(state)
        prediction = np.dot(theta, features)
        error = target - prediction
        
        # Gradient descent update
        theta -= learning_rate * (-2 * error * features)


def compute_target_value(state, time_period, theta, data, discount_factor):
    """
    Compute the target value for a state by solving a small optimization problem.
    
    Args:
        state: Current state (price, wind, hydrogen, status)
        time_period: Current time period
        theta: Current VFA parameters
        data: Problem data
        discount_factor: Discount factor
        
    Returns:
        float: Target value
    """
    price, wind, hydrogen, status = state
    
    # For each possible action (electrolyzer on/off)
    best_value = float('inf')
    
    for next_status in [0, 1]:
        # If electrolyzer can't change state, skip
        if next_status == 0 and status == 0:
            continue
        
        # Sample next exogenous states
        total_cost = 0.0
        num_scenarios = 10
        
        for _ in range(num_scenarios):
            # Sample next exogenous state
            next_wind = wind_model(wind, wind, data)
            next_price = price_model(price, price, next_wind, data)
            
            # Calculate immediate cost and next hydrogen level
            immediate_cost, next_hydrogen = calculate_step_cost(
                price, wind, hydrogen, next_status, time_period, data
            )
            
            # Next state
            next_state = (next_price, next_wind, next_hydrogen, next_status)
            
            # Estimate future value
            future_cost = predict_value(next_state, theta)
            
            # Total cost
            total_cost += immediate_cost + discount_factor * future_cost
        
        # Average cost
        avg_cost = total_cost / num_scenarios
        
        # Update best value
        if avg_cost < best_value:
            best_value = avg_cost
    
    return best_value


def calculate_step_cost(price, wind, hydrogen, status, time_period, data):
    """
    Calculate the immediate cost and next hydrogen level for a given action.
    
    Args:
        price: Current electricity price
        wind: Current wind power
        hydrogen: Current hydrogen storage
        status: Electrolyzer status (0=off, 1=on)
        time_period: Current time period
        data: Problem data
        
    Returns:
        tuple: (immediate_cost, next_hydrogen)
    """
    # Get demand for current time period
    demand = data['demand_schedule'][time_period % len(data['demand_schedule'])]
    
    # Initialize variables
    p_to_h = 0.0  # Power to hydrogen
    h_to_p = 0.0  # Hydrogen to power
    grid_power = 0.0  # Power from grid
    
    # Power balance
    deficit = max(0, demand - wind)
    excess = max(0, wind - demand)
    
    if status == 1:  # Electrolyzer on
        # Can convert excess power to hydrogen
        p_to_h = min(excess, data['p2h_max_rate'])
        
        # Still need grid power for deficit
        grid_power = deficit
        
        # Update hydrogen level
        next_hydrogen = min(
            hydrogen + p_to_h * data['conversion_p2h'], 
            data['hydrogen_capacity']
        )
        
        # Electrolyzer cost
        electrolyzer_cost = data['electrolyzer_cost']
    else:  # Electrolyzer off
        # Convert hydrogen to power if needed
        h_to_p = min(hydrogen, deficit / data['conversion_h2p'])
        
        # Grid power for remaining deficit
        grid_power = max(0, deficit - h_to_p * data['conversion_h2p'])
        
        # Update hydrogen level
        next_hydrogen = hydrogen - h_to_p
        
        # No electrolyzer cost
        electrolyzer_cost = 0
    
    # Total immediate cost
    immediate_cost = electrolyzer_cost + grid_power * price
    
    return immediate_cost, next_hydrogen


def execute_policy(theta, initial_state=None, num_steps=24):
    """
    Execute the VFA policy from an initial state.
    
    Args:
        theta: Trained VFA parameters
        initial_state: Initial state (or None to generate)
        num_steps: Number of steps to simulate
        
    Returns:
        dict: Simulation results
    """
    data = get_fixed_data()
    
    # Generate trajectories
    wind_trajectory, price_trajectory = generate_trajectories(data)
    
    # Set initial state
    if initial_state is None:
        initial_state = (price_trajectory[0], wind_trajectory[0], 0.0, 0)
    
    # Initialize state and history
    state = initial_state
    
    history = {
        'time': [],
        'price': [],
        'wind': [],
        'hydrogen': [],
        'status': [],
        'action': [],
        'grid_power': [],
        'p_to_h': [],
        'h_to_p': [],
        'immediate_cost': []
    }
    
    total_cost = 0.0
    
    # Step 2: Policy Execution
    print("\nExecuting policy...")
    
    for t in range(num_steps):
        price, wind, hydrogen, status = state
        
        # Record state
        history['time'].append(t)
        history['price'].append(price)
        history['wind'].append(wind)
        history['hydrogen'].append(hydrogen)
        history['status'].append(status)
        
        # Step 2.1: At time τ, given current state, compute optimal action
        best_action, best_details = make_decision(state, t, theta, data)
        
        # Record action and details
        history['action'].append(best_action)
        history['grid_power'].append(best_details['grid_power'])
        history['p_to_h'].append(best_details['p_to_h'])
        history['h_to_p'].append(best_details['h_to_p'])
        history['immediate_cost'].append(best_details['cost'])
        
        # Update total cost
        total_cost += best_details['cost']
        
        # Update state for next time step
        if t < num_steps - 1:
            next_price = price_trajectory[t+1]
            next_wind = wind_trajectory[t+1]
            next_hydrogen = best_details['next_hydrogen']
            
            state = (next_price, next_wind, next_hydrogen, best_action)
    
    # Add total cost to history
    history['total_cost'] = total_cost
    
    return history


def make_decision(state, time_period, theta, data):
    """
    Make a decision for the current state using the trained VFA.
    
    Args:
        state: Current state (price, wind, hydrogen, status)
        time_period: Current time period
        theta: Trained VFA parameters
        data: Problem data
        
    Returns:
        tuple: (best_action, best_details)
    """
    price, wind, hydrogen, status = state
    
    # For each possible action (electrolyzer on/off)
    best_action = None
    best_value = float('inf')
    best_details = None
    
    # Sample next exogenous states
    next_states = []
    for _ in range(10):  # Sample 10 scenarios
        next_wind = wind_model(wind, wind, data)
        next_price = price_model(price, price, next_wind, data)
        next_states.append((next_price, next_wind))
    
    # For each possible action
    for next_status in [0, 1]:
        # If electrolyzer can't change state, skip
        if next_status == 0 and status == 0:
            continue
        
        # Calculate immediate cost and details
        immediate_details = calculate_decision_details(
            price, wind, hydrogen, next_status, time_period, data
        )
        
        # Calculate expected future cost
        future_cost = 0.0
        for next_price, next_wind in next_states:
            next_state = (
                next_price, 
                next_wind, 
                immediate_details['next_hydrogen'],
                next_status
            )
            future_cost += predict_value(next_state, theta)
        
        future_cost /= len(next_states)
        
        # Total cost
        total_cost = immediate_details['cost'] + 0.95 * future_cost
        
        # Update best action
        if total_cost < best_value:
            best_value = total_cost
            best_action = next_status
            best_details = immediate_details
    
    return best_action, best_details


def calculate_decision_details(price, wind, hydrogen, status, time_period, data):
    """
    Calculate detailed information for a decision.
    
    Args:
        price: Current electricity price
        wind: Current wind power
        hydrogen: Current hydrogen storage
        status: Electrolyzer status (0=off, 1=on)
        time_period: Current time period
        data: Problem data
        
    Returns:
        dict: Decision details
    """
    # Get demand for current time period
    demand = data['demand_schedule'][time_period % len(data['demand_schedule'])]
    
    # Initialize variables
    p_to_h = 0.0  # Power to hydrogen
    h_to_p = 0.0  # Hydrogen to power
    grid_power = 0.0  # Power from grid
    
    # Power balance
    deficit = max(0, demand - wind)
    excess = max(0, wind - demand)
    
    if status == 1:  # Electrolyzer on
        # Can convert excess power to hydrogen
        p_to_h = min(excess, data['p2h_max_rate'])
        
        # Still need grid power for deficit
        grid_power = deficit
        
        # Update hydrogen level
        next_hydrogen = min(
            hydrogen + p_to_h * data['conversion_p2h'], 
            data['hydrogen_capacity']
        )
        
        # Electrolyzer cost
        electrolyzer_cost = data['electrolyzer_cost']
    else:  # Electrolyzer off
        # Convert hydrogen to power if needed
        h_to_p = min(hydrogen, deficit / data['conversion_h2p'])
        
        # Grid power for remaining deficit
        grid_power = max(0, deficit - h_to_p * data['conversion_h2p'])
        
        # Update hydrogen level
        next_hydrogen = hydrogen - h_to_p
        
        # No electrolyzer cost
        electrolyzer_cost = 0
    
    # Total immediate cost
    immediate_cost = electrolyzer_cost + grid_power * price
    
    return {
        'grid_power': grid_power,
        'p_to_h': p_to_h,
        'h_to_p': h_to_p,
        'next_hydrogen': next_hydrogen,
        'electrolyzer_cost': electrolyzer_cost,
        'grid_cost': grid_power * price,
        'cost': immediate_cost
    }


def visualize_policy_execution(history):
    """
    Visualize the results of the policy execution.
    
    Args:
        history: Dictionary of simulation history
    """
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Wind and price
    plt.subplot(3, 1, 1)
    plt.plot(history['time'], history['wind'], 'b-', label='Wind Power')
    plt.ylabel('Wind Power')
    plt.twinx()
    plt.plot(history['time'], history['price'], 'r-', label='Electricity Price')
    plt.ylabel('Price')
    plt.title('Wind Power and Electricity Price')
    plt.legend(loc='upper left')
    plt.grid(True)
    
    # Plot 2: Hydrogen storage and electrolyzer status
    plt.subplot(3, 1, 2)
    plt.plot(history['time'], history['hydrogen'], 'g-', label='Hydrogen Storage')
    plt.ylabel('Hydrogen Storage')
    plt.twinx()
    plt.step(history['time'], history['status'], 'k-', label='Current Status')
    plt.step(history['time'], history['action'], 'r--', label='Decision')
    plt.ylabel('Status (0/1)')
    plt.title('Hydrogen Storage and Electrolyzer Status')
    plt.legend(loc='upper left')
    plt.grid(True)
    
    # Plot 3: Energy flows
    plt.subplot(3, 1, 3)
    plt.plot(history['time'], history['grid_power'], 'r-', label='Grid Power')
    plt.plot(history['time'], history['p_to_h'], 'g-', label='Power to Hydrogen')
    plt.plot(history['time'], history['h_to_p'], 'b-', label='Hydrogen to Power')
    plt.ylabel('Power Flow')
    plt.title('Energy Flows')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('vfa_policy_execution.png')
    plt.show()


def compare_with_milp(theta, num_steps=24):
    """
    Compare VFA policy with MILP solution.
    
    Args:
        theta: Trained VFA parameters
        num_steps: Number of steps to simulate
    """
    data = get_fixed_data()
    
    # Generate trajectories for fair comparison
    wind_trajectory, price_trajectory = generate_trajectories(data)
    
    # Solve with MILP
    print("Solving with MILP...")
    milp_results = solve_milp(wind_trajectory, price_trajectory, data, obj=False)
    milp_cost = solve_milp(wind_trajectory, price_trajectory, data, obj=True)
    
    # Execute VFA policy
    print("Executing VFA policy...")
    initial_state = (price_trajectory[0], wind_trajectory[0], 0.0, 0)
    vfa_history = execute_policy(theta, initial_state, num_steps)
    vfa_cost = vfa_history['total_cost']
    
    # Print comparison
    print("\nCost Comparison:")
    print(f"MILP Cost: {milp_cost}")
    print(f"VFA Cost:  {vfa_cost:.2f}")
    print(f"Cost Ratio (VFA/MILP): {vfa_cost/milp_cost:.2f}")
    
    # Plot comparison
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Electrolyzer status
    plt.subplot(2, 1, 1)
    plt.step(range(num_steps), milp_results['electrolyzer_status'][:num_steps], 'b-', where='post', label='MILP')
    plt.step(range(num_steps), vfa_history['status'], 'r--', where='post', label='VFA')
    plt.ylabel('Electrolyzer Status')
    plt.title('Electrolyzer Status: MILP vs VFA')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Hydrogen storage
    plt.subplot(2, 1, 2)
    plt.plot(range(num_steps), milp_results['hydrogen_storage_level'][:num_steps], 'b-', label='MILP')
    plt.plot(range(num_steps), vfa_history['hydrogen'], 'r--', label='VFA')
    plt.xlabel('Time')
    plt.ylabel('Hydrogen Storage Level')
    plt.title('Hydrogen Storage: MILP vs VFA')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('vfa_milp_comparison.png')
    plt.show()
    
    return {
        'milp_cost': milp_cost,
        'vfa_cost': vfa_cost,
        'milp_results': milp_results,
        'vfa_history': vfa_history
    }


if __name__ == "__main__":
    print("="*80)
    print(" Approximate Dynamic Programming for Electrolyzer Problem ")
    print("="*80)
    
    # Train the VFA
    print("\nTraining Value Function Approximation...")
    theta = train_value_function(num_iterations=5)  # Use more iterations in practice
    
    # Print final parameters
    print("\nFinal VFA Parameters:")
    param_names = ["Constant", "Price", "Wind", "Hydrogen", "Status", "Hydrogen×Price"]
    for i, (param, name) in enumerate(zip(theta, param_names)):
        print(f"  θ{i} ({name}): {param:.4f}")
    
    # Execute policy
    print("\nExecuting policy with trained VFA...")
    history = execute_policy(theta)
    
    # Visualize results
    print("\nVisualizing policy execution...")
    visualize_policy_execution(history)
    
    # Compare with MILP
    print("\nComparing with MILP solution...")
    comparison = compare_with_milp(theta)
    
    print("\nPerformance Summary:")
    print(f"  Total cost with VFA policy: {comparison['vfa_cost']:.2f}")
    print(f"  Total cost with MILP:       {comparison['milp_cost']}")
    print(f"  Cost ratio (VFA/MILP):      {comparison['vfa_cost']/comparison['milp_cost']:.2f}")
    print(f"  (Lower ratio is better, with 1.0 being optimal)")
    
    print("\nDone! Visualization files saved:")
    print("  - vfa_training.png")
    print("  - vfa_policy_execution.png")
    print("  - vfa_milp_comparison.png")