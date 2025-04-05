import numpy as np
from typing import Dict, List, Tuple, Any, Callable, Optional
from utils.WindProcess import wind_model
from utils.PriceProcess import price_model
import pickle

# ============================================
# || Utility Functions                      ||
# ============================================

def features(state, t: Optional[int] = 0):
    """
    Extract features from the state for value function approximation.
    """
    # Basic features
    basic_features = np.array([
        1.0,  # Bias term
        state['wind'],
        state['price'],
        state['electrolyzer_status'],
        state['hydrogen_level'],
    ])
    
    # Interaction terms
    interaction_features = np.array([
        state['wind'] * state['price'],  # Interaction between wind and price
        state['hydrogen_level'] * state['price'],  # Interaction between hydrogen storage and price
        state['electrolyzer_status'] * 1,  # Cost of running electrolyzer
        state['wind'] * state['hydrogen_level'],  # Potential for storing excess wind as hydrogen
    ])
    
    # Nonlinear transformations
    nonlinear_features = np.array([
        state['hydrogen_level'] ** 2,  # Quadratic term for hydrogen level
        np.maximum(0, 5.0 - state['wind']),  # ReLU-like feature for low wind
        np.maximum(0, state['price'] - 35.0),  # ReLU-like feature for high price
    ])
    
    return np.concatenate([basic_features, interaction_features, nonlinear_features])

def value_approximation(state, theta):
    return np.dot(features(state), theta)

def reward(electrolyzer_status, grid_power, grid_price, electrolyzer_cost):
    total_cost = grid_power * grid_price + electrolyzer_cost * electrolyzer_status
    return -total_cost  # Return negative cost as reward

def transition(
    electrolyzer_status: int,
    hydrogen_level: float,
    electrolyzer_on: int,
    electrolyzer_off: int,
    p_grid: float,
    p_p2h: float,
    p_h2p: float,
    wind_power: float,
    demand: float,
    data: Dict[str, Any]
) -> Tuple[int, float]:
    """
    Calculate the next state given current state and actions.
    """
    next_electrolyzer_status = electrolyzer_status + electrolyzer_on - electrolyzer_off
    
    available_capacity = data['hydrogen_capacity'] - hydrogen_level
    h_addition = min(data['conversion_p2h'] * p_p2h, available_capacity)
    p_p2h = h_addition / data['conversion_p2h'] if h_addition < data['conversion_p2h'] * p_p2h else p_p2h

    next_hydrogen_level = hydrogen_level + h_addition - p_h2p
    
    if next_hydrogen_level < 0 or next_hydrogen_level > data['hydrogen_capacity']:
        print(f"Warning: Hydrogen level out of bounds: {next_hydrogen_level}")
        next_hydrogen_level = min(max(0, next_hydrogen_level), data['hydrogen_capacity'])

    return next_electrolyzer_status, next_hydrogen_level

def sample_next_exogenous(
    current_wind: float,
    previous_wind: float,
    current_price: float,
    previous_price: float,
    data: Dict[str, Any],
    num_samples: int = 50
) -> List[Dict[str, float]]:
    """
    Sample possible next exogenous states (wind and price).
    """
    s = []
    for _ in range(num_samples):
        next_wind = wind_model(current_wind, previous_wind, data)
        next_price = price_model(current_price, previous_price, next_wind, data)
        s.append({
            'wind': next_wind,
            'price': next_price
        })
    
    return s

def next_state_sampler(wind, prev_wind, price, prev_price, data, num_samples=100):
    return sample_next_exogenous(wind, prev_wind, price, prev_price, data, num_samples)

def get_nba(
    state: Dict[str, Any],
    next_state_sampler: Callable,
    theta: np.ndarray,
    demand: float,
    data: Dict[str, Any],
    discount_factor: float = 0.95
) -> Tuple[int, int, float, float, float]:
    """
    Determine the optimal action for the current state using the approximate value function.
        
    Returns:
        Tuple of (electrolyzer_on, electrolyzer_off, p_grid, p_p2h, p_h2p)
    """
    # Get state 
    electrolyzer_status = state['electrolyzer_status']
    hydrogen_level = state['hydrogen_level']
    wind_power = state['wind']
    grid_price = state['price']
    
    # Get data
    #r_p2h = data['conversion_p2h']
    r_h2p = data['conversion_h2p']
    p2h_max = data['p2h_max_rate']
    h2p_max = data['h2p_max_rate']
    electrolyzer_cost = data['electrolyzer_cost']
    
    # Sample 
    next_exogenous_samples = next_state_sampler(
        wind_power,
        state.get('previous_wind', wind_power),
        grid_price,
        state.get('previous_price', grid_price),
        data
    )

    # ========================================
    # || Grid Search for "good" next action ||
    # ========================================

    best_value = float('-inf')
    best_action = None
    
    # Define the grid of possible actions
    possible_electrolyzer_on = [0, 1] if electrolyzer_status == 0 else [0]
    possible_electrolyzer_off = [0, 1] if electrolyzer_status == 1 else [0]
    
    # Only one of electrolyzer_on and electrolyzer_off can be 1
    action_pairs = [(on, off) for on in possible_electrolyzer_on for off in possible_electrolyzer_off 
                   if not (on == 1 and off == 1)]
    
    # Grid search
    for electrolyzer_on, electrolyzer_off in action_pairs:
        #next_electrolyzer_status = electrolyzer_status + electrolyzer_on - electrolyzer_off
        
        # Try different levels of p_p2h and p_h2p
        for p_p2h_factor in [0.0,0.2,0.4,0.6,0.8,1.0] if electrolyzer_status == 1 else [0.0]:
            p_p2h = p_p2h_factor * p2h_max * electrolyzer_status # electro t-1 
            
            for p_h2p_factor in [0.0,0.2,0.4,0.6,0.8,1.0]:
                p_h2p = min(p_h2p_factor * h2p_max, hydrogen_level)

                # here it is possible to take power from grid and store it
                # ___________________________
                # Separate grid power into two components: for demand and for storage
                p_grid_demand = max(0, demand - wind_power - r_h2p * p_h2p)
                p_grid_storage = p_p2h
                p_grid = p_grid_demand + p_grid_storage

                r = reward(
                    electrolyzer_status, 
                    p_grid, 
                    grid_price, 
                    electrolyzer_cost
                )
                
                # Calculate expected future value
                future_values = []
                for next_exo in next_exogenous_samples:
                    # Calculate next state
                    next_el_status, next_h_level = transition(
                        electrolyzer_status,
                        hydrogen_level,
                        electrolyzer_on,
                        electrolyzer_off,
                        p_grid,
                        p_p2h,
                        p_h2p,
                        wind_power,
                        demand,
                        data
                    )
                    
                    # Create next state dictionary
                    next_state = {
                        'wind': next_exo['wind'],
                        'price': next_exo['price'],
                        'electrolyzer_status': next_el_status,
                        'hydrogen_level': next_h_level,
                        'previous_wind': wind_power,
                        'previous_price': grid_price
                    }
                    
                    # Approximate value of next state
                    future_values.append(value_approximation(next_state, theta))
                

                expected_future_value = sum(future_values) / len(future_values)
                total_value = r + discount_factor * expected_future_value
                
                if total_value > best_value:
                    best_value = total_value
                    best_action = (electrolyzer_on, electrolyzer_off, p_grid, p_p2h, p_h2p)
    
    return best_action if best_action else (0, 0, max(0, demand - wind_power), 0, 0)
# ============================================
# || VFA Training ||
# ============================================

def generate_state_samples(num_samples: int, data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate random state samples for training the value function approximation.
    """
    samples = []
    
    wind_samples = np.zeros(num_samples)
    price_samples = np.zeros(num_samples)
    wind_samples[0] = 5
    price_samples[0] = 30

    for i in range(1, num_samples):
        if i == 1:
            wind_samples[i] = wind_model(wind_samples[0], 4, data)
            price_samples[i] = price_model(price_samples[0], 28, wind_samples[i], data)
        else:
            wind_samples[i] = wind_model(wind_samples[i-1], wind_samples[i-2], data)
            price_samples[i] = price_model(price_samples[i-1], price_samples[max(0, i-2)], wind_samples[i], data)
        
    # Generate random hydrogen levels and electrolyzer statuses
    # assuming uniform distribution
    hydrogen_capacity = data['hydrogen_capacity']
    hydrogen_samples = np.random.uniform(0, hydrogen_capacity, num_samples)
    electrolyzer_samples = np.random.choice([0, 1], num_samples)

    for i in range(num_samples):
        state = {
            'wind': wind_samples[i],
            'price': price_samples[i],
            'hydrogen_level': hydrogen_samples[i],
            'electrolyzer_status': electrolyzer_samples[i]
        }
        
        if i > 0:
            state['previous_wind'] = wind_samples[i-1]
            state['previous_price'] = price_samples[i-1]
        else:
            state['previous_wind'] = wind_samples[i]
            state['previous_price'] = price_samples[i]
        
        samples.append(state)
    
    return samples

def train_vfa(
    data: Dict[str, Any],
    num_iterations: int = 10,
    num_time_steps: int = 24,
    discount_factor: float = 0.95,
    num_state_samples: int = 100,
    num_next_state_samples: int = 50
) -> List[np.ndarray]:
    """
    Train value function approximation using backward recursion.

    Returns:
        List of theta parameter vectors for each time step
    """
    # Number of features
    num_features = len(features({'wind': 0, 'price': 0, 'electrolyzer_status': 0, 'hydrogen_level': 0}))
    
    # Initialize theta parameters
    theta_list = [np.zeros(num_features) for _ in range(num_time_steps + 1)]
    
    # Demand schedule
    demands = data['demand_schedule']
    
    # Loop over iterations
    for iteration in range(num_iterations):
        print(f"VFA Training Iteration {iteration+1}/{num_iterations}")
        
        # Backward recursion over time steps
        for t in reversed(range(num_time_steps)):
            print(f"Processing time step {t}")
            
            # Generate state samples for this time step
            state_samples = generate_state_samples(num_state_samples, data)
            
            # Prepare features and targets for regression
            features_matrix = []
            targets = []
            
            # For each state sample
            for state in state_samples:
                
                # Get the current demand for this time step
                current_demand = demands[t]
                
                # Get the optimal action for this state
                optimal_action = get_nba(
                    state,
                    next_state_sampler,
                    theta_list[t+1] if t < num_time_steps - 1 else np.zeros(num_features),
                    current_demand,
                    data,
                    discount_factor
                )
                
                # Calculate immediate reward
                electrolyzer_on, electrolyzer_off, p_grid, p_p2h, p_h2p = optimal_action
                immediate_reward = reward(
                    state['electrolyzer_status'],
                    p_grid,
                    state['price'],
                    data['electrolyzer_cost']
                )
                
                # Calculate expected future value
                future_values = []
                next_exo_samples = next_state_sampler(
                    state['wind'],
                    state.get('previous_wind', state['wind']),
                    state['price'],
                    state.get('previous_price', state['price']),
                    data
                )
                
                for next_exo in next_exo_samples:
                    # Calculate next state
                    next_electrolyzer_status, next_hydrogen_level = transition(
                        state['electrolyzer_status'],
                        state['hydrogen_level'],
                        electrolyzer_on,
                        electrolyzer_off,
                        p_grid,
                        p_p2h,
                        p_h2p,
                        state['wind'],
                        current_demand,
                        data
                    )
                    
                    # Create next state dictionary
                    next_state = {
                        'wind': next_exo['wind'],
                        'price': next_exo['price'],
                        'electrolyzer_status': next_electrolyzer_status,
                        'hydrogen_level': next_hydrogen_level,
                        'previous_wind': state['wind'],
                        'previous_price': state['price']
                    }
                    
                    # Approximate value of next state
                    next_value = 0
                    if t < num_time_steps - 1:
                        next_value = value_approximation(next_state, theta_list[t+1])
                    
                    future_values.append(next_value)
                
                # Expected future value
                expected_future_value = sum(future_values) / len(future_values)
                
                # Target value for this state
                target_value = immediate_reward + discount_factor * expected_future_value
                
                # Store feature vector and target
                features_matrix.append(features(state,t))
                targets.append(target_value)
            
            # Convert to numpy arrays
            features_matrix = np.array(features_matrix)
            targets = np.array(targets)
            
            # Train linear regression model to get new theta
            # Use least squares to find theta that minimizes ||X*theta - y||^2
            try:
                theta_list[t] = np.linalg.lstsq(features_matrix, targets, rcond=None)[0]
            except np.linalg.LinAlgError:
                print(f"Warning: LinAlgError at t={t}")
    
    return theta_list

# ============================================
# || ADP Policy ||
# ============================================

def adp_policy_final(
    current_time: int,
    electrolyzer_status: int,
    hydrogen_level: float,
    wind_power: float,
    grid_price: float,
    demand: float,
    data: Dict[str, Any],
    previous_wind: float = None,
    previous_price: float = None
) -> Tuple[int, int, float, float, float]:
    """
    ADP policy using value function approximation.
    """
    global _trained_theta_list
    
    # Check if we've already trained the value function approximation
    if '_trained_theta_list' not in globals():
        print("Training VFA parameters (first run)...")
        _trained_theta_list = train_vfa(data)
        # Save the trained theta list to a file
        with open('/Users/khs/code/DMUU/task_3/trained_theta_list.pkl', 'wb') as f:
            pickle.dump(_trained_theta_list, f)
        print("VFA training complete!")
    
    state = {
        'wind': wind_power,
        'price': grid_price,
        'electrolyzer_status': electrolyzer_status,
        'hydrogen_level': hydrogen_level,
        'previous_wind': previous_wind if previous_wind is not None else wind_power,
        'previous_price': previous_price if previous_price is not None else grid_price
    }
    
    # Get the appropriate theta for the current time
    theta = _trained_theta_list[min(current_time, len(_trained_theta_list) - 1)]

    optimal_action = get_nba(
        state,
        next_state_sampler,
        theta,
        demand,
        data
    )
    
    return optimal_action
