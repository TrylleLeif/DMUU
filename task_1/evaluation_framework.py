import numpy as np
from typing import Dict, List, Tuple, Any, Callable

# Import the necessary modules
from utils.data import get_fixed_data
from utils.WindProcess import wind_model
from utils.PriceProcess import price_model
from task_1.energy_hub_policies import dummy_policy

def create_experiments(
        num_experiments: int, 
        data: Dict[str, Any]
    ) -> Tuple[List[int], np.ndarray, np.ndarray]:

    num_timeslots = data['num_timeslots']
    wind_trajectories = np.zeros((num_experiments, num_timeslots))
    price_trajectories = np.zeros((num_experiments, num_timeslots))
    
    for e in range(num_experiments):
        # Initialize first two values for each trajectory
        wind_trajectories[e, 0] = data['target_mean_wind']
        wind_trajectories[e, 1] = data['target_mean_wind']
        price_trajectories[e, 0] = data['mean_price']
        price_trajectories[e, 1] = data['mean_price']
        
        # Generate trajectories for the experiment
        for t in range(2, num_timeslots):
            wind_trajectories[e, t] = wind_model(
                wind_trajectories[e, t-1],
                wind_trajectories[e, t-2],
                data
            )
            price_trajectories[e, t] = price_model(
                price_trajectories[e, t-1],
                price_trajectories[e, t-2],
                wind_trajectories[e, t],
                data
            )
    
    return list(range(num_experiments)), wind_trajectories, price_trajectories

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

def evaluate_policy(
        policy: Callable,
        num_experiments: int = 20,
        verbose: bool = True
    ) -> Tuple[float, Dict[str, Any]]:
    
    # Load the fixed data
    data = get_fixed_data()
    
    # Create experiments
    expers, wind_trajectories, price_trajectories = create_experiments(num_experiments, data)
    
    # constants
    num_timeslots = data['num_timeslots']
    r_p2h = data['conversion_p2h']
    r_h2p = data['conversion_h2p']
    hydrogen_capacity = data['hydrogen_capacity']
    p2h_max_rate = data['p2h_max_rate'] 
    h2p_max_rate = data['h2p_max_rate'] 
    electrolyzer_cost = data['electrolyzer_cost']
    demand_schedule = data['demand_schedule']
    
    # arrays to track simulation results
    policy_cost = np.full((num_experiments, num_timeslots), np.nan)
    hydrogen_storage = np.full((num_experiments, num_timeslots+1), np.nan)
    electrolyzer_status_history = np.full((num_experiments, num_timeslots+1), np.nan)
    p_grid_history = np.full((num_experiments, num_timeslots), np.nan)
    p_p2h_history = np.full((num_experiments, num_timeslots), np.nan)
    p_h2p_history = np.full((num_experiments, num_timeslots), np.nan)
    electrolyzer_on_history = np.full((num_experiments, num_timeslots), np.nan) 
    electrolyzer_off_history = np.full((num_experiments, num_timeslots), np.nan) 

    # Set initial conditions for all experiments
    for e in expers:
        hydrogen_storage[e, 0] = 0
        electrolyzer_status_history[e, 0] = 0
    
    for e in expers:
        if verbose and e % 5 == 0:
            print(f"Processing experiment {e}...")
        
        # Simulate through all timeslots
        for t in range(num_timeslots):
            # Capture current state for policy decision
            current_electrolyzer_status = electrolyzer_status_history[e, t]
            current_hydrogen_level = hydrogen_storage[e, t]
            current_wind_power = wind_trajectories[e, t]
            current_grid_price = price_trajectories[e, t]
            current_demand = demand_schedule[t]
            
            # Call the policy to make a decision
            electrolyzer_on, electrolyzer_off, p_grid, p_p2h, p_h2p = policy(
                t, 
                current_electrolyzer_status, 
                current_hydrogen_level, 
                current_wind_power, 
                current_grid_price, 
                current_demand, 
                data
            )
            
            # Check if the decision is feasible given the current state
            is_feasible = check_feasibility(
                current_electrolyzer_status,
                electrolyzer_on,
                electrolyzer_off,
                p_grid,
                p_p2h,
                p_h2p,
                current_hydrogen_level,
                current_wind_power,
                current_demand,
                p2h_max_rate,
                h2p_max_rate,
                r_p2h,
                r_h2p,
                hydrogen_capacity
            )

            # Fall back to dummy policy if constraints are violated
            if not is_feasible:
                print(f"DECISION DOES NOT MEET THE CONSTRAINTS FOR EXPERIMENT {e}, TIMESLOT {t}. THE DUMMY POLICY WILL BE USED INSTEAD")
                electrolyzer_on, electrolyzer_off, p_grid, p_p2h, p_h2p = dummy_policy(
                    t, 
                    current_electrolyzer_status, 
                    current_hydrogen_level, 
                    current_wind_power, 
                    current_grid_price, 
                    current_demand, 
                    data
                )
            

            # Calculate the cost for this timeslot based on grid usage and electrolyzer operation
            cost = current_grid_price * p_grid + electrolyzer_cost * current_electrolyzer_status
            policy_cost[e, t] = cost
            
            # Determine the next electrolyzer status based on switching decisions
            # The status transitions from 0->1 if ON=1, from 1->0 if OFF=1, otherwise stays the same
            next_electrolyzer_status = current_electrolyzer_status + electrolyzer_on - electrolyzer_off # works because electrolyzer_on and electrolyzer_off are mutually exclusive in feasibility check
            
            # Update hydrogen storage based on power-to-hydrogen conversion and hydrogen-to-power usage
            next_hydrogen_level = current_hydrogen_level + r_p2h * p_p2h - p_h2p
            next_hydrogen_level = min(max(0, next_hydrogen_level), hydrogen_capacity)  # Enforce physical limits
            
            # Store the updated state for the next timeslot
            hydrogen_storage[e, t+1] = next_hydrogen_level
            electrolyzer_status_history[e, t+1] = next_electrolyzer_status
            
            # Store decision history for analysis
            p_grid_history[e, t] = p_grid
            p_p2h_history[e, t] = p_p2h
            p_h2p_history[e, t] = p_h2p
            electrolyzer_on_history[e, t] = electrolyzer_on
            electrolyzer_off_history[e, t] = electrolyzer_off
    
    # Calculate the total cost for each experiment
    total_costs = np.sum(policy_cost, axis=1)
    
    # Calculate the average cost across all experiments to measure policy performance
    average_cost = np.mean(total_costs)
    
    if verbose:
        print(f"Average policy cost: {average_cost:.2f}")
        print(f"Min cost: {np.min(total_costs):.2f}, Max cost: {np.max(total_costs):.2f}")
    
    return average_cost, {
        'total_costs': total_costs,
        'policy_cost': policy_cost,
        'hydrogen_storage': hydrogen_storage,
        'electrolyzer_status': electrolyzer_status_history,
        'electrolyzer_on': electrolyzer_on_history,  # Added to return history
        'electrolyzer_off': electrolyzer_off_history,  # Added to return history
        'p_grid': p_grid_history,
        'p_p2h': p_p2h_history,
        'p_h2p': p_h2p_history,
        'wind_trajectories': wind_trajectories,
        'price_trajectories': price_trajectories
    }
