import numpy as np
from typing import Dict, List, Tuple, Any
import pyomo.environ as pyo
from sklearn.cluster import KMeans
from utils.WindProcess import wind_model
from utils.PriceProcess import price_model

def scenario_generation(
    current_wind: float,
    previous_wind: float,
    current_price: float,
    previous_price: float,
    data: Dict[str, Any],
    num_samples: int = 100,
    num_scenarios: int = 10,
    lookahead_horizon: int = 3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate scenarios for stochastic programming.
    
    Args:
        current_wind: Current wind power
        previous_wind: Previous wind power
        current_price: Current electricity price
        previous_price: Previous electricity price
        data: Problem data dictionary
        num_samples: Number of initial Monte Carlo samples
        num_scenarios: Number of scenarios to reduce to
        lookahead_horizon: Number of time periods to look ahead
        
    Returns:
        scenarios: Reduced scenarios (num_scenarios, num_periods, 2)
        probabilities: Probability of each scenario
        full_scenarios: All generated scenarios before reduction (for analysis)
    """
    # Generate initial Monte Carlo samples
    wind_scenarios = np.zeros((num_samples, lookahead_horizon))
    price_scenarios = np.zeros((num_samples, lookahead_horizon))
    
    for s in range(num_samples):
        # Initialize with the current and previous values
        prev_wind, prev_prev_wind = current_wind, previous_wind
        prev_price, prev_prev_price = current_price, previous_price
        
        for t in range(lookahead_horizon):
            # Generate next wind and price values using the stochastic processes
            next_wind = wind_model(prev_wind, prev_prev_wind, data)
            next_price = price_model(prev_price, prev_prev_price, next_wind, data)
            
            # Store the generated values
            wind_scenarios[s, t] = next_wind
            price_scenarios[s, t] = next_price
            
            # Update for next iteration
            prev_prev_wind, prev_wind = prev_wind, next_wind
            prev_prev_price, prev_price = prev_price, next_price
    
    # Combine wind and price into a single array for clustering
    # First reshape to have one row per sample and columns for all time periods
    combined_scenarios = np.column_stack((wind_scenarios.reshape(num_samples, -1), 
                                         price_scenarios.reshape(num_samples, -1)))
    
    # Apply K-means clustering for scenario reduction
    kmeans = KMeans(n_clusters=num_scenarios, random_state=42, n_init=10)
    labels = kmeans.fit_predict(combined_scenarios)
    centroids = kmeans.cluster_centers_
    
    # Calculate probabilities based on cluster sizes
    unique_labels, counts = np.unique(labels, return_counts=True)
    probabilities = counts / num_samples
    
    # Reshape centroids back to (scenarios, time_periods, variables)
    n_vars = 2  # wind and price
    reduced_scenarios = np.zeros((num_scenarios, lookahead_horizon, n_vars))
    for i in range(num_scenarios):
        # Split the centroid back into wind and price components
        mid_point = lookahead_horizon
        wind_vals = centroids[i, :mid_point]
        price_vals = centroids[i, mid_point:]
        reduced_scenarios[i, :, 0] = wind_vals
        reduced_scenarios[i, :, 1] = price_vals
    
    # Store the full scenarios for analysis (optional)
    full_scenarios = np.zeros((num_samples, lookahead_horizon, n_vars))
    full_scenarios[:, :, 0] = wind_scenarios
    full_scenarios[:, :, 1] = price_scenarios
    
    return reduced_scenarios, probabilities, full_scenarios

def stochastic_programming(
    current_time: int,
    electrolyzer_status: int,
    hydrogen_level: float,
    wind_power: float,
    grid_price: float,
    demand: float,
    data: Dict[str, Any],
    lookahead_horizon: int = 3,
    num_scenarios: int = 10,
    initial_samples: int = 100
) -> Tuple[int, int, float, float, float]:
    """
    Stochastic programming policy for the energy hub problem.
    
    Args:
        current_time: Current time slot
        electrolyzer_status: Current electrolyzer status (0=off, 1=on)
        hydrogen_level: Current hydrogen storage level
        wind_power: Current wind power generation
        grid_price: Current electricity price
        demand: Current electricity demand
        data: Problem data dictionary
        lookahead_horizon: Number of time periods to look ahead
        num_scenarios: Number of scenarios to consider
        initial_samples: Number of initial Monte Carlo samples
        
    Returns:
        electrolyzer_on: Decision to turn electrolyzer on (0 or 1)
        electrolyzer_off: Decision to turn electrolyzer off (0 or 1)
        p_grid: Power drawn from the grid
        p_p2h: Power converted to hydrogen
        p_h2p: Hydrogen converted to power
    """
    # Get previous wind and price for scenario generation
    # If at start of simulation, use current values as previous values too
    if current_time <= 1:
        previous_wind = wind_power
        previous_price = grid_price
    else:
        # In a real implementation, you would track these values from previous iterations
        # For simplicity, we'll use an approximation
        previous_wind = wind_power * (1 - 0.1 * np.random.randn())
        previous_price = grid_price * (1 - 0.05 * np.random.randn())
    
    # Generate scenarios
    scenarios, probabilities, _ = scenario_generation(
        wind_power, previous_wind, grid_price, previous_price,
        data, initial_samples, num_scenarios, lookahead_horizon
    )
    
    # Create and solve the stochastic optimization model
    model = pyo.ConcreteModel()
    
    # Define sets
    model.T = pyo.RangeSet(0, lookahead_horizon-1)  # Time periods (0 = here-and-now)
    model.S = pyo.RangeSet(0, num_scenarios-1)      # Scenarios
    
    # Define parameters for each scenario
    def wind_param_init(model, t, s):
        if t == 0:
            return wind_power  # Current wind is known
        else:
            return scenarios[s, t-1, 0]  # Future wind from scenarios
    
    def price_param_init(model, t, s):
        if t == 0:
            return grid_price  # Current price is known
        else:
            return scenarios[s, t-1, 1]  # Future price from scenarios
    
    def demand_param_init(model, t):
        future_time = current_time + t
        if future_time < len(data['demand_schedule']):
            return data['demand_schedule'][future_time]
        else:
            return data['demand_schedule'][-1]  # Use last known demand if beyond horizon
    
    model.wind = pyo.Param(model.T, model.S, initialize=wind_param_init)
    model.price = pyo.Param(model.T, model.S, initialize=price_param_init)
    model.demand = pyo.Param(model.T, initialize=demand_param_init)
    model.probability = pyo.Param(model.S, initialize=lambda model, s: probabilities[s])
    
    # Define variables
    # First-stage and second-stage variables
    model.y_on = pyo.Var(model.T, model.S, domain=pyo.Binary)
    model.y_off = pyo.Var(model.T, model.S, domain=pyo.Binary)
    model.x = pyo.Var(model.T, model.S, domain=pyo.Binary)
    model.p_grid = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    model.p_p2h = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    model.p_h2p = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    model.h = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    
    # Define objective function - minimize expected cost
    def obj_rule(model):
        return sum(model.probability[s] * sum(model.price[t, s] * model.p_grid[t, s] + 
                                              data['electrolyzer_cost'] * model.x[t, s] 
                                              for t in model.T) 
                  for s in model.S)
    
    model.objective = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
    
    # Define constraints
    
    # Power balance constraint
    def power_balance_rule(model, t, s):
        return (model.wind[t, s] + model.p_grid[t, s] + 
                data['conversion_h2p'] * model.p_h2p[t, s] - model.p_p2h[t, s] >= model.demand[t])
    
    model.power_balance = pyo.Constraint(model.T, model.S, rule=power_balance_rule)
    
    # Power-to-hydrogen rate constraint
    def p2h_limit_rule(model, t, s):
        return model.p_p2h[t, s] <= data['p2h_max_rate'] * model.x[t, s]
    
    model.p2h_limit = pyo.Constraint(model.T, model.S, rule=p2h_limit_rule)
    
    # Hydrogen-to-power rate constraint
    def h2p_limit_rule(model, t, s):
        return model.p_h2p[t, s] <= data['h2p_max_rate']
    
    model.h2p_limit = pyo.Constraint(model.T, model.S, rule=h2p_limit_rule)
    
    # Hydrogen storage balance constraint
    def storage_balance_rule(model, t, s):
        if t == 0:
            return model.h[t, s] == hydrogen_level
        else:
            return (model.h[t, s] == model.h[t-1, s] + 
                   data['conversion_p2h'] * model.p_p2h[t-1, s] - model.p_h2p[t-1, s])
    
    model.storage_balance = pyo.Constraint(model.T, model.S, rule=storage_balance_rule)
    
    # Hydrogen availability constraint
    def h2p_availability_rule(model, t, s):
        return model.p_h2p[t, s] <= model.h[t, s]
    
    model.h2p_availability = pyo.Constraint(model.T, model.S, rule=h2p_availability_rule)
    
    # Hydrogen storage capacity constraint
    def storage_capacity_rule(model, t, s):
        return model.h[t, s] <= data['hydrogen_capacity']
    
    model.storage_capacity = pyo.Constraint(model.T, model.S, rule=storage_capacity_rule)
    
    # Electrolyzer status update constraint
    def status_update_rule(model, t, s):
        if t == 0:
            return model.x[t, s] == electrolyzer_status
        else:
            return model.x[t, s] == model.x[t-1, s] + model.y_on[t-1, s] - model.y_off[t-1, s]
    
    model.status_update = pyo.Constraint(model.T, model.S, rule=status_update_rule)
    
    # Constraint on switching actions
    def switch_limit_rule(model, t, s):
        return model.y_on[t, s] + model.y_off[t, s] <= 1
    
    model.switch_limit = pyo.Constraint(model.T, model.S, rule=switch_limit_rule)
    
    # Non-anticipativity constraints for first-stage variables
    def nonanticipativity_y_on_rule(model, s, s_prime):
        if s == s_prime:
            return pyo.Constraint.Skip
        return model.y_on[0, s] == model.y_on[0, s_prime]
    
    def nonanticipativity_y_off_rule(model, s, s_prime):
        if s == s_prime:
            return pyo.Constraint.Skip
        return model.y_off[0, s] == model.y_off[0, s_prime]
    
    def nonanticipativity_p_grid_rule(model, s, s_prime):
        if s == s_prime:
            return pyo.Constraint.Skip
        return model.p_grid[0, s] == model.p_grid[0, s_prime]
    
    def nonanticipativity_p_p2h_rule(model, s, s_prime):
        if s == s_prime:
            return pyo.Constraint.Skip
        return model.p_p2h[0, s] == model.p_p2h[0, s_prime]
    
    def nonanticipativity_p_h2p_rule(model, s, s_prime):
        if s == s_prime:
            return pyo.Constraint.Skip
        return model.p_h2p[0, s] == model.p_h2p[0, s_prime]
    
    model.nonanticipativity_y_on = pyo.Constraint(model.S, model.S, rule=nonanticipativity_y_on_rule)
    model.nonanticipativity_y_off = pyo.Constraint(model.S, model.S, rule=nonanticipativity_y_off_rule)
    model.nonanticipativity_p_grid = pyo.Constraint(model.S, model.S, rule=nonanticipativity_p_grid_rule)
    model.nonanticipativity_p_p2h = pyo.Constraint(model.S, model.S, rule=nonanticipativity_p_p2h_rule)
    model.nonanticipativity_p_h2p = pyo.Constraint(model.S, model.S, rule=nonanticipativity_p_h2p_rule)
    
    # Solve the model
    solver = pyo.SolverFactory('gurobi')
    solver.options['TimeLimit'] = 60  # Set a time limit to ensure the policy terminates
    
    try:
        results = solver.solve(model, tee=False)
        
        # Extract first-stage (here-and-now) decisions
        if results.solver.termination_condition == pyo.TerminationCondition.optimal:
            # Take first scenario's first-stage decisions (they should be the same across scenarios)
            electrolyzer_on = int(pyo.value(model.y_on[0, 0]))
            electrolyzer_off = int(pyo.value(model.y_off[0, 0]))
            p_grid = pyo.value(model.p_grid[0, 0])
            p_p2h = pyo.value(model.p_p2h[0, 0])
            p_h2p = pyo.value(model.p_h2p[0, 0])
        else:
            # If optimization failed, make a simple decision based on current state
            electrolyzer_on = 0
            electrolyzer_off = 0
            p_grid = max(0, demand - wind_power)
            p_p2h = 0
            p_h2p = 0
    except Exception as e:
        print(f"Error in stochastic optimization: {e}")
        # Fall back to a simple decision rule
        electrolyzer_on = 0
        electrolyzer_off = 0
        p_grid = max(0, demand - wind_power)
        p_p2h = 0
        p_h2p = 0
    
    return electrolyzer_on, electrolyzer_off, p_grid, p_p2h, p_h2p

def expected_value_policy(
    current_time: int,
    electrolyzer_status: int,
    hydrogen_level: float,
    wind_power: float,
    grid_price: float,
    demand: float,
    data: Dict[str, Any],
    lookahead_horizon: int = 3
) -> Tuple[int, int, float, float, float]:
    """
    Expected Value policy (simplified stochastic programming with a single expected scenario).
    
    Args:
        current_time: Current time slot
        electrolyzer_status: Current electrolyzer status (0=off, 1=on)
        hydrogen_level: Current hydrogen storage level
        wind_power: Current wind power generation
        grid_price: Current electricity price
        demand: Current electricity demand
        data: Problem data dictionary
        lookahead_horizon: Number of time periods to look ahead
        
    Returns:
        electrolyzer_on: Decision to turn electrolyzer on (0 or 1)
        electrolyzer_off: Decision to turn electrolyzer off (0 or 1)
        p_grid: Power drawn from the grid
        p_p2h: Power converted to hydrogen
        p_h2p: Hydrogen converted to power
    """
    # Generate expected wind and price trajectories
    expected_wind = np.zeros(lookahead_horizon)
    expected_price = np.zeros(lookahead_horizon)
    
    # Initialize with current values
    prev_wind, prev_prev_wind = wind_power, wind_power
    prev_price, prev_prev_price = grid_price, grid_price
    
    # Generate expected trajectories by repeatedly taking mean of many samples
    num_samples = 50
    for t in range(lookahead_horizon):
        wind_samples = np.zeros(num_samples)
        price_samples = np.zeros(num_samples)
        
        for s in range(num_samples):
            wind_samples[s] = wind_model(prev_wind, prev_prev_wind, data)
            price_samples[s] = price_model(prev_price, prev_prev_price, wind_samples[s], data)
        
        expected_wind[t] = np.mean(wind_samples)
        expected_price[t] = np.mean(price_samples)
        
        prev_prev_wind, prev_wind = prev_wind, expected_wind[t]
        prev_prev_price, prev_price = prev_price, expected_price[t]
    
    # Create and solve a deterministic optimization model
    model = pyo.ConcreteModel()
    
    # Define sets
    model.T = pyo.RangeSet(0, lookahead_horizon-1)  # Time periods (0 = here-and-now)
    
    # Define parameters
    def wind_param_init(model, t):
        if t == 0:
            return wind_power  # Current wind is known
        else:
            return expected_wind[t-1]  # Expected future wind
    
    def price_param_init(model, t):
        if t == 0:
            return grid_price  # Current price is known
        else:
            return expected_price[t-1]  # Expected future price
    
    def demand_param_init(model, t):
        future_time = current_time + t
        if future_time < len(data['demand_schedule']):
            return data['demand_schedule'][future_time]
        else:
            return data['demand_schedule'][-1]  # Use last known demand if beyond horizon
    
    model.wind = pyo.Param(model.T, initialize=wind_param_init)
    model.price = pyo.Param(model.T, initialize=price_param_init)
    model.demand = pyo.Param(model.T, initialize=demand_param_init)
    
    # Define variables
    model.y_on = pyo.Var(model.T, domain=pyo.Binary)
    model.y_off = pyo.Var(model.T, domain=pyo.Binary)
    model.x = pyo.Var(model.T, domain=pyo.Binary)
    model.p_grid = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.p_p2h = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.p_h2p = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.h = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    
    # Define objective function - minimize expected cost
    def obj_rule(model):
        return sum(model.price[t] * model.p_grid[t] + 
                  data['electrolyzer_cost'] * model.x[t] 
                  for t in model.T)
    
    model.objective = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
    
    # Define constraints
    
    # Power balance constraint
    def power_balance_rule(model, t):
        return (model.wind[t] + model.p_grid[t] + 
                data['conversion_h2p'] * model.p_h2p[t] - model.p_p2h[t] >= model.demand[t])
    
    model.power_balance = pyo.Constraint(model.T, rule=power_balance_rule)
    
    # Power-to-hydrogen rate constraint
    def p2h_limit_rule(model, t):
        return model.p_p2h[t] <= data['p2h_max_rate'] * model.x[t]
    
    model.p2h_limit = pyo.Constraint(model.T, rule=p2h_limit_rule)
    
    # Hydrogen-to-power rate constraint
    def h2p_limit_rule(model, t):
        return model.p_h2p[t] <= data['h2p_max_rate']
    
    model.h2p_limit = pyo.Constraint(model.T, rule=h2p_limit_rule)
    
    # Hydrogen storage balance constraint
    def storage_balance_rule(model, t):
        if t == 0:
            return model.h[t] == hydrogen_level
        else:
            return (model.h[t] == model.h[t-1] + 
                   data['conversion_p2h'] * model.p_p2h[t-1] - model.p_h2p[t-1])
    
    model.storage_balance = pyo.Constraint(model.T, rule=storage_balance_rule)
    
    # Hydrogen availability constraint
    def h2p_availability_rule(model, t):
        return model.p_h2p[t] <= model.h[t]
    
    model.h2p_availability = pyo.Constraint(model.T, rule=h2p_availability_rule)
    
    # Hydrogen storage capacity constraint
    def storage_capacity_rule(model, t):
        return model.h[t] <= data['hydrogen_capacity']
    
    model.storage_capacity = pyo.Constraint(model.T, rule=storage_capacity_rule)
    
    # Electrolyzer status update constraint
    def status_update_rule(model, t):
        if t == 0:
            return model.x[t] == electrolyzer_status
        else:
            return model.x[t] == model.x[t-1] + model.y_on[t-1] - model.y_off[t-1]
    
    model.status_update = pyo.Constraint(model.T, rule=status_update_rule)
    
    # Constraint on switching actions
    def switch_limit_rule(model, t):
        return model.y_on[t] + model.y_off[t] <= 1
    
    model.switch_limit = pyo.Constraint(model.T, rule=switch_limit_rule)
    
    # Solve the model
    solver = pyo.SolverFactory('gurobi')
    solver.options['TimeLimit'] = 60  # Set a time limit to ensure the policy terminates
    
    try:
        results = solver.solve(model, tee=False)
        
        # Extract first-stage (here-and-now) decisions
        if results.solver.termination_condition == pyo.TerminationCondition.optimal:
            electrolyzer_on = int(pyo.value(model.y_on[0]))
            electrolyzer_off = int(pyo.value(model.y_off[0]))
            p_grid = pyo.value(model.p_grid[0])
            p_p2h = pyo.value(model.p_p2h[0])
            p_h2p = pyo.value(model.p_h2p[0])
        else:
            # If optimization failed, make a simple decision based on current state
            electrolyzer_on = 0
            electrolyzer_off = 0
            p_grid = max(0, demand - wind_power)
            p_p2h = 0
            p_h2p = 0
    except Exception as e:
        print(f"Error in optimization: {e}")
        # Fall back to a simple decision rule
        electrolyzer_on = 0
        electrolyzer_off = 0
        p_grid = max(0, demand - wind_power)
        p_p2h = 0
        p_h2p = 0
    
    return electrolyzer_on, electrolyzer_off, p_grid, p_p2h, p_h2p

# Example configuration for different stochastic programming policies
def create_sp_policy(horizon: int, scenarios: int, samples: int = 100):
    """
    Creates a stochastic programming policy with specified parameters.
    """
    def policy(current_time, electrolyzer_status, hydrogen_level, wind_power, 
               grid_price, demand, data):
        return stochastic_programming(
            current_time, electrolyzer_status, hydrogen_level, wind_power, 
            grid_price, demand, data, horizon, scenarios, samples
        )
    return policy

def create_ev_policy(horizon: int):
    """
    Creates an expected value policy with specified lookahead horizon.
    """
    def policy(current_time, electrolyzer_status, hydrogen_level, wind_power, 
               grid_price, demand, data):
        return expected_value_policy(
            current_time, electrolyzer_status, hydrogen_level, wind_power, 
            grid_price, demand, data, horizon
        )
    return policy

sp_policy_short_horizon = create_sp_policy(horizon=2, scenarios=10)
sp_policy_long_horizon = create_sp_policy(horizon=5, scenarios=4)
sp_policy_medium = create_sp_policy(horizon=3, scenarios=6)
ev_policy = create_ev_policy(horizon=3)