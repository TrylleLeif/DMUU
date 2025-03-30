import numpy as np
from typing import Dict, List, Tuple, Any
import pyomo.environ as pyo
from pyomo.environ import (ConcreteModel, Var, Binary, NonNegativeReals, Set, RangeSet,
                         Objective, Constraint, SolverFactory, TerminationCondition, 
                         minimize, value, Param)
from sklearn.cluster import KMeans
from utils.WindProcess import wind_model
from utils.PriceProcess import price_model

# ===================================================
# || Generate scenarios for stochastic programming ||
# || Taken from lecture 4                          ||
# ===================================================

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

    #  =======================================
    #  || Step 1:                           ||
    #  || We generate a lot of scenarios by ||
    #  || Monte Carlo sampling              ||
    #  =======================================
    wind_scenarios = np.zeros((num_samples, lookahead_horizon))
    price_scenarios = np.zeros((num_samples, lookahead_horizon))
    
    for s in range(num_samples):
        prev_wind, prev_prev_wind = current_wind, previous_wind
        prev_price, prev_prev_price = current_price, previous_price
        
        for t in range(lookahead_horizon):
            wind_scenarios[s, t] = wind_model(prev_wind, prev_prev_wind, data)
            price_scenarios[s, t] = price_model(prev_price, prev_prev_price, wind_scenarios[s, t], data)
            
            prev_prev_wind, prev_wind = prev_wind, wind_scenarios[s, t]
            prev_prev_price, prev_price = prev_price, price_scenarios[s, t]
    # Combine wind and price into a single array for clustering
    # reshapeen to have one row per sample and columns for all time periods
    combined_scenarios = np.column_stack((wind_scenarios.reshape(num_samples, -1), 
                                         price_scenarios.reshape(num_samples, -1)))
    
    # Apply K-means clustering for scenario reduction
    kmeans = KMeans(n_clusters=num_scenarios, random_state=42)
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
    m = ConcreteModel()
    
    # Define sets
    m.T = RangeSet(0, lookahead_horizon-1)  # Time periods (0 = here-and-now)
    m.S = RangeSet(0, num_scenarios-1)      # Scenarios
    
    # Define parameters for each scenario
    def wind_param_init(m, t, s):
        if t == 0:
            return wind_power  # Current wind is known
        else:
            return scenarios[s, t-1, 0]  # Future wind from scenarios
    
    def price_param_init(m, t, s):
        if t == 0:
            return grid_price  # Current price is known
        else:
            return scenarios[s, t-1, 1]  # Future price from scenarios
    
    def demand_param_init(m, t):
        future_time = current_time + t
        if future_time < len(data['demand_schedule']):
            return data['demand_schedule'][future_time]
        else:
            return data['demand_schedule'][-1]  # Use last known demand if beyond horizon
    
    m.wind = Param(m.T, m.S, initialize=wind_param_init)
    m.price = Param(m.T, m.S, initialize=price_param_init)
    m.demand = Param(m.T, initialize=demand_param_init)
    m.probability = Param(m.S, initialize=lambda m, s: probabilities[s])
    
    # Define variables
    # First-stage and second-stage variables
    m.x = Var(m.T, m.S, within=Binary)
    m.y_on = Var(m.T, m.S, within=Binary)
    m.y_off = Var(m.T, m.S, within=Binary)
    m.h = Var(m.T, m.S, within=NonNegativeReals)
    m.p_grid = Var(m.T, m.S, within=NonNegativeReals)
    m.p_p2h = Var(m.T, m.S, within=NonNegativeReals)
    m.p_h2p = Var(m.T, m.S, within=NonNegativeReals)
    
    
    # Define objective function - minimize expected cost
    def obj_rule(m):
        return sum(m.probability[s] * sum(m.price[t, s] * m.p_grid[t, s] + 
                                              data['electrolyzer_cost'] * m.x[t, s] 
                                              for t in m.T) 
                                                for s in m.S)
    
    m.objective = Objective(rule=obj_rule, sense=minimize)
    
    # Power balance constraint
    def power_balance_rule(m, t, s):
        return (m.wind[t, s] + m.p_grid[t, s] + 
                data['conversion_h2p'] * m.p_h2p[t, s] - m.p_p2h[t, s] >= m.demand[t])
    
    m.power_balance = Constraint(m.T, m.S, rule=power_balance_rule)
    
    # Power-to-hydrogen limit constraint
    def p2h_limit_rule(m, t, s):
        return m.p_p2h[t, s] <= data['p2h_max_rate'] * m.x[t, s]
    
    m.p2h_limit = Constraint(m.T, m.S, rule=p2h_limit_rule)
    
    # Hydrogen-to-power limit constraint
    def h2p_limit_rule(m, t, s):
        return m.p_h2p[t, s] <= data['h2p_max_rate']
    
    m.h2p_limit = Constraint(m.T, m.S, rule=h2p_limit_rule)
    
    # Storage balance constraint
    def storage_balance_rule(m, t, s):
        if t == 0:
            return m.h[t, s] == hydrogen_level
        else:
            return (m.h[t, s] == m.h[t-1, s] + 
                   data['conversion_p2h'] * m.p_p2h[t-1, s] - m.p_h2p[t-1, s])
    
    m.storage_balance = Constraint(m.T, m.S, rule=storage_balance_rule)
    
    # TODO: Check if this is needed: 
    # Hydrogen availability constraint
    def h2p_availability_rule(m, t, s):
        return m.p_h2p[t, s] <= m.h[t, s]
    
    m.h2p_availability = Constraint(m.T, m.S, rule=h2p_availability_rule)
    
    # Storage capacity constraint
    def storage_capacity_rule(m, t, s):
        return m.h[t, s] <= data['hydrogen_capacity']
    
    m.storage_capacity = Constraint(m.T, m.S, rule=storage_capacity_rule)
    
    # def electrolyzer_rule(m, t, s):
    #     return m.y_on[t,s] + m.y_off[t,s] <= 1
    # m.electrolyzer = Constraint(m.T, m.S, rule=electrolyzer_rule)

    # def electrolyzer_on_rule(m, t, s):
    #     return m.y_on[t, s] <= m.x[t, s]
    # m.electrolyzer_on = Constraint(m.T, m.S, rule=electrolyzer_on_rule)

    # def electrolyzer_off_rule(m, t, s):
    #     return m.x[t, s] <= 1 - m.y_off[t, s]
    # m.electrolyzer_off = Constraint(m.T, m.S, rule=electrolyzer_off_rule)

    #     # Initialize y[0] = 0
    # def y_on_init_rule(m, t, s):
    #     if t == 0:
    #         return m.y_on[t, s] == 0
    #     return Constraint.Skip
    # m.y_on_init = Constraint(m.T, m.S, rule=y_on_init_rule)
    # def y_off_init_rule(m, t, s):
    #     if t == 0:
    #         return m.y_off[t, s] == 0
    #     return Constraint.Skip
    # m.y_off_init = Constraint(m.T, m.S, rule=y_off_init_rule)
    # Electrolyzer status update constraint
    def status_update_rule(m, t, s):
        if t == 0:
            return m.x[t, s] == electrolyzer_status
        else:
            return m.x[t, s] == m.x[t-1, s] + m.y_on[t-1, s] - m.y_off[t-1, s]
    
    m.status_update = Constraint(m.T, m.S, rule=status_update_rule)
    
    # Constraint on switching actions
    def switch_limit_rule(m, t, s):
        return m.y_on[t, s] + m.y_off[t, s] <= 1
    
    m.switch_limit = Constraint(m.T, m.S, rule=switch_limit_rule)
    
    # Non-anticipativity constraints for first-stage variables
    def nonanticipativity_y_on_rule(m, s, s_prime):
        if s == s_prime:
            return Constraint.Skip
        return m.y_on[0, s] == m.y_on[0, s_prime]
    
    def nonanticipativity_y_off_rule(m, s, s_prime):
        if s == s_prime:
            return Constraint.Skip
        return m.y_off[0, s] == m.y_off[0, s_prime]
    
    def nonanticipativity_p_grid_rule(m, s, s_prime):
        if s == s_prime:
            return Constraint.Skip
        return m.p_grid[0, s] == m.p_grid[0, s_prime]
    
    def nonanticipativity_p_p2h_rule(m, s, s_prime):
        if s == s_prime:
            return Constraint.Skip
        return m.p_p2h[0, s] == m.p_p2h[0, s_prime]
    
    def nonanticipativity_p_h2p_rule(m, s, s_prime):
        if s == s_prime:
            return Constraint.Skip
        return m.p_h2p[0, s] == m.p_h2p[0, s_prime]
    
    m.nonanticipativity_y_on = Constraint(m.S, m.S, rule=nonanticipativity_y_on_rule)
    m.nonanticipativity_y_off = Constraint(m.S, m.S, rule=nonanticipativity_y_off_rule)
    m.nonanticipativity_p_grid = Constraint(m.S, m.S, rule=nonanticipativity_p_grid_rule)
    m.nonanticipativity_p_p2h = Constraint(m.S, m.S, rule=nonanticipativity_p_p2h_rule)
    m.nonanticipativity_p_h2p = Constraint(m.S, m.S, rule=nonanticipativity_p_h2p_rule)
    
    # Solve the model
    solver = SolverFactory('gurobi')
    solver.options['TimeLimit'] = 60  # Set a time limit to ensure the policy terminates
    
    try:
        results = solver.solve(m, tee=False)
        
        # Extract first-stage (here-and-now) decisions
        if results.solver.termination_condition == TerminationCondition.optimal:
            # Take first scenario's first-stage decisions (they should be the same across scenarios)
            electrolyzer_on = int(value(m.y_on[0, 0]))
            electrolyzer_off = int(value(m.y_off[0, 0]))
            p_grid = value(m.p_grid[0, 0])
            p_p2h = value(m.p_p2h[0, 0])
            p_h2p = value(m.p_h2p[0, 0])
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

sp_policy_short_horizon = create_sp_policy(horizon=1, scenarios=5)
sp_policy_long_horizon = create_sp_policy(horizon=10, scenarios=5)
sp_policy_medium = create_sp_policy(horizon=5, scenarios=5)
ev_policy = create_ev_policy(horizon=5)