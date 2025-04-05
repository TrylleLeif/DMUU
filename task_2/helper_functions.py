
from typing import Dict, List, Tuple, Any
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from utils.WindProcess import wind_model
from utils.PriceProcess import price_model
import numpy as np
from pyomo.environ import (ConcreteModel, Var, Binary, NonNegativeReals, Set, RangeSet,
                         Objective, Constraint, SolverFactory, TerminationCondition, 
                         minimize, value, Param)
#np.import_array()
# ===================================================
# || Generate scenarios for stochastic programming ||
# || Taken from lecture 4 & 5                      ||
# || claude.ai was also used for helping the build ||
# ===================================================

def scenario_tree_generation(
    current_wind: float,
    previous_wind: float,
    current_price: float,
    previous_price: float,
    data: Dict[str, Any],
    branches_per_stage: int,
    lookahead_horizon: int,
    initial_samples: int = 1000,
    clustering_method: str = "kmeans",  # Options: "kmeans", "kmedoids"
    _debug: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict[int, List[List[int]]]]:
    """
    Generate scenario tree for multi-stage SP.

    Returns:
        scenarios: Scenarios represented as (num_scenarios, horizon, 2) array
        probabilities: Probability of each scenario
        non_anticipativity_sets: Dictionary mapping time stages to lists of 
                                scenario groups that share history
    """
    if _debug:
        print(f"\n--- SCENARIO TREE GENERATION (using {clustering_method}) ---")
        print(f"Starting with wind={current_wind:.2f}, price={current_price:.2f}")
        print(f"Using {branches_per_stage} branches per stage, {lookahead_horizon} stages")
    
    # tree structure
    tree = {0: [{"wind": current_wind, 
                "price": current_price, 
                "probability": 1.0, 
                "parent": None}]}
    
    # Think of building the tree stage by stage
    for stage in range(1, lookahead_horizon):
        tree[stage] = []
        
        if _debug:
            print(f"\nGenerating stage {stage} from {len(tree[stage-1])} parent nodes")
        
        # For each node in the previous stage
        for parent_idx, parent_node in enumerate(tree[stage-1]):
            parent_wind = parent_node["wind"]
            parent_price = parent_node["price"]
            parent_prob = parent_node["probability"]
            
            if _debug and parent_idx == 0:
                print(f"  Parent {parent_idx}: wind={parent_wind:.2f}, price={parent_price:.2f}, prob={parent_prob:.3f}")
            
            # Generate samples from this parent node
            wind_samples = np.zeros(initial_samples)
            price_samples = np.zeros(initial_samples)
            
            for i in range(initial_samples):
                wind_samples[i] = wind_model(parent_wind, previous_wind, data)
                price_samples[i] = price_model(parent_price, previous_price, wind_samples[i], data)
            
            # Prepare data for clustering
            samples = np.column_stack((wind_samples, price_samples))
            
            # Apply selected clustering method
            if clustering_method.lower() == "kmeans":
                # K-means clustering
                kmeans = KMeans(n_clusters=branches_per_stage, random_state=42)
                kmeans.fit(samples)
                
                # Calculate probabilities for each cluster
                cluster_labels = kmeans.labels_
                cluster_centers = kmeans.cluster_centers_
                
            elif clustering_method.lower() == "kmedoids":
                # K-medoids clustering
                kmedoids = KMedoids(n_clusters=branches_per_stage, random_state=42)
                kmedoids.fit(samples)
                
                # Calculate probabilities for each cluster
                cluster_labels = kmedoids.labels_
                # For k-medoids, the centers are actual data points
                cluster_centers = samples[kmedoids.medoid_indices_]
                
            else:
                raise ValueError(f"Unsupported clustering method: {clustering_method}")
            
            # Calculate probabilities for each cluster
            cluster_counts = np.bincount(cluster_labels, minlength=branches_per_stage)
            cluster_probs = cluster_counts / initial_samples
            
            if _debug and parent_idx == 0:
                print(f"  Generated {branches_per_stage} branches with cluster sizes: {cluster_counts}")
            
            # Create child nodes for each cluster/branch
            for branch in range(branches_per_stage):
                child_node = {
                    "wind": cluster_centers[branch, 0],
                    "price": cluster_centers[branch, 1],
                    "probability": parent_prob * cluster_probs[branch],
                    "parent": parent_idx
                }
                tree[stage].append(child_node)
                
                if _debug and parent_idx == 0:
                    print(f"    Branch {branch}: wind={child_node['wind']:.2f}, price={child_node['price']:.2f}, prob={child_node['probability']:.3f}")
    
    # Get scenarios from the tree (each path from root to leaf)
    num_scenarios = len(tree[lookahead_horizon-1])
    scenarios = np.zeros((num_scenarios, lookahead_horizon, 2))
    probabilities = np.zeros(num_scenarios)
    
    if _debug:
        print(f"\nExtracting {num_scenarios} scenarios from the tree")
    
    # For each leaf node (scenario)
    for scenario_idx, leaf_node in enumerate(tree[lookahead_horizon-1]):
        # Record the probability of this scenario
        probabilities[scenario_idx] = leaf_node["probability"]
        
        # Work backwards from leaf to root to build the scenario
        current_node = leaf_node
        current_stage = lookahead_horizon - 1
        
        if _debug and scenario_idx < 3:  # Show only first few scenarios
            print(f"\nScenario {scenario_idx} (prob={leaf_node['probability']:.3f}):")
            print(f"  Stage {current_stage}: wind={current_node['wind']:.2f}, price={current_node['price']:.2f}")
        
        while current_stage >= 0:
            # Record wind and price for this stage
            scenarios[scenario_idx, current_stage, 0] = current_node["wind"]
            scenarios[scenario_idx, current_stage, 1] = current_node["price"]
            
            # Move to parent node
            if current_stage > 0:
                parent_idx = current_node["parent"]
                current_node = tree[current_stage-1][parent_idx]
                
                if _debug and scenario_idx < 3:  # Show only first few scenarios
                    print(f"  Stage {current_stage-1}: wind={current_node['wind']:.2f}, price={current_node['price']:.2f}")
            
            current_stage -= 1
    
    # Determine non-anticipativity sets
    non_anticipativity_sets = {}
    
    for t in range(lookahead_horizon):
        non_anticipativity_sets[t] = []
        # Map each node at stage t to the scenarios that pass through it
        node_to_scenarios = {}
        
        for s in range(num_scenarios):
            # Create a key representing the scenario's history up to stage t
            history_key = tuple(tuple(round(scenarios[s, i, j], 4) for j in range(2)) for i in range(t+1))
            
            if history_key not in node_to_scenarios:
                node_to_scenarios[history_key] = []
            
            node_to_scenarios[history_key].append(s)
        
        # Add each group of scenarios sharing history to the sets
        for scenario_group in node_to_scenarios.values():
            if len(scenario_group) > 1:  # Only needed for groups with multiple scenarios
                non_anticipativity_sets[t].append(scenario_group)
    
    if _debug:
        print("\nNon-anticipativity sets:")
        for t in range(lookahead_horizon):
            if non_anticipativity_sets[t]:
                print(f"  Stage {t}: {len(non_anticipativity_sets[t])} sets")
                for i, group in enumerate(non_anticipativity_sets[t][:2]):  # Show only first few groups
                    print(f"    Group {i}: {group}")
                if len(non_anticipativity_sets[t]) > 2:
                    print(f"    ... and {len(non_anticipativity_sets[t])-2} more groups")
            else:
                print(f"  Stage {t}: No non-anticipativity sets")
                
        # Verify probabilities sum to 1
        print(f"\nSum of scenario probabilities: {np.sum(probabilities):.6f}")
    
    return scenarios, probabilities, non_anticipativity_sets

def stochastic_programming_policy(
    current_time: int,
    electrolyzer_status: int,
    hydrogen_level: float,
    wind_power: float,
    grid_price: float,
    demand: float,
    data: Dict[str, Any],
    previous_wind: float = None,
    previous_price: float = None,
    lookahead_horizon: int = 3,
    branches_per_stage: int = 3,
    _debug: bool = False,
    clustering_method: str = "kmeans"
) -> Tuple[int, int, float, float, float]:
    """
    Multi-stage stochastic programming policy for the energy hub problem.
    """
    if _debug:
        print("\n==== STOCHASTIC PROGRAMMING POLICY ====")
        print(f"Current state: time={current_time}, electrolyzer={'ON' if electrolyzer_status else 'OFF'}")
        print(f"H level={hydrogen_level:.2f}, wind={wind_power:.2f}, price={grid_price:.2f}, demand={demand:.2f}")
        print(f"Using lookahead_horizon={lookahead_horizon}, branches_per_stage={branches_per_stage}")
        
        # Calculate number of variables
        total_scenarios = branches_per_stage**(lookahead_horizon-1)
        total_vars = 5 * lookahead_horizon * total_scenarios
        print(f"Total scenarios: {total_scenarios}, Total variables: {total_vars}")
    
    # use constants given by prof
    if previous_wind is None:
        previous_wind =  4 # given by professor
    if previous_price is None:
        previous_price = 28 # given by professor
    
    # Generate scenario tree
    scenarios, probabilities, non_anticipativity_sets = scenario_tree_generation(
        wind_power, previous_wind, grid_price, previous_price,
        data, branches_per_stage, lookahead_horizon, _debug=_debug, clustering_method=clustering_method
    )
    
    num_scenarios = len(probabilities)
    
    if _debug:
        print("\n--- OPTIMIZATION MODEL ---")
        print(f"Building model with {num_scenarios} scenarios and {lookahead_horizon} stages")
    
    
    
    m = ConcreteModel()
    
    # Define sets
    m.T = RangeSet(0, lookahead_horizon-1)  # Time periods (0 = here-and-now)
    m.S = RangeSet(0, num_scenarios-1)      # Scenarios
    
    # Define parameters for each scenario
    def wind_param_init(m, t, s):
        if t == 0:
            return wind_power  # Current wind is known
        else:
            return scenarios[s, t, 0]  # Future wind from scenarios
    
    def price_param_init(m, t, s):
        if t == 0:
            return grid_price  # Current price is known
        else:
            return scenarios[s, t, 1]  # Future price from scenarios
    
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
    m.y_on = Var(m.T, m.S, domain=Binary)
    m.y_off = Var(m.T, m.S, domain=Binary)
    m.x = Var(m.T, m.S, domain=Binary)
    m.p_grid = Var(m.T, m.S, domain=NonNegativeReals)
    m.p_p2h = Var(m.T, m.S, domain=NonNegativeReals)
    m.p_h2p = Var(m.T, m.S, domain=NonNegativeReals)
    m.h = Var(m.T, m.S, domain=NonNegativeReals)
    
    # Define objective function - minimize expected cost
    def obj_rule(m):
        return sum(m.probability[s] * sum(m.price[t, s] * m.p_grid[t, s] + 
                                              data['electrolyzer_cost'] * m.x[t, s] 
                                              for t in m.T) 
                                            for s in m.S)
    
    m.objective = Objective(rule=obj_rule, sense=minimize)
    
    # Define constraints
    
    # Power balance constraint
    def power_balance_rule(m, t, s):
        return (m.wind[t, s] + m.p_grid[t, s] + 
                data['conversion_h2p'] * m.p_h2p[t, s] - m.p_p2h[t, s] >= m.demand[t])
    
    m.power_balance = Constraint(m.T, m.S, rule=power_balance_rule)
    
    # Power-to-hydrogen rate constraint
    def p2h_limit_rule(m, t, s):
        return m.p_p2h[t, s] <= data['p2h_max_rate'] * m.x[t, s]
    
    m.p2h_limit = Constraint(m.T, m.S, rule=p2h_limit_rule)
    
    # Hydrogen-to-power rate constraint
    def h2p_limit_rule(m, t, s):
        return m.p_h2p[t, s] <= data['h2p_max_rate']
    
    m.h2p_limit = Constraint(m.T, m.S, rule=h2p_limit_rule)
    
    # Hydrogen storage balance constraint
    def storage_balance_rule(m, t, s):
        if t == 0:
            return m.h[t, s] == hydrogen_level
        else:
            return (m.h[t, s] == m.h[t-1, s] + 
                   data['conversion_p2h'] * m.p_p2h[t-1, s] - m.p_h2p[t-1, s])
    
    m.storage_balance = Constraint(m.T, m.S, rule=storage_balance_rule)
    
    # Hydrogen storage capacity constraint
    def storage_capacity_rule(m, t, s):
        return m.h[t, s] <= data['hydrogen_capacity']
    
    m.storage_capacity = Constraint(m.T, m.S, rule=storage_capacity_rule)
    
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
    
    # Non-anticipativity constraints based on the scenario tree structure
    na_count = 0
    for t in range(lookahead_horizon):
        for scenario_group in non_anticipativity_sets[t]:
            base_s = scenario_group[0]  # Reference scenario
            for other_s in scenario_group[1:]:  # Other scenarios in the same group
                # Only need these constraints for t+1 since t is already enforced
                if t < lookahead_horizon - 1:
                    m.add_component(f"na_y_on_{t}_{base_s}_{other_s}", 
                        Constraint(expr=m.y_on[t, base_s] == m.y_on[t, other_s]))
                    m.add_component(f"na_y_off_{t}_{base_s}_{other_s}", 
                        Constraint(expr=m.y_off[t, base_s] == m.y_off[t, other_s]))
                    m.add_component(f"na_p_grid_{t}_{base_s}_{other_s}", 
                        Constraint(expr=m.p_grid[t, base_s] == m.p_grid[t, other_s]))
                    m.add_component(f"na_p_p2h_{t}_{base_s}_{other_s}", 
                        Constraint(expr=m.p_p2h[t, base_s] == m.p_p2h[t, other_s]))
                    m.add_component(f"na_p_h2p_{t}_{base_s}_{other_s}", 
                        Constraint(expr=m.p_h2p[t, base_s] == m.p_h2p[t, other_s]))
                    na_count += 5
    
    if _debug:
        print(f"Added {na_count} non-anticipativity constraints")
        print("Solving optimization model...")
    
    # Solve the model
    solver = SolverFactory('gurobi')
    solver.options['TimeLimit'] = 60  # time limit (just in case ;))
    
    try:
        results = solver.solve(m, tee=False)
        
        # Extract first-stage (here-and-now) decisions
        if results.solver.termination_condition == TerminationCondition.optimal:
            electrolyzer_on = int(value(m.y_on[0, 0]))
            electrolyzer_off = int(value(m.y_off[0, 0]))
            p_grid = value(m.p_grid[0, 0])
            p_p2h = value(m.p_p2h[0, 0])
            p_h2p = value(m.p_h2p[0, 0])
            
            if _debug:
                print("\n--- OPTIMAL SOLUTION ---")
                print(f"Solver status: {results.solver.status}, termination condition: {results.solver.termination_condition}")
                print(f"Objective value: {value(m.objective):.2f}")
                print("First-stage decisions:")
                print(f"  electrolyzer_on = {electrolyzer_on}")
                print(f"  electrolyzer_off = {electrolyzer_off}")
                print(f"  p_grid = {p_grid:.2f}")
                print(f"  p_p2h = {p_p2h:.2f}")
                print(f"  p_h2p = {p_h2p:.2f}")
                
                # Verify non-anticipativity is enforced
                print("\nVerifying non-anticipativity for first stage:")
                for s in range(min(3, num_scenarios)):  # Check just a few scenarios
                    print(f"  Scenario {s}: y_on={value(m.y_on[0, s])}, " + 
                          f"y_off={value(m.y_off[0, s])}, " +
                          f"p_grid={value(m.p_grid[0, s]):.2f}")
        else:
            print(f"\nOptimization failed: {results.solver.termination_condition}")
            electrolyzer_on = 0
            electrolyzer_off = 0
            p_grid = max(0, demand - wind_power)
            p_p2h = 0
            p_h2p = 0
            
            if _debug:
                print(f"\nOptimization failed: {results.solver.termination_condition}")
                print("Using fallback policy decisions")
    except Exception as e:
        print(f"Error in stochastic optimization: {e}")
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
    previous_wind=None,
    previous_price=None,
    lookahead_horizon: int = 3,
    num_samples: int = 100
) -> Tuple[int, int, float, float, float]:
    """
    Expected Value policy for the energy hub problem.
    """
    # use constants given by prof
    if previous_wind is None:
        previous_wind =  4 # given by professor
    if previous_price is None:
        previous_price = 28 # given by professor
    
    # Limit the lookahead horizon to the number of remaining time steps
    horizon = min(lookahead_horizon, data['num_timeslots'] - current_time)
    
    expected_wind = np.zeros(horizon)
    expected_price = np.zeros(horizon)
    
    # Current values are known
    expected_wind[0] = wind_power
    expected_price[0] = grid_price
    
    # For each future time step, generate samples and take the average
    prev_wind = wind_power
    prev_wind2 = previous_wind
    prev_price = grid_price
    prev_price2 = previous_price
    
    for t in range(1, horizon):
        wind_samples = np.zeros(num_samples)
        price_samples = np.zeros(num_samples)
        
        for i in range(num_samples):
            # Generate one sample of wind and price
            wind_samples[i] = wind_model(prev_wind, prev_wind2, data)
            price_samples[i] = price_model(prev_price, prev_price2, wind_samples[i], data)
        
        # Take the average as the expected value
        expected_wind[t] = np.mean(wind_samples)
        expected_price[t] = np.mean(price_samples)
        
        # Update previous values for next iteration
        prev_wind2 = prev_wind
        prev_wind = expected_wind[t]
        prev_price2 = prev_price
        prev_price = expected_price[t]
    

    m = ConcreteModel()
    m.T = RangeSet(0, horizon-1)  # Time periods (0 = here-and-now)
    
    # Define parameters
    def wind_param_init(m, t):
        return expected_wind[t]
    
    def price_param_init(m, t):
        return expected_price[t]
    
    def demand_param_init(m, t):
        future_time = current_time + t
        if future_time < len(data['demand_schedule']):
            return data['demand_schedule'][future_time]
        else:
            return data['demand_schedule'][-1]  # Use last known demand if beyond horizon
    
    m.wind = Param(m.T, initialize=wind_param_init)
    m.price = Param(m.T, initialize=price_param_init)
    m.demand = Param(m.T, initialize=demand_param_init)
    
    # Define variables
    m.y_on = Var(m.T, domain=Binary)
    m.y_off = Var(m.T, domain=Binary)
    m.x = Var(m.T, domain=Binary)
    m.p_grid = Var(m.T, domain=NonNegativeReals)
    m.p_p2h = Var(m.T, domain=NonNegativeReals)
    m.p_h2p = Var(m.T, domain=NonNegativeReals)
    m.h = Var(m.T, domain=NonNegativeReals)
    
    # Define objective function
    def obj_rule(m):
        return sum(m.price[t] * m.p_grid[t] + 
                   data['electrolyzer_cost'] * m.x[t] 
                   for t in m.T)
    
    m.objective = Objective(rule=obj_rule, sense=minimize)
    
    # Define constraints (same as in stochastic programming policy)
    
    # Power balance constraint
    def power_balance_rule(m, t):
        return (m.wind[t] + m.p_grid[t] + 
                data['conversion_h2p'] * m.p_h2p[t] - m.p_p2h[t] >= m.demand[t])
    
    m.power_balance = Constraint(m.T, rule=power_balance_rule)
    
    # Power-to-hydrogen rate constraint
    def p2h_limit_rule(m, t):
        return m.p_p2h[t] <= data['p2h_max_rate'] * m.x[t]
    
    m.p2h_limit = Constraint(m.T, rule=p2h_limit_rule)
    
    # Hydrogen-to-power rate constraint
    def h2p_limit_rule(m, t):
        return m.p_h2p[t] <= data['h2p_max_rate']
    
    m.h2p_limit = Constraint(m.T, rule=h2p_limit_rule)
    
    # Hydrogen storage balance constraint
    def storage_balance_rule(m, t):
        if t == 0:
            return m.h[t] == hydrogen_level
        else:
            return (m.h[t] == m.h[t-1] + 
                   data['conversion_p2h'] * m.p_p2h[t-1] - m.p_h2p[t-1])
    
    m.storage_balance = Constraint(m.T, rule=storage_balance_rule)
    
    # Hydrogen availability constraint
    def h2p_availability_rule(m, t):
        return m.p_h2p[t] <= m.h[t]
    
    m.h2p_availability = Constraint(m.T, rule=h2p_availability_rule)
    
    # Hydrogen storage capacity constraint
    def storage_capacity_rule(m, t):
        return m.h[t] <= data['hydrogen_capacity']
    
    m.storage_capacity = Constraint(m.T, rule=storage_capacity_rule)
    
    # Electrolyzer status update constraint
    def status_update_rule(m, t):
        if t == 0:
            return m.x[t] == electrolyzer_status
        else:
            return m.x[t] == m.x[t-1] + m.y_on[t-1] - m.y_off[t-1]
    
    m.status_update = Constraint(m.T, rule=status_update_rule)
    
    # Constraint on switching actions
    def switch_limit_rule(m, t):
        return m.y_on[t] + m.y_off[t] <= 1
    
    m.switch_limit = Constraint(m.T, rule=switch_limit_rule)
    
    # Solve the m
    solver = SolverFactory('gurobi')
    solver.options['TimeLimit'] = 60 # time limit (just in case ;)
    
    try:
        results = solver.solve(m, tee=False)
        
        # Extract first-stage (here-and-now) decisions
        if results.solver.termination_condition == TerminationCondition.optimal:
            electrolyzer_on = int(value(m.y_on[0]))
            electrolyzer_off = int(value(m.y_off[0]))
            p_grid = value(m.p_grid[0])
            p_p2h = value(m.p_p2h[0])
            p_h2p = value(m.p_h2p[0])
        else:
            electrolyzer_on = 0
            electrolyzer_off = 0
            p_grid = max(0, demand - wind_power)
            p_p2h = 0
            p_h2p = 0
    except Exception as e:
        print(f"Error in expected value optimization: {e}")
        electrolyzer_on = 0
        electrolyzer_off = 0
        p_grid = max(0, demand - wind_power)
        p_p2h = 0
        p_h2p = 0
    
    return electrolyzer_on, electrolyzer_off, p_grid, p_p2h, p_h2p


# ===================================================
# || Helper functions for creating policies        ||
# || make it easier to create policies in main     ||
# ===================================================

def create_sp_policy(horizon: int, branches_per_stage: int, clustering_method: str = "kmeans"):
    """
    Creates a stochastic programming policy with specified parameters.
    """
    def policy(current_time, electrolyzer_status, hydrogen_level, wind_power, 
               grid_price, demand, data, previous_wind=None, previous_price=None
               ,*args, **kwargs):
        return stochastic_programming_policy(
            current_time, electrolyzer_status, hydrogen_level, wind_power, 
            grid_price, demand, data, previous_wind, previous_price,
            lookahead_horizon=horizon, branches_per_stage=branches_per_stage, clustering_method=clustering_method
        )
    return policy

def create_ev_policy(horizon: int = 5, num_samples: int = 100):
    """
    Creates an expected value policy with specified parameters.
    """
    def policy(current_time, electrolyzer_status, hydrogen_level, wind_power, 
               grid_price, demand, data, previous_wind=None, previous_price=None
               ,*args, **kwargs):
        return expected_value_policy(
            current_time, electrolyzer_status, hydrogen_level, wind_power, 
            grid_price, demand, data, previous_wind, previous_price,
            lookahead_horizon=horizon, num_samples=num_samples
        )
    return policy