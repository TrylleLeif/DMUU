import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.cluster import KMeans
from utils.WindProcess import wind_model
from utils.PriceProcess import price_model
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
    initial_samples: int = 100,
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
        print("\n--- SCENARIO TREE GENERATION ---")
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
            
            # gen samples from this parent node
            wind_samples = np.zeros(initial_samples)
            price_samples = np.zeros(initial_samples)
            
            for i in range(initial_samples):
                wind_samples[i] = wind_model(parent_wind, previous_wind, data)
                price_samples[i] = price_model(parent_price, previous_price, wind_samples[i], data)
            
            # Cluster to create branch points
            kmeans = KMeans(n_clusters=branches_per_stage, random_state=42)
            kmeans.fit(np.column_stack((wind_samples, price_samples)))
            
            # Calculate probabilities for each cluster
            cluster_counts = np.bincount(kmeans.labels_, minlength=branches_per_stage)
            cluster_probs = cluster_counts / initial_samples
            
            if _debug and parent_idx == 0:
                print(f"  Generated {branches_per_stage} branches with cluster sizes: {cluster_counts}")
            
            # Create child nodes for each cluster/branch
            for branch in range(branches_per_stage):
                child_node = {
                    "wind": kmeans.cluster_centers_[branch, 0],
                    "price": kmeans.cluster_centers_[branch, 1],
                    "probability": parent_prob * cluster_probs[branch],
                    "parent": parent_idx
                }
                tree[stage].append(child_node)
                
                if _debug and parent_idx == 0:
                    print(f"    Branch {branch}: wind={child_node['wind']:.2f}, price={child_node['price']:.2f}, prob={child_node['probability']:.3f}")
    
    # get scenarios from the tree (each path from root to leaf)
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
    _debug: bool = False
) -> Tuple[int, int, float, float, float]:
    """
    Multi-stage stochastic programming policy for the energy hub problem.
    
    Args:
        current_time: Current time slot
        electrolyzer_status: Current electrolyzer status (0=off, 1=on)
        hydrogen_level: Current hydrogen storage level
        wind_power: Current wind power generation
        grid_price: Current electricity price
        demand: Current electricity demand
        data: Problem data dictionary
        previous_wind: Previous wind power generation 
        previous_price: Previous electricity price
        previous_wind_2: Wind power from two time steps ago
        previous_price_2: Price from two time steps ago
        lookahead_horizon: Number of time periods to look ahead
        branches_per_stage: Number of branches at each stage
        _debug: Whether to print validation information
        
    Returns:
        electrolyzer_on: Decision to turn electrolyzer on (0 or 1)
        electrolyzer_off: Decision to turn electrolyzer off (0 or 1)
        p_grid: Power drawn from the grid
        p_p2h: Power converted to hydrogen
        p_h2p: Hydrogen converted to power
    """
    if _debug:
        print("\n==== STOCHASTIC PROGRAMMING POLICY ====")
        print(f"Current state: time={current_time}, electrolyzer={'ON' if electrolyzer_status else 'OFF'}")
        print(f"H level={hydrogen_level:.2f}, wind={wind_power:.2f}, price={grid_price:.2f}, demand={demand:.2f}")
        print(f"Using lookahead_horizon={lookahead_horizon}, branches_per_stage={branches_per_stage}")
        
        # Calculate number of variables
        total_scenarios = branches_per_stage**(lookahead_horizon-1)
        total_vars = 7 * lookahead_horizon * total_scenarios
        print(f"Total scenarios: {total_scenarios}, Total variables: {total_vars}")
    
    # Use provided previous wind and price if available, otherwise use estimates
    if previous_wind is None:
        previous_wind = wind_power * 0.95  # Simple estimate
    if previous_price is None:
        previous_price = grid_price * 0.98  # Simple estimate
    
    # Generate scenario tree
    scenarios, probabilities, non_anticipativity_sets = scenario_tree_generation(
        wind_power, previous_wind, grid_price, previous_price,
        data, branches_per_stage, lookahead_horizon, _debug=_debug
    )
    
    num_scenarios = len(probabilities)
    
    if _debug:
        print("\n--- OPTIMIZATION MODEL ---")
        print(f"Building model with {num_scenarios} scenarios and {lookahead_horizon} stages")
    
    # Create and solve the stochastic optimization model
    import pyomo.environ as pyo
    
    model = pyo.ConcreteModel()
    
    # Define sets
    model.T = pyo.RangeSet(0, lookahead_horizon-1)  # Time periods (0 = here-and-now)
    model.S = pyo.RangeSet(0, num_scenarios-1)      # Scenarios
    
    # Define parameters for each scenario
    def wind_param_init(model, t, s):
        if t == 0:
            return wind_power  # Current wind is known
        else:
            return scenarios[s, t, 0]  # Future wind from scenarios
    
    def price_param_init(model, t, s):
        if t == 0:
            return grid_price  # Current price is known
        else:
            return scenarios[s, t, 1]  # Future price from scenarios
    
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
    
    # Non-anticipativity constraints based on the scenario tree structure
    na_count = 0
    for t in range(lookahead_horizon):
        for scenario_group in non_anticipativity_sets[t]:
            base_s = scenario_group[0]  # Reference scenario
            for other_s in scenario_group[1:]:  # Other scenarios in the same group
                # Only need these constraints for t+1 since t is already enforced
                if t < lookahead_horizon - 1:
                    model.add_component(f"na_y_on_{t}_{base_s}_{other_s}", 
                        pyo.Constraint(expr=model.y_on[t, base_s] == model.y_on[t, other_s]))
                    model.add_component(f"na_y_off_{t}_{base_s}_{other_s}", 
                        pyo.Constraint(expr=model.y_off[t, base_s] == model.y_off[t, other_s]))
                    model.add_component(f"na_p_grid_{t}_{base_s}_{other_s}", 
                        pyo.Constraint(expr=model.p_grid[t, base_s] == model.p_grid[t, other_s]))
                    model.add_component(f"na_p_p2h_{t}_{base_s}_{other_s}", 
                        pyo.Constraint(expr=model.p_p2h[t, base_s] == model.p_p2h[t, other_s]))
                    model.add_component(f"na_p_h2p_{t}_{base_s}_{other_s}", 
                        pyo.Constraint(expr=model.p_h2p[t, base_s] == model.p_h2p[t, other_s]))
                    na_count += 5
    
    if _debug:
        print(f"Added {na_count} non-anticipativity constraints")
        print("Solving optimization model...")
    
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
            
            if _debug:
                print("\n--- OPTIMAL SOLUTION ---")
                print(f"Solver status: {results.solver.status}, termination condition: {results.solver.termination_condition}")
                print(f"Objective value: {pyo.value(model.objective):.2f}")
                print("First-stage decisions:")
                print(f"  electrolyzer_on = {electrolyzer_on}")
                print(f"  electrolyzer_off = {electrolyzer_off}")
                print(f"  p_grid = {p_grid:.2f}")
                print(f"  p_p2h = {p_p2h:.2f}")
                print(f"  p_h2p = {p_h2p:.2f}")
                
                # Verify non-anticipativity is enforced
                print("\nVerifying non-anticipativity for first stage:")
                for s in range(min(3, num_scenarios)):  # Check just a few scenarios
                    print(f"  Scenario {s}: y_on={pyo.value(model.y_on[0, s])}, " + 
                          f"y_off={pyo.value(model.y_off[0, s])}, " +
                          f"p_grid={pyo.value(model.p_grid[0, s]):.2f}")
        else:
            # If optimization failed, make a simple decision based on current state
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
    previous_wind=None,
    previous_price=None,
    lookahead_horizon: int = 5,
    num_samples: int = 100,
    *args, **kwargs
) -> Tuple[int, int, float, float, float]:
    """
    Expected Value policy for the energy hub problem.
    
    This policy uses a deterministic model with expected values of future 
    wind and price trajectories to make decisions.
    
    Args:
        current_time: Current time slot
        electrolyzer_status: Current electrolyzer status (0=off, 1=on)
        hydrogen_level: Current hydrogen storage level
        wind_power: Current wind power generation
        grid_price: Current electricity price
        demand: Current electricity demand
        data: Problem data dictionary
        previous_wind: Previous wind power generation
        previous_price: Previous electricity price
        lookahead_horizon: Number of time periods to look ahead
        num_samples: Number of samples for expected value calculation
        
    Returns:
        electrolyzer_on: Decision to turn electrolyzer on (0 or 1)
        electrolyzer_off: Decision to turn electrolyzer off (0 or 1)
        p_grid: Power drawn from the grid
        p_p2h: Power converted to hydrogen
        p_h2p: Hydrogen converted to power
    """
    import pyomo.environ as pyo
    from utils.WindProcess import wind_model
    from utils.PriceProcess import price_model
    
    # Use provided previous wind and price if available, otherwise use estimates
    if previous_wind is None:
        previous_wind = wind_power * 0.95  # Simple estimate
    if previous_price is None:
        previous_price = grid_price * 0.98  # Simple estimate
    
    # Generate expected future trajectories
    horizon = min(lookahead_horizon, data['num_timeslots'] - current_time)
    
    # Initialize arrays for expected trajectories
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
    
    # Create and solve deterministic optimization model
    model = pyo.ConcreteModel()
    
    # Define sets
    model.T = pyo.RangeSet(0, horizon-1)  # Time periods (0 = here-and-now)
    
    # Define parameters
    def wind_param_init(model, t):
        return expected_wind[t]
    
    def price_param_init(model, t):
        return expected_price[t]
    
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
    
    # Define objective function
    def obj_rule(model):
        return sum(model.price[t] * model.p_grid[t] + 
                   data['electrolyzer_cost'] * model.x[t] 
                   for t in model.T)
    
    model.objective = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
    
    # Define constraints (same as in stochastic programming policy)
    
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
    solver.options['TimeLimit'] = 30  # Set a time limit
    
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
            # Fall back to a simple decision rule if optimization fails
            electrolyzer_on = 0
            electrolyzer_off = 0
            p_grid = max(0, demand - wind_power)
            p_p2h = 0
            p_h2p = 0
    except Exception as e:
        print(f"Error in expected value optimization: {e}")
        # Fall back to a simple decision rule
        electrolyzer_on = 0
        electrolyzer_off = 0
        p_grid = max(0, demand - wind_power)
        p_p2h = 0
        p_h2p = 0
    
    return electrolyzer_on, electrolyzer_off, p_grid, p_p2h, p_h2p


# ===================================================
# || Helper functions for creating policies        ||
# ===================================================

def create_sp_policy(horizon: int, branches_per_stage: int, samples: int = 100):
    """
    Creates a stochastic programming policy with specified parameters.
    """
    def policy(current_time, electrolyzer_status, hydrogen_level, wind_power, 
               grid_price, demand, data, previous_wind=None, previous_price=None
               ,*args, **kwargs):
        return stochastic_programming_policy(
            current_time, electrolyzer_status, hydrogen_level, wind_power, 
            grid_price, demand, data, previous_wind, previous_price,
            lookahead_horizon=horizon, branches_per_stage=branches_per_stage
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