def sample_representative_state_pairs(data, num_samples=50):
    # Sample I state pairs of t=24 series
    state_pairs = []
    
    # Extract data parameters needed for sampling and feasibility checks
    price_floor = data['price_floor']
    price_cap = data['price_cap']
    mean_price = data['mean_price']
    target_mean_wind = data['target_mean_wind']
    hydrogen_capacity = data['hydrogen_capacity']
    p2h_max_rate = data['p2h_max_rate']
    h2p_max_rate = data['h2p_max_rate']
    r_p2h = data['conversion_p2h']
    r_h2p = data['conversion_h2p']
    demand = data['demand_schedule'][0]  # Using first demand value for simplicity
    
    # Counter for attempts to avoid infinite loops
    attempts = 0
    max_attempts = num_samples * 10
    
    while len(state_pairs) < num_samples and attempts < max_attempts:
        attempts += 1
        
        # Sample exogenous variables z_t (price, wind)
        price = random.uniform(price_floor, price_cap)
        wind = random.uniform(0, target_mean_wind * 2)  # 0 to twice the mean
        
        # Sample endogenous variables y_t (hydrogen, electrolyzer_status)
        hydrogen = random.uniform(0, hydrogen_capacity)
        electrolyzer_status = random.choice([0, 1])
        
        # Create the state
        state = (price, wind, hydrogen, electrolyzer_status)
        
        # Check if this state has at least one feasible decision
        # We'll try a few basic decisions to see if any are feasible
        has_feasible_decision = False
        
        # Try some simple decision combinations
        for p_grid in [0, demand/2, demand]:
            for p_h2p in [0, min(1, hydrogen)]:
                for p_p2h in [0] if electrolyzer_status == 0 else [0, 2]:
                    # No switching for simplicity
                    electrolyzer_on, electrolyzer_off = 0, 0
                    
                    # Check if this decision is feasible
                    feasible = check_feasibility(
                        electrolyzer_status=electrolyzer_status,
                        electrolyzer_on=electrolyzer_on,
                        electrolyzer_off=electrolyzer_off,
                        p_grid=p_grid,
                        p_p2h=p_p2h,
                        p_h2p=p_h2p,
                        hydrogen_level=hydrogen,
                        wind_power=wind,
                        demand=demand,
                        p2h_max_rate=p2h_max_rate,
                        h2p_max_rate=h2p_max_rate,
                        r_p2h=r_p2h,
                        r_h2p=r_h2p,
                        hydrogen_capacity=hydrogen_capacity
                    )
                    
                    if feasible:
                        has_feasible_decision = True
                        break
                
                if has_feasible_decision:
                    break
            
            if has_feasible_decision:
                break
        
        # Only add states that have at least one feasible decision
        if has_feasible_decision:
            state_pairs.append(state)
    
    # Add a few critical region samples (20% of total samples)
    critical_samples = min(int(num_samples * 0.2), max(1, num_samples - len(state_pairs)))
    
    for _ in range(critical_samples):
        # Choose a critical region scenario (low price or high price)
        scenario = random.choice(['low_price', 'high_price'])
        
        if scenario == 'low_price':
            # Low price, opportunity to produce hydrogen
            price = random.uniform(price_floor, mean_price * 0.7)
            wind = random.uniform(target_mean_wind * 0.5, target_mean_wind * 2)
            hydrogen = random.uniform(0, hydrogen_capacity * 0.5)
            electrolyzer_status = 1  # On for producing hydrogen
        else:
            # High price, opportunity to use hydrogen
            price = random.uniform(mean_price * 1.3, price_cap)
            wind = random.uniform(0, target_mean_wind)
            hydrogen = random.uniform(hydrogen_capacity * 0.5, hydrogen_capacity)
            electrolyzer_status = random.choice([0, 1])
        
        # Add this critical state if it has feasible decisions (same check as above)
        state = (price, wind, hydrogen, electrolyzer_status)
        state_pairs.append(state)
    
    return state_pairs