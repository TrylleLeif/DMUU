# ============================================
# || import all necessary libraries         ||
# ============================================

import numpy as np
import matplotlib.pyplot as plt
import random
from typing import Dict, List, Tuple, Any

from pyomo.environ import (ConcreteModel, Var, Binary, NonNegativeReals, Set, 
                         Objective, Constraint, SolverFactory, TerminationCondition, 
                         minimize, value, Param)

from utils.data import get_fixed_data
from utils.WindProcess import wind_model
from utils.PriceProcess import price_model

# ============================================
# || Generate wind and price trajectories   ||
# ============================================

def generate_trajectories(data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    num_timeslots = data['num_timeslots']
    wind_trajectory = np.zeros(num_timeslots)
    price_trajectory = np.zeros(num_timeslots)
    
    # init first two values
    wind_trajectory[0] = data['target_mean_wind']
    wind_trajectory[1] = data['target_mean_wind']
    price_trajectory[0] = data['mean_price']
    price_trajectory[1] = data['mean_price']
    
    # Generate trajectories
    for t in range(2, num_timeslots):
        wind_trajectory[t] = wind_model(wind_trajectory[t-1], wind_trajectory[t-2], data)
        price_trajectory[t] = price_model(price_trajectory[t-1], price_trajectory[t-2], 
                                        wind_trajectory[t], data)
    
    return wind_trajectory, price_trajectory

# ============================================
# || Solve MILP model                       ||
# || with given wind and price trajectories ||
# ============================================

def solve_milp(wind_trajectory: np.ndarray, price_trajectory: np.ndarray, data: Dict[str, Any], obj: bool): #-> Dict[str, List[float]]:
    m = ConcreteModel()
    
    # Sets and parameters 
    m.T = Set(initialize=range(data['num_timeslots']))
    m.D = Param(m.T, initialize=lambda m, t: data['demand_schedule'][t])
    
    # Variables
    m.x = Var(m.T, within=Binary, initialize=0)
    m.y_on = Var(m.T, within=Binary, initialize=0)
    m.y_off = Var(m.T, within=Binary, initialize=0)
    m.h = Var(m.T, within=NonNegativeReals, initialize=0)
    m.p_grid = Var(m.T, within=NonNegativeReals)
    m.p_p2h = Var(m.T, within=NonNegativeReals)
    m.p_h2p = Var(m.T, within=NonNegativeReals)

    # Constants from data
    R_p2h = data['conversion_p2h']
    R_h2p = data['conversion_h2p']
    C = data['hydrogen_capacity']
    P2H_max = data['p2h_max_rate']
    H2P_max = data['h2p_max_rate']
    C_elzr = data['electrolyzer_cost']
    
    # Objective function
    m.cost = Objective(
        expr=sum(price_trajectory[t] * m.p_grid[t] + C_elzr * m.x[t] 
                 for t in m.T),
        sense=minimize
    )
    
    # Power balance constraint
    def power_balance_rule(m, t):
        return (wind_trajectory[t] + m.p_grid[t] + 
                R_h2p * m.p_h2p[t] - m.p_p2h[t] >= m.D[t])
    m.power_balance = Constraint(m.T, rule=power_balance_rule)
    
    # Power-to-hydrogen limit constraint
    def p2h_limit_rule(m, t):
        return m.p_p2h[t] <= P2H_max * m.x[t]
    m.p2h_limit = Constraint(m.T, rule=p2h_limit_rule)
    
    # Hydrogen-to-power limit constraint
    def h2p_limit_rule(m, t):
        return m.p_h2p[t] <= H2P_max
    m.h2p_limit = Constraint(m.T, rule=h2p_limit_rule)
    
    # Storage balance constraint
    def storage_balance_rule(m, t):
        if t == 0:
            return m.h[t] == 0
        return m.h[t] == (m.h[t-1] + (R_p2h * m.p_p2h[t-1]) * m.x[t-1] - m.p_h2p[t-1])
    m.storage_balance = Constraint(m.T, rule=storage_balance_rule)
    
    # Storage capacity constraint
    def storage_capacity_rule(m, t):
        return m.h[t] <= C
    m.storage_capacity = Constraint(m.T, rule=storage_capacity_rule)
    

# make it into two constraints 

# binary variable x[t] 
# y on 
# y off 
# yon+yoff <= 1
# if electo if of i cant turn of 

    def electrolyzer_rule(m, t):
        return m.y_on[t] + m.y_off[t] <= 1
    m.electrolyzer = Constraint(m.T, rule=electrolyzer_rule)

    def electrolyzer_on_rule(m, t):
        return m.y_on[t] <= m.x[t]
    m.electrolyzer_on = Constraint(m.T, rule=electrolyzer_on_rule)

    def electrolyzer_off_rule(m, t):
        return m.x[t] <= 1 - m.y_off[t]
    m.electrolyzer_off = Constraint(m.T, rule=electrolyzer_off_rule)

    # # State transition constraint
    # def state_transition_rule(m, t):
    #     if t == 0:
    #         return m.x[t] == 0
    #     return m.x[t] == m.x[t-1] + m.y[t-1] - 2 * m.x[t-1] * m.y[t-1]
    # m.state_transition = Constraint(m.T, rule=state_transition_rule)
    
    # Initialize y[0] = 0
    def y_on_init_rule(m, t):
        if t == 0:
            return m.y_on[t] == 0
        return Constraint.Skip
    m.y_on_init = Constraint(m.T, rule=y_on_init_rule)
    def y_off_init_rule(m, t):
        if t == 0:
            return m.y_off[t] == 0
        return Constraint.Skip
    m.y_off_init = Constraint(m.T, rule=y_off_init_rule)
    
    # Solve
    solver = SolverFactory('gurobi')
    results = solver.solve(m)
    if obj:
        return round(value(m.cost),ndigits=2)
    else:
        if results.solver.termination_condition == TerminationCondition.optimal:
            print(f"Optimal cost: {round(value(m.cost),ndigits=2)}")
            return {
                'electrolyzer_status': [value(m.x[t]) for t in m.T],
                'hydrogen_storage_level': [value(m.h[t]) for t in m.T],
                'power_to_hydrogen': [value(m.p_p2h[t]) for t in m.T],
                'hydrogen_to_power': [value(m.p_h2p[t]) for t in m.T],
                'grid_power': [value(m.p_grid[t]) for t in m.T]
            }
        else:
            raise Exception("No optimal solution found")
