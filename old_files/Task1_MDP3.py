# ============================================
# || MDP Formulation & Policy Evaluation    ||
# ============================================

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Callable, Optional
import random

from DMUU.utils.data import get_fixed_data
from DMUU.utils.WindProcess import wind_model
from DMUU.utils.PriceProcess import price_model

class EnvError(Exception):
    """Custom exception for environment errors."""
    pass

class EnergyHubMDP:
    """
    Markov Decision Process environment for the Energy Hub problem.
    """
    # =============================================
    # Deliverable 1: MDP Formulation
    # =============================================
    # State variables x_t = {x_{1,t}, x_{2,t}, ...}:
    # - wind_previous: Previous wind generation 
    # - wind_current: Current wind generation
    # - price_previous: Previous price
    # - price_current: Current price
    # - hydrogen_level: Hydrogen storage level
    # - electrolyzer_status: Electrolyzer ON/OFF
    # - timeslot: Current time period
    #
    # Decision variables u_t = {u_{1,t}, u_{2,t}, ...}:
    # - y_on: Decision to turn electrolyzer ON
    # - y_off: Decision to turn electrolyzer OFF
    # - p2h: Amount of power to convert to hydrogen
    # - h2p: Amount of hydrogen to convert to power
    #
    # Electrolyzer transition rules:
    # - y_on + y_off <= 1 (can't decide to turn ON and OFF simultaneously)
    # - y_on <= x (can only turn ON if electrolyzer is currently OFF)
    # - x <= 1 - y_off (can only turn OFF if electrolyzer is currently ON)
    # - At t=0: y_on = 0, y_off = 0, x = 0 (initial conditions)
    #
    # Dynamics (State Transition Function) x_{t+1} = f(x_t, u_t):
    # - hydrogen_level_{t+1} = min(C, hydrogen_level_t - h2p_t + R_p2h * p2h_t)
    # - electrolyzer_status_{t+1} = electrolyzer_status_t + y_on_t - y_off_t
    # - wind and price follow stochastic processes
    #
    # Cost Function c_t = g(x_t, u_t):
    # - grid_power = max(0, demand_t + p2h_t - wind_t - R_h2p * h2p_t)
    # - cost = price_t * grid_power + C_elzr * electrolyzer_status_t
    
    def __init__(self, data: Dict[str, Any]):
        """
        Initialize the MDP environment with the given data.
        
        Args:
            data (Dict[str, Any]): Fixed data containing model parameters.
        """
        self.data = data
        self.num_timeslots = data['num_timeslots']
        
        # Constants from data (immutable parameters)
        self.R_p2h = data['conversion_p2h']
        self.R_h2p = data['conversion_h2p']
        self.C = data['hydrogen_capacity']
        self.P2H_max = data['p2h_max_rate']
        self.H2P_max = data['h2p_max_rate']
        self.C_elzr = data['electrolyzer_cost']
        
        # Initialize environment 
        self.reset()
    
    def reset(self) -> Dict[str, Any]:
        """
        Reset the environment to the initial state.
        
        Returns:
            Dict[str, Any]: Initial observation for the policy.
        """
        # Internal environment state
        self._wind_previous = self.data['target_mean_wind']
        self._wind_current = self.data['target_mean_wind']
        self._price_previous = self.data['mean_price']
        self._price_current = self.data['mean_price']
        self._hydrogen_level = 0.0
        self._electrolyzer_status = 0  # 0 = OFF, 1 = ON
        self._timeslot = 0
        self._done = False
        
        # Generate full trajectories ahead of time
        self._wind_trajectory = np.zeros(self.num_timeslots)
        self._price_trajectory = np.zeros(self.num_timeslots)
        self._wind_trajectory[0] = self._wind_current
        self._price_trajectory[0] = self._price_current
        self._generate_trajectories()
        
        # Return initial observation for the policy
        return self._get_observation()
    
    def _generate_trajectories(self) -> None:
        """
        Generate wind and price trajectories for the current episode.
        Internal method used by reset().
        """
        self._wind_trajectory[0] = self._wind_current
        self._price_trajectory[0] = self._price_current
        
        # Generate trajectories for the remaining timeslots
        for t in range(1, self.num_timeslots):
            prev_idx = max(0, t-2)
            self._wind_trajectory[t] = wind_model(
                self._wind_trajectory[t-1], 
                self._wind_trajectory[prev_idx], 
                self.data
            )
            self._price_trajectory[t] = price_model(
                self._price_trajectory[t-1], 
                self._price_trajectory[prev_idx], 
                self._wind_trajectory[t], 
                self.data
            )
    
    def _get_observation(self) -> Dict[str, Any]:
        """
        Get the current observation for the policy.
        
        Returns:
            Dict[str, Any]: Observation dictionary for policy input.
            
        Raises:
            EnvError: If attempting to get observation after episode is done.
        """
        if self._done:
            raise EnvError("Cannot get observation after episode is done.")
            
        return {
            'wind_previous': self._wind_previous,
            'wind_current': self._wind_current,
            'price_previous': self._price_previous,
            'price_current': self._price_current,
            'hydrogen_level': self._hydrogen_level,
            'electrolyzer_status': self._electrolyzer_status,
            'timeslot': self._timeslot,
            'demand': self.data['demand_schedule'][self._timeslot]
        }
    
    # =============================================
    # System Constraints
    # =============================================
    def check_electrolyzer_transition_constraints(self, y_on: int, y_off: int) -> Tuple[bool, str, Dict[str, int]]:
        """
        Check electrolyzer state transition constraints and return feasibility, error message, and corrected values.
        
        Args:
            y_on (int): Decision to turn electrolyzer ON (0 or 1)
            y_off (int): Decision to turn electrolyzer OFF (0 or 1)
            
        Returns:
            Tuple[bool, str, Dict[str, int]]: 
                - Boolean indicating if constraints are satisfied
                - Error message (empty if feasible)
                - Dictionary with corrected values (may be adjusted to meet constraints)
        """
        corrected_values = {
            'y_on': y_on,
            'y_off': y_off
        }
        
        # Constraint 1: At t=0, initialize y_on and y_off to 0
        if self._timeslot == 0:
            corrected_values['y_on'] = 0
            corrected_values['y_off'] = 0
            return True, "Initial time step: y_on and y_off set to 0", corrected_values
        
        # Constraint 2: y_on + y_off <= 1 (can't decide to turn ON and OFF simultaneously)
        if y_on + y_off > 1:
            if self._electrolyzer_status == 1:
                # If already ON, prefer turning OFF
                corrected_values['y_on'] = 0
                corrected_values['y_off'] = 1
                return True, "Cannot turn ON and OFF simultaneously: preference given to turning OFF", corrected_values
            else:
                # If already OFF, prefer turning ON
                corrected_values['y_on'] = 1
                corrected_values['y_off'] = 0
                return True, "Cannot turn ON and OFF simultaneously: preference given to turning ON", corrected_values
        
        # Constraint 3: y_on <= (1 - electrolyzer_status) (can only turn ON if currently OFF)
        if y_on > 0 and self._electrolyzer_status == 1:
            corrected_values['y_on'] = 0
            return True, "Cannot turn ON when electrolyzer is already ON", corrected_values
        
        # Constraint 4: y_off <= electrolyzer_status (can only turn OFF if currently ON)
        if y_off > 0 and self._electrolyzer_status == 0:
            corrected_values['y_off'] = 0
            return True, "Cannot turn OFF when electrolyzer is already OFF", corrected_values
        
        # All constraints satisfied with possibly corrected values
        return True, "", corrected_values
    
    def check_constraints(self, 
                         electrolyzer_status: int, 
                         p_grid: float, 
                         p_p2h: float, 
                         p_h2p: float, 
                         hydrogen_level: float, 
                         wind_power: float, 
                         demand: float) -> Tuple[bool, str, Dict[str, float]]:
        """
        Check all system constraints and return feasibility, error message, and corrected values.
        
        This function centralizes all constraint checking in one place for better overview.
        
        Args:
            electrolyzer_status (int): Current electrolyzer status (0=OFF, 1=ON)
            p_grid (float): Power drawn from the grid
            p_p2h (float): Power converted to hydrogen
            p_h2p (float): Hydrogen converted to power
            hydrogen_level (float): Current hydrogen storage level
            wind_power (float): Available wind power
            demand (float): Power demand to be satisfied
            
        Returns:
            Tuple[bool, str, Dict[str, float]]: 
                - Boolean indicating if all constraints are satisfied
                - Error message (empty if feasible)
                - Dictionary with corrected values (may be adjusted to meet constraints)
        """
        corrected_values = {
            'p_grid': p_grid,
            'p_p2h': p_p2h,
            'p_h2p': p_h2p
        }
        
        # 1. Check power-to-hydrogen limit: p_p2h <= P2H_max * electrolyzer_status
        if p_p2h > self.P2H_max * electrolyzer_status:
            if electrolyzer_status == 0:
                corrected_values['p_p2h'] = 0
                message = f"P2H limit constraint adjusted: Electrolyzer is OFF, p2h set to 0"
            else:
                corrected_values['p_p2h'] = self.P2H_max
                message = f"P2H limit constraint adjusted: p2h reduced from {p_p2h} to {self.P2H_max}"
            p_p2h = corrected_values['p_p2h']  # Update for subsequent checks
        
        # 2. Check hydrogen-to-power limit: p_h2p <= H2P_max
        if p_h2p > self.H2P_max:
            corrected_values['p_h2p'] = self.H2P_max
            message = f"H2P limit constraint adjusted: h2p reduced from {p_h2p} to {self.H2P_max}"
            p_h2p = corrected_values['p_h2p']  # Update for subsequent checks
        
        # 3. Check hydrogen availability: p_h2p <= hydrogen_level
        if p_h2p > hydrogen_level:
            corrected_values['p_h2p'] = hydrogen_level
            message = f"Hydrogen availability constraint adjusted: h2p reduced from {p_h2p} to {hydrogen_level}"
            p_h2p = corrected_values['p_h2p']  # Update for subsequent checks
        
        # 4. Check power balance: wind + grid + h2p_power - p2h >= demand
        h2p_power = self.R_h2p * p_h2p
        available_power = wind_power + h2p_power + p_grid
        
        if available_power < demand + p_p2h:
            # Increase grid power to meet demand
            additional_power_needed = (demand + p_p2h) - available_power
            corrected_values['p_grid'] = p_grid + additional_power_needed
            message = f"Power balance constraint adjusted: grid power increased from {p_grid} to {corrected_values['p_grid']}"
        
        # 5. Final verification of power balance with corrected values
        h2p_power = self.R_h2p * corrected_values['p_h2p']
        available_power = wind_power + h2p_power + corrected_values['p_grid']
        
        if available_power < demand + corrected_values['p_p2h']:
            return False, f"Power balance constraint violated: Available power ({available_power}) < Required power ({demand + corrected_values['p_p2h']})", corrected_values
        
        # All constraints are satisfied (with possibly corrected values)
        return True, "", corrected_values
    
    def _validate_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize the action to ensure it's feasible.
        
        Args:
            action (Dict[str, Any]): Raw action from policy.
            
        Returns:
            Dict[str, Any]: Validated and constrained action.
            
        Raises:
            EnvError: If action is invalid or episode is already done.
        """
        if self._done:
            raise EnvError("Cannot take action after episode is done.")
            
        if not isinstance(action, dict):
            raise EnvError(f"Action must be a dictionary, got {type(action)}")
            
        # Extract action components with defaults
        validated_action = {
            'y_on': int(bool(action.get('y_on', 0))),      # Decision to turn electrolyzer ON
            'y_off': int(bool(action.get('y_off', 0))),    # Decision to turn electrolyzer OFF
            'p2h': float(action.get('p2h', 0.0)),          # Power to hydrogen
            'h2p': float(action.get('h2p', 0.0))           # Hydrogen to power
        }
        
        return validated_action
    
    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Take a step in the environment using the given action.
        
        Args:
            action (Dict[str, Any]): Action to take. Contains:
                - y_on: Decision to turn electrolyzer ON (0 or 1)
                - y_off: Decision to turn electrolyzer OFF (0 or 1)
                - p2h: Amount of power to convert to hydrogen
                - h2p: Amount of hydrogen to convert to power
        
        Returns:
            Tuple[Dict[str, Any], float, bool, Dict[str, Any]]: 
                - Observation for the next state
                - Reward (negative cost)
                - Done flag
                - Info dictionary with additional information
                
        Raises:
            EnvError: If action is invalid or episode is already done.
        """
        if self._done:
            raise EnvError("Episode is already done. Call reset() to start a new episode.")
            
        # Get basic action values (not yet validated against constraints)
        validated_action = self._validate_action(action)
        y_on = validated_action['y_on']
        y_off = validated_action['y_off']
        p2h = validated_action['p2h']
        h2p = validated_action['h2p']
        
        # Check electrolyzer transition constraints
        _, transition_message, corrected_transitions = self.check_electrolyzer_transition_constraints(y_on, y_off)
        y_on = corrected_transitions['y_on']
        y_off = corrected_transitions['y_off']
        
        # Get current demand, wind, and price for this timeslot
        demand = self.data['demand_schedule'][self._timeslot]
        wind = self._wind_current
        price = self._price_current
        
        # Calculate initial grid power needed (before constraint validation)
        power_available = wind + self.R_h2p * h2p
        power_needed = demand + p2h
        grid_power = max(0, power_needed - power_available)
        
        # Check all constraints and get corrected values
        feasible, error_message, corrected_values = self.check_constraints(
            electrolyzer_status=self._electrolyzer_status,
            p_grid=grid_power,
            p_p2h=p2h,
            p_h2p=h2p,
            hydrogen_level=self._hydrogen_level,
            wind_power=wind,
            demand=demand
        )
        
        if not feasible:
            raise EnvError(error_message)
        
        # Use corrected values for actions
        grid_power = corrected_values['p_grid']
        p2h = corrected_values['p_p2h']
        h2p = corrected_values['p_h2p']
        
        # Calculate cost for the current step
        step_cost = price * grid_power + self.C_elzr * self._electrolyzer_status
        
        # Update hydrogen level
        new_hydrogen_level = self._hydrogen_level - h2p + self.R_p2h * p2h
        # Apply hydrogen capacity constraint
        self._hydrogen_level = min(self.C, new_hydrogen_level)
        
        # Advance to next timeslot
        self._timeslot += 1
        
        # Check if this was the final time step
        if self._timeslot >= self.num_timeslots:
            self._done = True
            
            # Create final info dictionary
            info = {
                'final_hydrogen_level': self._hydrogen_level,
                'total_time_steps': self._timeslot,
                'final_step': True,
                'grid_power': grid_power,
                'p2h_used': p2h,
                'h2p_used': h2p,
                'y_on': y_on,
                'y_off': y_off
            }
            
            # Return final observation from the step before the end
            # Note: Since we're done, we don't need to update the environment state further
            # and we don't need to provide a new observation since it won't be used
            return {}, -step_cost, True, info
        
        # Update electrolyzer status based on transition decisions
        # electrolyzer_status_{t+1} = electrolyzer_status_t + y_on_t - y_off_t
        next_electrolyzer_status = self._electrolyzer_status + y_on - y_off
        
        # Ensure status remains binary (should always be the case with proper constraints)
        self._electrolyzer_status = min(1, max(0, next_electrolyzer_status))
        
        # Update wind and price states for next timeslot
        self._wind_previous = self._wind_current
        self._wind_current = self._wind_trajectory[self._timeslot]
        self._price_previous = self._price_current
        self._price_current = self._price_trajectory[self._timeslot]
        
        # Create info dictionary for the step
        info = {
            'grid_power': grid_power,
            'demand': demand,
            'wind': wind,
            'price': price,
            'hydrogen_level': self._hydrogen_level,
            'p2h_used': p2h,
            'h2p_used': h2p,
            'y_on': y_on,
            'y_off': y_off,
            'transition_message': transition_message if transition_message else "Valid transition"
        }
        
        # Return the new observation
        return self._get_observation(), -step_cost, False, info
    
    def run_episode(self, policy_fn: Callable) -> Tuple[float, Dict[str, List[float]]]:
        """
        Run a full episode using the given policy function.
        
        Args:
            policy_fn (Callable): Function that takes an observation and returns an action.
        
        Returns:
            Tuple[float, Dict[str, List[float]]]: Total cost and trajectory data
            
        Raises:
            EnvError: If policy returns invalid actions or constraints are violated.
        """
        observation = self.reset()
        total_cost = 0
        
        # Initialize trajectory data
        trajectory = {
            'wind': [self._wind_current],
            'price': [self._price_current],
            'hydrogen_level': [self._hydrogen_level],
            'electrolyzer_status': [self._electrolyzer_status],
            'y_on': [0],
            'y_off': [0],
            'p2h': [0],
            'h2p': [0],
            'grid_power': [0],
            'cost': [0]
        }
        
        done = False
        try:
            while not done:
                # Get action from policy
                action = policy_fn(observation)
                
                # Take step in environment
                next_observation, reward, done, info = self.step(action)
                
                # Update total cost (reward is negative cost)
                step_cost = -reward
                total_cost += step_cost
                
                # Record data for this step
                trajectory['wind'].append(info.get('wind', 0))
                trajectory['price'].append(info.get('price', 0))
                trajectory['hydrogen_level'].append(info.get('hydrogen_level', 0))
                trajectory['electrolyzer_status'].append(self._electrolyzer_status)
                trajectory['y_on'].append(info.get('y_on', 0))
                trajectory['y_off'].append(info.get('y_off', 0))
                trajectory['p2h'].append(info.get('p2h_used', 0))
                trajectory['h2p'].append(info.get('h2p_used', 0))
                trajectory['grid_power'].append(info.get('grid_power', 0))
                trajectory['cost'].append(step_cost)
                
                # Update observation
                observation = next_observation
                
        except EnvError as e:
            print(f"Environment error: {e}")
            # Even if there's an error, return what we have so far
        
        return total_cost, trajectory

# =============================================
# Deliverable 2: Policy Evaluation Framework
# =============================================
def evaluate_policy(policy_fn: Callable, num_experiments: int, data: Dict[str, Any], seed: int = 42) -> Tuple[float, List[float]]:
    """
    Evaluate a policy over multiple experiments.
    
    # Pseudocode:
    # =============================================
    # Input: policy (python function that returns decisions)
    # Initialize state variables
    # For experiment 1 to E:
    #   For stage 1 to H:
    #     decisions = policy(state)
    #     check/correct decisions if inconsistent
    #     calculate cost for this stage and experiment
    #     calculate state at next stage
    #   calculate total cost of policy for this experiment
    # Return: expected policy cost (average over experiments)
    # =============================================
    
    Args:
        policy_fn (Callable): Policy function that takes an observation and returns an action.
        num_experiments (int): Number of experiments to run.
        data (Dict[str, Any]): Fixed data containing model parameters.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
    
    Returns:
        Tuple[float, List[float]]: Average cost and list of costs for each experiment.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    costs = []
    
    # For experiment 1 to E:
    for i in range(num_experiments):
        # Create a fresh environment for each experiment
        env = EnergyHubMDP(data)
        
        try:
            # Initialize state variables (done in reset() inside run_episode)
            # For each stage 1 to H (done in the while loop inside run_episode):
            #   - Get decisions from policy
            #   - Check/correct decisions if inconsistent 
            #   - Calculate cost for this stage
            #   - Update state for next stage
            total_cost, _ = env.run_episode(policy_fn)
            costs.append(total_cost)
            print(f"Experiment {i+1}/{num_experiments}: Cost = {total_cost:.2f}")
        except Exception as e:
            print(f"Error in experiment {i+1}: {e}")
            # Skip this experiment if it fails
    
    if not costs:
        raise ValueError("All experiments failed. Check your policy function.")
        
    # Return expected policy cost (average over experiments)
    avg_cost = sum(costs) / len(costs)
    return avg_cost, costs

def dummy_policy(observation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dummy policy that never uses the electrolyzer.
    
    Args:
        observation (Dict[str, Any]): Current observation from the environment.
    
    Returns:
        Dict[str, Any]: Action to take.
    """
    return {
        'y_on': 0,     # Never turn electrolyzer ON
        'y_off': 1,    # Always decide to turn OFF if it's ON
        'p2h': 0,      # No power-to-hydrogen conversion
        'h2p': 0       # No hydrogen-to-power conversion
    }

def threshold_policy(observation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simple threshold policy that turns on electrolyzer when price is low and wind is high.
    
    Args:
        observation (Dict[str, Any]): Current observation from the environment.
    
    Returns:
        Dict[str, Any]: Action to take.
    """
    price = observation['price_current']
    wind = observation['wind_current']
    electrolyzer_status = observation['electrolyzer_status']
    hydrogen_level = observation['hydrogen_level']
    
    # Thresholds
    price_low = 30
    wind_high = 6
    hydrogen_high = 10
    
    # Decide on electrolyzer transitions
    y_on = 0
    y_off = 0
    
    if electrolyzer_status == 0 and price < price_low and wind > wind_high:
        # Turn ON when price is low and wind is high
        y_on = 1
    elif electrolyzer_status == 1 and (price > price_low or wind < wind_high):
        # Turn OFF when price is high or wind is low
        y_off = 1
    
    # Decide on p2h (only use if electrolyzer is ON)
    p2h = 0
    if electrolyzer_status == 1:
        p2h = min(wind - observation['demand'], 5)  # Use excess wind power, up to max rate
        p2h = max(0, p2h)  # Ensure non-negative
    
    # Decide on h2p (use when price is high and hydrogen is available)
    h2p = 0
    if price > 40 and hydrogen_level > 0:
        h2p = min(hydrogen_level, 5)  # Convert hydrogen to power, up to max rate
    
    return {
        'y_on': y_on,
        'y_off': y_off,
        'p2h': p2h,
        'h2p': h2p
    }

def plot_histogram(costs: List[float], title: str) -> None:
    """
    Plot a histogram of costs.
    
    Args:
        costs (List[float]): List of costs.
        title (str): Title for the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(costs, bins=10, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(sum(costs) / len(costs), color='red', linestyle='dashed', linewidth=2, label=f'Avg: {sum(costs) / len(costs):.2f}')
    plt.xlabel('Cost')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_trajectory(trajectory: Dict[str, List[float]], data: Dict[str, Any], title: str) -> None:
    """
    Plot the trajectory of a single episode.
    
    Args:
        trajectory (Dict[str, List[float]]): Trajectory data.
        data (Dict[str, Any]): Fixed data containing model parameters.
        title (str): Title for the plot.
    """
    times = range(data['num_timeslots'])
    
    plt.figure(figsize=(14, 10))
    plt.subplot(8, 1, 1)
    plt.plot(times, trajectory['wind'][:-1], label = "Wind Power", color = "blue")
    plt.ylabel("Wind Power")
    plt.legend()
    plt.title(title)
    
    plt.subplot(8, 1, 2)
    plt.plot(times, data['demand_schedule'], label = "Demand Schedule", color = "orange")
    plt.ylabel("Demand")
    plt.legend()
    
    plt.subplot(8, 1, 3)
    plt.step(times, trajectory['electrolyzer_status'][:-1], label = "Electrolyzer Status", color = "red", where = "post")
    plt.ylabel("El. Status")
    plt.legend()
    
    plt.subplot(8, 1, 4)
    plt.plot(times, trajectory['hydrogen_level'][:-1], label = "Hydrogen Level", color = "green")
    plt.ylabel("Hydr. Level")
    plt.legend()
    
    plt.subplot(8, 1, 5)
    plt.plot(times, trajectory['p2h'][:-1], label = "p2h", color = "orange")
    plt.ylabel("p2h")
    plt.legend()
    
    plt.subplot(8, 1, 6)
    plt.plot(times, trajectory['h2p'][:-1], label = "h2p", color = "blue")
    plt.ylabel("h2p")
    plt.legend()
    
    plt.subplot(8, 1, 7)
    plt.plot(times, trajectory['grid_power'][:-1], label = "Grid Power", color = "green")
    plt.ylabel("Grid Power")
    plt.legend()
    
    plt.subplot(8, 1, 8)
    plt.plot(times, trajectory['price'][:-1], label = "price", color = "red")
    plt.ylabel("Price")
    plt.xlabel("Time")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Get the fixed data
    data = get_fixed_data()
    
    # Number of experiments
    num_experiments = 20
    
    choosen_policy = dummy_policy#or threshold_policy

    # Evaluate the dummy policy
    print(f"\nEvaluating {choosen_policy} (never uses electrolyzer)...")
    avg_cost, costs = evaluate_policy(choosen_policy, num_experiments, data)
    print(f"\nDummy Policy - Average cost over {num_experiments} experiments: {avg_cost:.2f}")

    # Plot histograms
    plot_histogram(costs, f"{choosen_policy} Costs (Avg: {avg_cost:.2f})")
    
    # Run and plot trajectories
    # Dummy policy
    env = EnergyHubMDP(data)
    _, trajectory = env.run_episode(choosen_policy)
    plot_trajectory(trajectory, data, f"{choosen_policy} Trajectory")