from typing import Dict, Tuple, Any

# Dummy police, takes more input than needed so it is easier to replace it with a real policy
def dummy_policy(
    current_time: int,
    electrolyzer_status: int,
    hydrogen_level: float,
    wind_power: float,
    grid_price: float,
    demand: float,
    data: Dict[str, Any]
) -> Tuple[int, float, float, float]:

    electrolyzer_on = 0
    electrolyzer_off = 1 if electrolyzer_status == 1 else 0
    
    # Calculate the power needed from the grid
    p_grid = max(0, demand - wind_power)
    
    # No power to/from hydrogen
    p_p2h = 0
    p_h2p = 0
    
    return electrolyzer_on, electrolyzer_off, p_grid, p_p2h, p_h2p
