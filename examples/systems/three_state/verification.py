import torch
from dreal import And, Not, CheckSatisfiability
import json
import logging

logger = logging.getLogger(__name__)

def dreal_three_state_system_BRS(dreal_partials, dreal_variables, epsilon=0.5, 
                                reachMode='forward', reachAim='reach', 
                                setType='set', save_directory="./",
                                k1=1.0, k2=1.0, c1=0.5, c2=0.5, u_max=1.0):
    """Verifies if the HJB equation holds using dReal for three state system."""
    
    # Extract variables
    t = dreal_variables["x_1_1"]
    x_1 = dreal_variables["x_1_2"]
    x_2 = dreal_variables["x_1_3"]
    x_3 = dreal_variables["x_1_4"]

    dv_dt = dreal_partials["partial_x_1_1"]
    p_1 = dreal_partials["partial_x_1_2"]
    p_2 = dreal_partials["partial_x_1_3"]
    p_3 = dreal_partials["partial_x_1_4"]

    # Define Hamiltonian based on reachAim
    base_terms = (p_1 * x_2
                 - k1 * p_2 * x_1
                 - c1 * p_2 * x_2
                 - k2 * p_3 * x_3
                 + c2 * p_3 * x_1)

    if reachAim == 'avoid':
        hamiltonian = base_terms + abs(p_2) * u_max  # Maximize control
    else:
        hamiltonian = base_terms - abs(p_2) * u_max  # Minimize control

    if reachMode == 'backward':
        hamiltonian = -hamiltonian

    # Define constraints
    condition_1 = abs(dv_dt + hamiltonian) <= epsilon
    condition_2 = abs(dv_dt) <= epsilon

    if setType == 'tube':
        final_condition = Not(And(condition_1, condition_2))
    else:
        final_condition = Not(condition_1)

    state_constraints = And(
        t >= 0, t <= 1,
        x_1 >= -1, x_1 <= 1,
        x_2 >= -1, x_2 <= 1,
        x_3 >= -1, x_3 <= 1
    )

    all_constraints = And(final_condition, state_constraints)

    # Check constraints with larger delta for speed
    result = CheckSatisfiability(all_constraints, 1e-2)

    # Save results
    result_data = {
        "epsilon": epsilon,
        "set": f"{reachMode}_{reachAim}_{setType}", 
        "result": str(result) if result else "HJB Equation Satisfied"
    }

    result_file = f"{save_directory}/dreal_result.json"
    with open(result_file, "w") as f:
        json.dump(result_data, f, indent=4)

    logger.info(f"Saved result to {result_file}")
    return result_data
