import torch
from dreal import And, Not, CheckSatisfiability
import json
import logging

logger = logging.getLogger(__name__)

def dreal_double_integrator_BRS(dreal_partials, dreal_variables, epsilon=0.5, 
                              reachMode='forward', reachAim='reach', 
                              setType='set', save_directory="./"):
    """Verifies if the HJB equation holds using dReal for a double integrator system."""
    
    # Extract variables
    t = dreal_variables["x_1_1"]
    x_1 = dreal_variables["x_1_2"]
    x_2 = dreal_variables["x_1_3"]

    dv_dt = dreal_partials["partial_x_1_1"]
    p_1 = dreal_partials["partial_x_1_2"]
    p_2 = dreal_partials["partial_x_1_3"]

    # Define Hamiltonian
    hamiltonian = p_1 * x_2

    # Modify based on reachAim
    if reachAim == 'avoid':
        hamiltonian -= abs(p_2)
    elif reachAim == 'reach':
        hamiltonian += abs(p_2)

    if reachMode == 'backward':
        hamiltonian = -hamiltonian

    # Define constraints
    condition_1 = abs(dv_dt + hamiltonian) <= epsilon
    condition_2 = abs(dv_dt) <= epsilon

    if setType=='tube':
        final_condition = Not(And(condition_1, condition_2))
    else:
        final_condition = Not(condition_1)

    state_constraints = And(
        t >= 0, t <= 1,
        x_1 >= -1, x_1 <= 1,
        x_2 >= -1, x_2 <= 1
    )

    all_constraints = And(final_condition, state_constraints)

    # Check constraints
    result = CheckSatisfiability(all_constraints, 1e-5)

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
