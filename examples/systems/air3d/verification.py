import torch
from dreal import And, Not, CheckSatisfiability
import json
import logging
import numpy as np

logger = logging.getLogger(__name__)

def dreal_air3d_BRS(dreal_partials, dreal_variables, epsilon=0.5, 
                    reachMode='forward', reachAim='reach', 
                    setType='set', save_directory="./",
                    velocity=0.6, omega_max=1.1):
    """Verifies if the HJB equation holds using dReal for Air3D system."""
    
    # Extract variables
    t = dreal_variables["x_1_1"]
    x = dreal_variables["x_1_2"]
    y = dreal_variables["x_1_3"]
    theta = dreal_variables["x_1_4"]

    dv_dt = dreal_partials["partial_x_1_1"]
    p_x = dreal_partials["partial_x_1_2"]
    p_y = dreal_partials["partial_x_1_3"]
    p_theta = dreal_partials["partial_x_1_4"]

    # Define Hamiltonian for air3d dynamics
    # dx/dt = v*cos(theta), dy/dt = v*sin(theta), dtheta/dt = omega
    hamiltonian = (p_x * velocity * dreal.cos(theta) + 
                  p_y * velocity * dreal.sin(theta))

    # Modify based on reachAim
    if reachAim == 'avoid':
        hamiltonian -= abs(p_theta) * omega_max  # Maximize angular velocity
    elif reachAim == 'reach':
        hamiltonian += abs(p_theta) * omega_max  # Minimize angular velocity

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
        x >= -1, x <= 1,
        y >= -1, y <= 1,
        theta >= -np.pi, theta <= np.pi  # Periodic boundary
    )

    all_constraints = And(final_condition, state_constraints)
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
