import torch
import json
import logging
import numpy as np
from typing import Dict, Any, Union, Optional

# Import dReal conditionally to allow module import without dReal installed
try:
    import dreal
    from dreal import And, Not, CheckSatisfiability, Variable, sin, cos, abs
    DREAL_AVAILABLE = True
except ImportError:
    DREAL_AVAILABLE = False

logger = logging.getLogger(__name__)

def verify_air3d_hjb(
    dreal_partials: Dict[str, Any], 
    dreal_variables: Dict[str, Any], 
    epsilon: float = 0.5, 
    reachMode: str = 'forward',
    reachAim: str = 'reach', 
    setType: str = 'set',
    save_directory: str = "./",
    velocity: float = 0.6, 
    omega_max: float = 1.1
) -> Dict[str, Any]:
    """
    Verifies if the HJB equation holds using dReal for Air3D system.
    
    Args:
        dreal_partials: Dictionary of partial derivatives from the neural network
        dreal_variables: Dictionary of dReal variables for the input coordinates
        epsilon: Verification epsilon
        reachMode: 'forward' or 'backward' reachability
        reachAim: 'reach' or 'avoid' set computation
        setType: 'set' or 'tube' computation
        save_directory: Directory to save verification results
        velocity: Velocity of the system
        omega_max: Maximum angular velocity
        
    Returns:
        Dictionary with verification results
    """
    if not DREAL_AVAILABLE:
        raise ImportError("dReal is not available. Please install dReal to use this function.")
    
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
    hamiltonian = (p_x * velocity * cos(theta) + 
                  p_y * velocity * sin(theta))

    # Modify based on reachAim
    if reachAim == 'avoid':
        hamiltonian -= abs(p_theta) * omega_max  # Maximize angular velocity
    elif reachAim == 'reach':
        hamiltonian += abs(p_theta) * omega_max  # Minimize angular velocity

    if reachMode == 'backward':
        hamiltonian = -hamiltonian

    # Define HJB equation and conditions to check
    condition_1 = abs(dv_dt + hamiltonian) <= epsilon
    condition_2 = abs(dv_dt) <= epsilon

    if setType == 'tube':
        final_condition = Not(And(condition_1, condition_2))
    else:
        final_condition = Not(condition_1)

    # Define state space bounds
    state_constraints = And(
        t >= 0, t <= 1,
        x >= -1.5, x <= 1.5,
        y >= -1.5, y <= 1.5,
        theta >= -np.pi, theta <= np.pi
    )

    all_constraints = And(final_condition, state_constraints)
    result = CheckSatisfiability(all_constraints, 1e-4)

    # Save results
    result_data = {
        "epsilon": epsilon,
        "set": f"{reachMode}_{reachAim}_{setType}",
        "velocity": velocity,
        "omega_max": omega_max,
        "result": str(result) if result else "HJB Equation Satisfied"
    }

    if result:
        # Extract counterexample if verification failed
        model = result.model()
        counterexample = {
            "t": model[t], 
            "x": model[x], 
            "y": model[y], 
            "theta": model[theta]
        }
        result_data["counterexample"] = counterexample
        logger.info(f"Verification failed. Counterexample: {counterexample}")
    else:
        logger.info(f"Verification succeeded with epsilon={epsilon}")

    result_file = f"{save_directory}/dreal_result.json"
    with open(result_file, "w") as f:
        json.dump(result_data, f, indent=4)

    logger.info(f"Saved verification result to {result_file}")
    return result_data
