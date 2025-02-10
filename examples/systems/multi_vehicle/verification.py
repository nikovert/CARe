import torch
from dreal import And, Not, CheckSatisfiability, Variable
import json
import logging
import numpy as np

logger = logging.getLogger(__name__)

def dreal_multi_vehicle_BRS(dreal_partials, dreal_variables, epsilon=0.5, 
                          reachMode='forward', reachAim='reach', 
                          setType='set', save_directory="./",
                          num_vehicles=2, velocity=0.6, omega_max=1.1):
    """Verifies if the HJB equation holds using dReal for multi-vehicle system."""
    
    # Extract time variable
    t = dreal_variables["x_1_1"]
    
    # Extract state variables for each vehicle
    vehicles_states = []
    vehicles_partials = []
    
    for i in range(num_vehicles):
        base_idx = i * 3 + 2  # +2 because time is x_1_1
        # States: [x, y, theta] for each vehicle
        states = {
            'x': dreal_variables[f"x_1_{base_idx}"],
            'y': dreal_variables[f"x_1_{base_idx + 1}"],
            'theta': dreal_variables[f"x_1_{base_idx + 2}"]
        }
        # Partials: [dx/dt, dy/dt, dtheta/dt] for each vehicle
        partials = {
            'x': dreal_partials[f"partial_x_1_{base_idx}"],
            'y': dreal_partials[f"partial_x_1_{base_idx + 1}"],
            'theta': dreal_partials[f"partial_x_1_{base_idx + 2}"]
        }
        vehicles_states.append(states)
        vehicles_partials.append(partials)

    # Extract time derivative
    dv_dt = dreal_partials["partial_x_1_1"]

    # Define Hamiltonian for multi-vehicle system
    hamiltonian = 0
    for v_states, v_partials in zip(vehicles_states, vehicles_partials):
        # Each vehicle's dynamics contribution
        vehicle_ham = (v_partials['x'] * velocity * dreal.cos(v_states['theta']) + 
                      v_partials['y'] * velocity * dreal.sin(v_states['theta']))
        
        # Control input contribution
        if reachAim == 'avoid':
            vehicle_ham -= abs(v_partials['theta']) * omega_max  # Maximize angular velocity
        elif reachAim == 'reach':
            vehicle_ham += abs(v_partials['theta']) * omega_max  # Minimize angular velocity
            
        hamiltonian += vehicle_ham

    if reachMode == 'backward':
        hamiltonian = -hamiltonian

    # Define constraints
    condition_1 = abs(dv_dt + hamiltonian) <= epsilon
    condition_2 = abs(dv_dt) <= epsilon

    if setType == 'tube':
        final_condition = Not(And(condition_1, condition_2))
    else:
        final_condition = Not(condition_1)

    # State space constraints for all vehicles
    state_constraints = [t >= 0, t <= 1]
    for states in vehicles_states:
        state_constraints.extend([
            states['x'] >= -1, states['x'] <= 1,
            states['y'] >= -1, states['y'] <= 1,
            states['theta'] >= -np.pi, states['theta'] <= np.pi
        ])

    all_constraints = And(final_condition, And(*state_constraints))

    # Check constraints
    result = CheckSatisfiability(all_constraints, 1e-5)

    # Save results
    result_data = {
        "epsilon": epsilon,
        "set": f"{reachMode}_{reachAim}_{setType}", 
        "result": str(result) if result else "HJB Equation Satisfied",
        "num_vehicles": num_vehicles
    }

    result_file = f"{save_directory}/dreal_result.json"
    with open(result_file, "w") as f:
        json.dump(result_data, f, indent=4)

    logger.info(f"Saved result to {result_file}")
    return result_data