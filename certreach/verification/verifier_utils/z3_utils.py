import json
import time
import logging
import z3
import os
import sympy

logger = logging.getLogger(__name__)

def _check_constraint(constraints) -> z3.Model:
    """Helper function to check constraints with Z3."""
    solver = z3.Solver()
    solver.add(constraints)
    if solver.check() == z3.sat:
        model = solver.model()
        logger.info(f"Z3 found a counterexample: {model}")
        return model
    return None

def parse_counterexample(model: z3.Model):
    """
    Parse the counterexample from a Z3 model.
    
    Returns:
        dict: Counterexample as a dictionary where each variable is mapped to a tuple (value, value).
    """
    counterexample = {}
    for d in model.decls():
        value = model[d]
        try:
            num_val = float(value.numeral_as_decimal(10).rstrip('?'))
            counterexample[d.name()] = (num_val, num_val)
        except Exception as e:
            logger.error(f"Error parsing variable {d.name()}: {e}")
    return counterexample

def verify_with_z3(z3_value_fn, z3_partials, z3_variables, compute_hamiltonian, compute_boundary, epsilon=0.5,
                   reachMode='forward', setType='set', save_directory="./", additional_constraints=None):
    """
    Verify the HJB equation using Z3.
    
    Follows a similar structure to dReal's verification.
    """
    # Extract time variable (assume key "x_1_1")
    t = z3_variables["x_1_1"]
    
    state_vars = []
    partials = []
    # Collect state variables and partial derivatives (starting from index 2)
    for i in range(2, len(z3_variables) + 1):
        state_vars.append(z3_variables[f"x_1_{i}"])
        partials.append(z3_partials[f"partial_x_1_{i}"])
    
    dv_dt = z3_partials["partial_x_1_1"]
    
    hamiltonian_value = compute_hamiltonian(state_vars, partials)
    if reachMode == 'backward':
        hamiltonian_value = -hamiltonian_value

    # Define Z3 constraints using z3.Abs for absolute values
    condition_1 = z3.Abs(dv_dt + hamiltonian_value) <= epsilon
    condition_2 = z3.Abs(dv_dt) <= epsilon

    derivative_condition = z3.Not(z3.And(condition_1, condition_2)) if setType=='tube' else z3.Not(condition_1)
    
    boundary_value = compute_boundary(state_vars)
    boundary_condition = z3.Abs(z3_value_fn - boundary_value) > epsilon

    state_constraints = z3.And(
        t >= 0, t <= 1,
        *[z3.And(var >= -1, var <= 1) for var in state_vars]
    )

    initial_state_constraints = z3.And(
        t == 0,
        *[z3.And(var >= -1, var <= 1) for var in state_vars]
    )

    if additional_constraints:
        boundary_constraints = z3.And(boundary_condition, initial_state_constraints, *additional_constraints)
        derivative_constraints = z3.And(derivative_condition, state_constraints, *additional_constraints)
    else:
        boundary_constraints = z3.And(boundary_condition, initial_state_constraints)
        derivative_constraints = z3.And(derivative_condition, state_constraints)
    
    timing_info = {}
    result = None

    # Sequential check
    start_boundary = time.monotonic()
    boundary_model = _check_constraint(boundary_constraints)
    timing_info["boundary_time"] = time.monotonic() - start_boundary

    if boundary_model:
        result = boundary_model
        logger.info("Boundary constraint counterexample found.")
    else:
        start_derivative = time.monotonic()
        derivative_model = _check_constraint(derivative_constraints)
        timing_info["derivative_time"] = time.monotonic() - start_derivative
        if derivative_model:
            result = derivative_model
            logger.info("Derivative constraint counterexample found.")
    
    if not result:
        success = True
        verification_result = {
            "epsilon": epsilon,
            "result": "HJB Equation Satisfied",
            "counterexample": None,
            "timing": timing_info
        }
        logger.info("No counterexample found with Z3.")
    else:
        success = False
        verification_result = {
            "epsilon": epsilon,
            "result": "HJB Equation Not Satisfied",
            "counterexample": parse_counterexample(result),
            "timing": timing_info
        }
    
    # Save result to file
    os.makedirs(save_directory, exist_ok=True)
    result_file = os.path.join(save_directory, "z3_result.json")
    with open(result_file, "w") as f:
        json.dump(verification_result, f, indent=4)
    logger.debug(f"Saved Z3 result to {result_file}")

    counterexample = parse_counterexample(result) if not success else None
    return success, counterexample

def extract_z3_partials(final_symbolic_expression):
    """
    Extracts Z3-compatible variables, value_function and partial derivatives 
    from a given symbolic expression.

    Args:
        final_symbolic_expression (sympy.Matrix): The symbolic expression from the neural network.

    Returns:
        dict: A dictionary containing Z3 variables and their partial derivatives.
    """
    # Get input symbols from first layer (x_1_1, x_1_2, etc.)
    input_symbols = [sym for sym in final_symbolic_expression.free_symbols 
                    if str(sym).startswith('x_1_')]
    input_symbols.sort(key=lambda x: int(str(x).split('_')[2]))  # Sort by index
    input_symbols = sympy.Matrix(input_symbols)

    # Compute symbolic partial derivatives
    partials = [final_symbolic_expression[0].diff(var) for var in input_symbols]

    # Convert SymPy symbols to Z3 variables
    z3_variables = {str(sym): z3.Real(str(sym)) for sym in input_symbols}

    # Convert symbolic partial derivatives to Z3 expressions
    z3_partials = {}
    for var, partial in zip(input_symbols, partials):
        z3_partials[f"partial_{str(var)}"] = z3.simplify(z3.RealVal(partial))

    return {
        "z3_variables": z3_variables,
        "z3_partials": z3_partials,
        "z3_value_fn": z3.simplify(z3.RealVal(final_symbolic_expression[0]))
    }
