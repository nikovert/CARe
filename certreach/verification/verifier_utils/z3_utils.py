import json
import time
import logging
import z3
import os
import sympy

logger = logging.getLogger(__name__)

func_map = {'abs': z3.Abs}

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
    """
    counterexample = [float(model[d].as_decimal(10)) for d in model.decls()]
    return counterexample

def verify_with_z3(z3_value_fn, z3_partials, z3_variables, compute_hamiltonian, compute_boundary, epsilon=0.5,
                   reach_mode='forward', reach_aim='avoid', min_with='none', set_type='set', save_directory="./", additional_constraints=None):
    """
    Verify the HJB equation using Z3.
    
    Parameters:
        z3_value_fn: The Z3 expression for the value function
        z3_partials: Dictionary of Z3 expressions for partial derivatives
        z3_variables: Dictionary of Z3 variables
        compute_hamiltonian: Function to compute the Hamiltonian
        compute_boundary: Function to compute boundary values
        epsilon: Verification tolerance
        reach_mode: 'forward' (default) or 'backward' reach set computation
        reach_aim: 'avoid' (default) or 'reach' computation goal
        min_with: Minimum value computation method ('none', 'zero', or 'target')
        set_type: 'set' (default) or 'tube' for target set type
        save_directory: Directory to save verification results
        additional_constraints: Additional Z3 constraints to include
    
    Returns:
        tuple: (success, counterexample)
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
    
    hamiltonian_value = compute_hamiltonian(state_vars, partials, func_map)
    if reach_mode == 'backward':
        hamiltonian_value = -hamiltonian_value

    # Define Z3 constraints using z3.Abs for absolute values
    condition_1 = z3.Abs(dv_dt + hamiltonian_value) <= epsilon
    condition_2 = z3.Abs(dv_dt) <= epsilon

    derivative_condition = z3.Not(z3.And(condition_1, condition_2)) if set_type=='tube' else z3.Not(condition_1)
    
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

def sympy_to_z3_converter(syms: dict, exp: sympy.Expr, to_number=lambda x: float(x), expand_pow=True):
    """
    Convert a SymPy expression to a Z3-compatible expression.

    Args:
        syms (dict): Dictionary mapping SymPy symbols to Z3 variables.
        exp (sympy.Expr): The SymPy expression to be converted.
        to_number (callable): Function for numeric conversion (default: float).
        expand_pow (bool): Whether to expand powers manually (default: True).

    Returns:
        Z3 expression: The equivalent expression in Z3 format.
    
    Raises:
        ValueError: If the expression cannot be converted.
    """
    
    # Handle symbols: Look up corresponding Z3 variable
    if isinstance(exp, sympy.Symbol):
        exp_str = str(exp)
        if exp_str not in syms:
            raise ValueError(f"[Error] Symbol '{exp_str}' not found in provided symbols dictionary. "
                             f"Available symbols: {list(syms.keys())}")
        return syms[exp_str]

    # Handle numeric constants
    elif isinstance(exp, sympy.Number):
        try:
            value = to_number(exp)
            return z3.RealVal(value)
        except Exception:
            return z3.RealVal(float(sympy.Float(exp, len(str(exp)))))

    # Handle addition: Sum all terms
    elif isinstance(exp, sympy.Add):
        terms = [sympy_to_z3_converter(syms, arg, to_number, expand_pow) for arg in exp.args]
        result = terms[0]
        for term in terms[1:]:
            result += term
        return result

    # Handle multiplication: Multiply all terms
    elif isinstance(exp, sympy.Mul):
        terms = [sympy_to_z3_converter(syms, arg, to_number, expand_pow) for arg in exp.args]
        result = terms[0]
        for term in terms[1:]:
            result *= term
        return result

    # Handle exponentiation (powers)
    elif isinstance(exp, sympy.Pow):
        base = sympy_to_z3_converter(syms, exp.args[0], to_number, expand_pow)
        exponent = exp.args[1]

        # Expand integer powers if requested
        if expand_pow:
            try:
                exp_val = float(exponent)
                if exp_val.is_integer():
                    exp_val = int(exp_val)
                    if exp_val >= 0:  # Positive exponent
                        result = 1
                        for _ in range(exp_val):
                            result *= base
                        return result
                    else:  # Negative exponent - use division
                        result = 1
                        for _ in range(-exp_val):
                            result /= base
                        return result
            except Exception:
                pass
        
        # For non-integer exponents or if expansion fails
        exponent_converted = sympy_to_z3_converter(syms, exponent, to_number, expand_pow)
        return z3.simplify(z3.Pow(base, exponent_converted))

    # Handle # Handle Heaviside function using ITE (if-then-else)
    elif isinstance(exp, sympy.Heaviside):
            arg = sympy_to_z3_converter(syms, exp.args[0], to_number, expand_pow)
            return z3.If(arg < 0, 0, 1)

    # Handle Max/Min expressions
    elif isinstance(exp, sympy.Max) or isinstance(exp, sympy.Min):
        args = [sympy_to_z3_converter(syms, arg, to_number, expand_pow) for arg in exp.args]
        if isinstance(exp, sympy.Max):
            # Process pairwise for multiple arguments
            result = args[0]
            for arg in args[1:]:
                result = z3.If(result > arg, result, arg)
            return result
        else:  # Min
            result = args[0]
            for arg in args[1:]:
                result = z3.If(result < arg, result, arg)
            return result

    # Raise an error if the term is unsupported
    logger.error(f"Unsupported term in Z3 conversion: {exp} (type: {type(exp)})")
    raise ValueError(f"[Error] Unsupported term in Z3 conversion: {exp} (type: {type(exp)})")

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
        partial_result = sympy_to_z3_converter(z3_variables, partial)
        z3_partials[f"partial_{str(var)}"] = partial_result

    # Convert symbolic value function to Z3 expression
    z3_value_fn = sympy_to_z3_converter(z3_variables, final_symbolic_expression[0])

    return {
        "input_symbols": input_symbols,
        "partials": partials,
        "z3_variables": z3_variables,
        "z3_partials": z3_partials,
        "z3_value_fn": z3_value_fn,
        **{f"sympy_partial_{i+1}": partial for i, partial in enumerate(partials)},
        **{f"z3_partial_{i+1}": z3_partials[f"partial_{str(input_symbols[i])}"] 
           for i in range(len(input_symbols))}
    }
