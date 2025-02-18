import json
from dreal import Variable
import dreal
import sympy
import logging
import time 
from dreal import And, Not, CheckSatisfiability
import concurrent.futures
from typing import Optional

logger = logging.getLogger(__name__)

def _check_constraint(constraint: dreal.Formula, precision: float) -> Optional[dreal.Box]:
    """Helper function to check a single constraint."""
    logger.debug(f"Starting constraint check with precision {precision}")
    result = CheckSatisfiability(constraint, precision)
    logger.debug(f"Constraint check completed. Found counterexample: {result is not None}")
    return result

def verify_with_dreal(d_real_value_fn, dreal_partials, dreal_variables, compute_hamiltonian, compute_boundary, epsilon=0.5,
                      reachMode='forward', setType='set', save_directory="./", execution_mode="sequential"):
    """
    Verifies if the HJB equation holds using dReal for a double integrator system.
    
    Parameters:
      ...
      execution_mode (str): "parallel" (default) runs constraint checks concurrently,
                            "sequential" runs the boundary and derivative checks in sequence while timing each.
    """
    
    # Extract time variable
    t = dreal_variables["x_1_1"]
    
    # Extract state variables and partial derivatives dynamically
    state_vars = []
    partials = []
    for i in range(2, len(dreal_variables) + 1):
        state_vars.append(dreal_variables[f"x_1_{i}"])
        partials.append(dreal_partials[f"partial_x_1_{i}"])

    dv_dt = dreal_partials["partial_x_1_1"]

    # Use class method for Hamiltonian computation
    hamiltonian_value = compute_hamiltonian(state_vars, partials)

    if reachMode == 'backward':
        hamiltonian_value = -hamiltonian_value

    # Define derivative constraints
    condition_1 = abs(dv_dt + hamiltonian_value) <= epsilon
    condition_2 = abs(dv_dt) <= epsilon

    if setType=='tube': # this can be parallelized
        derivative_condition = Not(And(condition_1, condition_2))
    else:
        derivative_condition = Not(condition_1)

    # Define boundary constraints (assuming T=1)
    boundary_value = compute_boundary(state_vars)
    boundary_condition = abs(d_real_value_fn - boundary_value) > epsilon

    # Define State constraints
    state_constraints = And(
        t >= 0, t <= 1,
        *[And(var >= -1, var <= 1) for var in state_vars]
    )

    initial_state_constraints = And(
        t == 0,
        *[And(var >= -1, var <= 1) for var in state_vars]
    )

    boundary_constraints = And(boundary_condition, initial_state_constraints)
    derivative_constraints = And(derivative_condition, state_constraints)
    
    result = None
    timing_info = {}
    
    if execution_mode == "parallel":
        logger.info("Starting parallel constraint checks...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            logger.debug("Submitting boundary and derivative constraint checks")
            futures = [
                executor.submit(_check_constraint, boundary_constraints, 1e-3),
                executor.submit(_check_constraint, derivative_constraints, 1e-3)
            ]
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res:
                    result = res
                    logger.info("Found counterexample; cancelling remaining tasks.")
                    for f in futures:
                        if f is not future:
                            f.cancel()
                    break
        logger.info("Parallel constraint checks completed.")
    elif execution_mode == "sequential":
        logger.info("Starting sequential constraint checks with timing...")
        start_boundary = time.monotonic()
        boundary_result = _check_constraint(boundary_constraints, 1e-3)
        boundary_time = time.monotonic() - start_boundary
        timing_info["boundary_time"] = boundary_time
        logger.info(f"Boundary check completed in {boundary_time:.4f} seconds.")
        if boundary_result:
            result = boundary_result
        else:
            start_derivative = time.monotonic()
            derivative_result = _check_constraint(derivative_constraints, 1e-3)
            derivative_time = time.monotonic() - start_derivative
            timing_info["derivative_time"] = derivative_time
            logger.info(f"Derivative check completed in {derivative_time:.4f} seconds.")
            result = derivative_result
    else:
        logger.error(f"Unknown execution_mode: {execution_mode}.")
    
    if not result:
        logger.info("No counterexamples found in checks.")
    
    result_data = {
        "epsilon": epsilon,
        "set": f"{reachMode}_{setType}",
        "result": str(result) if result else "HJB Equation Satisfied",
        "timing": timing_info
    }
    result_file = f"{save_directory}/dreal_result.json"
    with open(result_file, "w") as f:
        json.dump(result_data, f, indent=4)
    logger.debug(f"Saved result to {result_file}")
    return result_data

def convert_symbols_to_dreal(input_symbols):
    """
    Convert SymPy symbols to dReal variables.

    Args:
        input_symbols (sympy.Matrix): Matrix of SymPy symbols.

    Returns:
        dict: A dictionary mapping SymPy symbol names to corresponding dReal variables.
    """
    logger = logging.getLogger(__name__)
    # Create a dictionary mapping SymPy symbols to dReal variables
    dreal_variables = {str(sym): Variable(str(sym)) for sym in input_symbols}

    # Print the mapping for debugging and verification
    logger.debug("Conversion from SymPy to dReal:")
    for sym, dreal_var in dreal_variables.items():
        logger.debug(f"{sym} -> {dreal_var}")

    return dreal_variables

def sympy_to_dreal_converter(syms: dict, exp: sympy.Expr, to_number=lambda x: float(x), expand_pow=True):
    """
    Convert a SymPy expression to a dReal-compatible expression.

    Args:
        syms (dict): Dictionary mapping SymPy symbols to dReal variables.
        exp (sympy.Expr): The SymPy expression to be converted.
        to_number (callable): Function for numeric conversion (default: float).
        expand_pow (bool): Whether to expand powers manually (default: True).

    Returns:
        dReal expression: The equivalent expression in dReal format.
    
    Raises:
        ValueError: If the expression cannot be converted.
    """

    # Handle symbols: Look up corresponding dReal variable
    if isinstance(exp, sympy.Symbol):
        exp_str = str(exp)
        if exp_str not in syms:
            raise ValueError(f"[Error] Symbol '{exp_str}' not found in provided symbols dictionary. "
                             f"Available symbols: {list(syms.keys())}")
        return syms[exp_str]

    # Handle numeric constants
    elif isinstance(exp, sympy.Number):
        try:
            return to_number(exp)
        except Exception:
            return sympy.Float(exp, len(str(exp)))

    # Handle addition: Sum all terms
    elif isinstance(exp, sympy.Add):
        terms = [sympy_to_dreal_converter(syms, arg, to_number, expand_pow) for arg in exp.args]
        return sum(terms)

    # Handle multiplication: Multiply all terms
    elif isinstance(exp, sympy.Mul):
        terms = [sympy_to_dreal_converter(syms, arg, to_number, expand_pow) for arg in exp.args]
        result = terms[0]
        for term in terms[1:]:
            result *= term
        return result

    # Handle exponentiation (powers)
    elif isinstance(exp, sympy.Pow):
        base = sympy_to_dreal_converter(syms, exp.args[0], to_number, expand_pow)
        exponent = sympy_to_dreal_converter(syms, exp.args[1], to_number, expand_pow)

        # Expand integer powers if requested
        if expand_pow:
            try:
                exp_val = float(exponent)
                if exp_val.is_integer():
                    exp_val = int(exp_val)
                    result = base
                    for _ in range(exp_val - 1):
                        result *= base
                    return result
            except Exception:
                pass

        # Use the power operator if expansion fails
        return base**exponent

    # Handle standard mathematical functions
    elif isinstance(exp, sympy.Function):
        func_map = {
            sympy.sin: dreal.sin,    # Sine function
            sympy.cos: dreal.cos,    # Cosine function
            sympy.exp: dreal.exp,    # Exponential function
            sympy.tanh: dreal.tanh,  # Hyperbolic tangent
        }

        for sympy_func, dreal_func in func_map.items():
            if isinstance(exp, sympy_func):
                # Convert the argument and apply the corresponding dReal function
                arg = sympy_to_dreal_converter(syms, exp.args[0], to_number, expand_pow)
                return dreal_func(arg)

        # Handle Heaviside expressions using dreal.if_then_else
        if isinstance(exp, sympy.Heaviside):
            arg = sympy_to_dreal_converter(syms, exp.args[0], to_number, expand_pow)
            return dreal.if_then_else(arg < 0, 0, 1)
        
    # Handle Max/Min expressions
    elif isinstance(exp, sympy.Max) or isinstance(exp, sympy.Min):
        arg1 = sympy_to_dreal_converter(syms, exp.args[0], to_number, expand_pow)
        arg2 = sympy_to_dreal_converter(syms, exp.args[1], to_number, expand_pow)
        return dreal.Max(arg1, arg2) if isinstance(exp, sympy.Max) else dreal.Min(arg1, arg2)

    # Raise an error if the term is unsupported
    logger.error(f"Unsupported term: {exp} (type: {type(exp)})")
    raise ValueError(f"[Error] Unsupported term: {exp} (type: {type(exp)})")

def extract_dreal_partials(final_symbolic_expression):
    """
    Extracts dReal-compatible variables, value_function and partial derivatives 
    from a given symbolic expression.

    Args:
        final_symbolic_expression (sympy.Matrix): The symbolic expression from the neural network.

    Returns:
        dict: A dictionary containing dReal variables and their partial derivatives.
    """
    # Get input symbols from first layer (x_1_1, x_1_2, etc.)
    input_symbols = [sym for sym in final_symbolic_expression.free_symbols 
                    if str(sym).startswith('x_1_')]
    input_symbols.sort(key=lambda x: int(str(x).split('_')[2]))  # Sort by index
    input_symbols = sympy.Matrix(input_symbols)

    # Compute symbolic partial derivatives
    partials = [final_symbolic_expression[0].diff(var) for var in input_symbols]

    # Convert SymPy symbols to dReal variables
    dreal_variables = convert_symbols_to_dreal(input_symbols)

    # Convert symbolic partial derivatives to dReal expressions
    dreal_partials = {
        f"partial_{str(var)}": sympy_to_dreal_converter(dreal_variables, partial)
        for var, partial in zip(input_symbols, partials)
    }

    # Convert symbolic value function to dReal expression - fix by accessing first element
    d_real_value_fn = sympy_to_dreal_converter(dreal_variables, final_symbolic_expression[0])

    return {
        "input_symbols": input_symbols,
        "partials": partials,
        "dreal_variables": dreal_variables,
        "dreal_partials": dreal_partials,
        "d_real_value_fn": d_real_value_fn,
        **{f"sympy_partial_{i+1}": partial for i, partial in enumerate(partials)},
        **{f"dreal_partial_{i+1}": dreal_partials[f"partial_{str(input_symbols[i])}"] 
           for i in range(len(input_symbols))}
    }

def process_dreal_result(json_path):
    """
    Process the dReal result from a JSON file to determine whether the HJB Equation is satisfied,
    display the epsilon value, and optionally return the counterexample if verification is not satisfied.

    Args:
        json_path (str): Path to the JSON file containing dReal results.

    Returns:
        dict: Parsed dReal result including epsilon, result details, and counterexample range if applicable.
    """
    logger = logging.getLogger(__name__)
    def parse_counterexample(result_str):
        """
        Parse the counterexample from the dReal result string.

        Args:
            result_str (str): The result string from the dReal output.

        Returns:
            dict: Parsed counterexample as a dictionary of variable ranges.
        """
        counterexample = {}
        try:
            for line in result_str.strip().split('\n'):
                # Parse each variable and its range
                variable, value_range = line.split(':')
                lower, upper = map(float, value_range.strip('[] ').split(','))
                counterexample[variable.strip()] = (lower, upper)
        except Exception as e:
            logger.error(f"Failed to parse counterexample: {e}")
            return None
        return counterexample

    try:
        # Load the JSON result from the specified file
        with open(json_path, "r") as file:
            dreal_result = json.load(file)

        # Extract epsilon value and result
        epsilon = dreal_result.get("epsilon", "Unknown")
        result = dreal_result.get("result", "Unknown")
        logger.info(f"Epsilon: {epsilon}")

        # Check and process the result
        if "HJB Equation Satisfied" in result:
            logger.info("dReal verification satisfied. HJB Equation is satisfied.")
            logger.info(f"Reachable Set: {dreal_result.get('set', 'Unknown')}")
            return {"epsilon": epsilon, "result": result, "counterexample": None}
        else:
            logger.info("dReal verification NOT satisfied. Counterexample found:")
            counterexample = parse_counterexample(result)
            if counterexample:
                for variable, (lower, upper) in counterexample.items():
                    logger.debug(f"  {variable}: [{lower}, {upper}]")
            return {"epsilon": epsilon, "result": result, "counterexample": counterexample}

    except FileNotFoundError:
        logger.error(f"File not found at {json_path}.")
        return {"error": "FileNotFound"}
    except json.JSONDecodeError:
        logger.error(f"Unable to parse JSON from the file at {json_path}.")
        return {"error": "JSONDecodeError"}
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return {"error": str(e)}


