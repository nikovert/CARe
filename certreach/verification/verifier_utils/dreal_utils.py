import json
import sympy
import logging
import dreal
from dreal import Variable, And, Or, Not
import multiprocessing as mp
from certreach.verification.verifier_utils.constraint_builder import (
    prepare_constraint_data_batch,
    process_check_advanced,
    serialize_dreal_expression,
    parse_counterexample as parse_ce
)
import os
import signal
import time

logger = logging.getLogger(__name__)
use_if_then_else = False

# Global function map for mathematical operations
func_map = {
    'sin': dreal.sin,    # Sine function
    'cos': dreal.cos,    # Cosine function
    'exp': dreal.exp,    # Exponential function
    'tanh': dreal.tanh,  # Hyperbolic tangent
    'abs': abs           # Absolute value
}


def verify_with_dreal(d_real_value_fn, dreal_partials, dreal_variables, compute_hamiltonian, compute_boundary, epsilon=0.5, delta = 0.001,
                      reach_mode='forward', min_with='none', set_type='set', save_directory="./", execution_mode="parallel", additional_constraints=None):
    """
    Verifies if the HJB equation holds using dReal for a double integrator system.
    
    Parameters:
      ...
      reach_mode (str): 'forward' (default) or 'backward' for reach set computation
      min_with (str): Specifies minimum value computation method ('none', or 'target')
      set_type (str): 'set' (default) or 'tube' for target set type
      execution_mode (str): "parallel" (default) runs constraint checks concurrently,
                           "sequential" runs the boundary and derivative checks in sequence while timing each.
    """
    
    
    # Extract state variables and partial derivatives dynamically
    state_vars = []
    partials = []
    for i in range(2, len(dreal_variables) + 1):
        state_vars.append(dreal_variables[f"x_1_{i}"])
        partials.append(dreal_partials[f"partial_x_1_{i}"])

    # Use class method for Hamiltonian computation and other setup
    hamiltonian_value = compute_hamiltonian(state_vars, partials, func_map)

    if set_type == 'tube': 
        hamiltonian_value = dreal.Max(hamiltonian_value, 0)

    if reach_mode == 'backward':
        hamiltonian_value = -hamiltonian_value

    hamiltonian_expr = serialize_dreal_expression(hamiltonian_value)
    
    # Serialize expressions
    value_fn_expr = serialize_dreal_expression(d_real_value_fn)
    partials_expr = {key: serialize_dreal_expression(val) for key, val in dreal_partials.items()}
    
    boundary_value = compute_boundary(state_vars)
    boundary_expr = serialize_dreal_expression(boundary_value)

    result = None
    timing_info = {}
    
    if execution_mode == "parallel":
        logger.info("Starting parallel constraint checks with multiprocessing.Pool...")
        
        # Prepare data for parallel execution
        state_dim = len(state_vars)
        
        # Use the constraint builder to create serializable constraint data
        constraint_data_batch = prepare_constraint_data_batch(
            state_dim=state_dim,
            epsilon=epsilon,
            delta=delta,
            min_with=min_with,
            reach_mode=reach_mode,
            set_type=set_type
        )
        
        # Create a multiprocessing pool
        pool = mp.Pool()
        
        # Track async results
        async_results = []
        
        # Submit all tasks asynchronously
        for constraint_data in constraint_data_batch:
            async_result = pool.apply_async(
                process_check_advanced,
                args=(
                    constraint_data,
                    hamiltonian_expr,
                    value_fn_expr,
                    boundary_expr,
                    partials_expr
                )
            )
            async_results.append(async_result)
        
        try:
            # Wait for results and process them as they arrive
            while async_results and not result:
                # Check each result that's ready without blocking
                for i, async_result in enumerate(list(async_results)):
                    if async_result.ready():
                        try:
                            constraint_id, constraint_result = async_result.get(0)  # Non-blocking
                            async_results.pop(i)  # Remove this result from the list
                            
                            if constraint_result and not constraint_result.startswith("Error"):
                                # We found a counterexample
                                result = constraint_result
                                logger.info(f"Found counterexample in constraint {constraint_id}")
                                break
                            elif constraint_result and constraint_result.startswith("Error"):
                                logger.error(f"Error in constraint {constraint_id}: {constraint_result}")
                        except Exception as e:
                            logger.error(f"Error getting result: {e}")
                            async_results.pop(i)  # Remove problematic result
                
                # If we found a counterexample or all tasks are done, break
                if result or not async_results:
                    break
                
                # Sleep briefly to avoid high CPU usage
                time.sleep(5)
                
        finally:
            # Immediately terminate the pool if a counterexample was found
            if result:
                logger.info("Terminating process pool due to counterexample found")
                pool.terminate()
            else:
                logger.info("Closing process pool normally")
                pool.close()
                
            # Always join the pool to clean up resources properly
            pool.join()
            
        logger.info("Parallel constraint checks completed.")
    
    elif execution_mode == "sequential":
        # ... existing sequential execution code ...
        pass
    else:
        logger.error(f"Unknown execution_mode: {execution_mode}.")
    
    # ... existing result processing code ...
    
    if not result:
        success = True  # HJB Equation is satisfied
        logger.info("No counterexamples found in checks.")
        verification_result = {
            "epsilon": epsilon,
            "result": "HJB Equation Satisfied",
            "counterexample": None,
            "timing": timing_info
        }
    else:
        success = False  # HJB Equation is not satisfied
        verification_result = {
            "epsilon": epsilon,
            "result": "HJB Equation Not Satisfied",
            "counterexample": parse_ce(str(result)),
            "timing": timing_info
        }
    
    # Optionally save result to file
    result_file = f"{save_directory}/dreal_result.json"
    with open(result_file, "w") as f:
        json.dump(verification_result, f, indent=4)
    logger.debug(f"Saved result to {result_file}")

    counterexample = parse_ce(str(result)) if not success else None

    return success, counterexample

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
            if "x_" in variable:
                counterexample[variable.strip()] = (lower, upper)
    except Exception as e:
        logger.error(f"Failed to parse counterexample: {e}")
        return None
    return counterexample

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
        if any(isinstance(term, dict) for term in terms):
            # Handle Heaviside formulas separately
            result = terms[0]
            for term in terms[1:]:
                result["expression"] +=term["expression"]
                result["variable_conditions"].append(*term["variable_conditions"])
            return result
            
        # Regular addition for other terms
        return sum(terms)

    # Handle multiplication: Multiply all terms
    elif isinstance(exp, sympy.Mul):
        for idx, arg in enumerate(exp.args):
            if idx == 0: 
                result = sympy_to_dreal_converter(syms, arg, to_number, expand_pow)
            elif isinstance(arg, sympy.Heaviside) and not use_if_then_else:
                # Handle multiplication with Heaviside expressions separately
                result = heaviside_sympy_to_dreal_converter(syms, result, exp.args[idx], to_number, expand_pow)
            else:
                result *= sympy_to_dreal_converter(syms, arg, to_number, expand_pow)
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

        # Handle Heaviside expressions using AND/OR gates with a result variable
        if isinstance(exp, sympy.Heaviside):
            arg = sympy_to_dreal_converter(syms, exp.args[0], to_number, expand_pow)
            # Use the result variable in the logical operations
            if use_if_then_else:
                return dreal.if_then_else(arg < 0, 0, 1)
            else:
                result_var = Variable(f"heaviside_result_{id(exp)}")  # Create unique variable name
                return Or(And(arg < 0, result_var == 0), 
                     And(Not(arg < 0), result_var == 1))

    # Handle Max/Min expressions
    elif isinstance(exp, sympy.Max) or isinstance(exp, sympy.Min):
        arg1 = sympy_to_dreal_converter(syms, exp.args[0], to_number, expand_pow)
        arg2 = sympy_to_dreal_converter(syms, exp.args[1], to_number, expand_pow)
        if isinstance(exp, sympy.Max):
            #return Or(And(arg1 > arg2, arg1), And(arg1 <= arg2, arg2))
            return dreal.Max(arg1, arg2)
        else:  # Min(a,b) = (a < b AND a) OR (a >= b AND b)
            #return Or(And(arg1 < arg2, arg1), And(arg1 >= b, arg2))
            return dreal.Min(arg1, arg2)

    # Raise an error if the term is unsupported
    logger.error(f"Unsupported term: {exp} (type: {type(exp)})")
    raise ValueError(f"[Error] Unsupported term: {exp} (type: {type(exp)})")
    
def heaviside_sympy_to_dreal_converter(syms: dict, exp1: sympy.Expr, exp2: sympy.Expr, to_number=lambda x: float(x), expand_pow=True):
    if isinstance(exp2, sympy.Heaviside):
        exp = exp2
        value = exp1
    else:
        exp = exp1
        value = exp2
    
    arg = sympy_to_dreal_converter(syms, exp.args[0], to_number, expand_pow)
    result_var = Variable(f"heaviside_result_{id(exp)}")  # Create unique variable name
    
    result = {"expression": result_var,
              "variable_conditions": [Or(And(arg < 0, result_var == 0), And(Not(arg < 0), result_var == value))]}
    
    # Use the result variable in the logical operations
    return result

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

    dreal_partials = {}
    additional_conditions = []
    for var, partial in zip(input_symbols, partials):
        partial_result = sympy_to_dreal_converter(dreal_variables, partial)
        if isinstance(partial_result, dict):
            dreal_partials[f"partial_{str(var)}"] = partial_result["expression"]
            additional_conditions.extend(partial_result["variable_conditions"])
        else:
            dreal_partials[f"partial_{str(var)}"] = partial_result

    # Convert symbolic value function to dReal expression - fix by accessing first element
    d_real_value_fn = sympy_to_dreal_converter(dreal_variables, final_symbolic_expression[0])

    results = {
        "input_symbols": input_symbols,
        "partials": partials,
        "dreal_variables": dreal_variables,
        "dreal_partials": dreal_partials,
        "d_real_value_fn": d_real_value_fn,
        **{f"sympy_partial_{i+1}": partial for i, partial in enumerate(partials)},
        **{f"dreal_partial_{i+1}": dreal_partials[f"partial_{str(input_symbols[i])}"] 
           for i in range(len(input_symbols))},
        "additional_conditions": additional_conditions
    }
    return results

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



