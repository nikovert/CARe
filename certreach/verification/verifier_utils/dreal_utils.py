import json
import sympy
import logging
import dreal
from dreal import Variable, And, Or, Not, CheckSatisfiability
from certreach.verification.verifier_utils.symbolic import compute_partial_deriv

logger = logging.getLogger(__name__)
use_if_then_else = False

def check_with_dreal(constraints, delta, **kwargs):
    result = CheckSatisfiability(constraints, delta)
    return result

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
    partials = compute_partial_deriv(final_symbolic_expression[0], input_symbols)

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
        "variables": dreal_variables,
        "partials": dreal_partials,
        "value_fn": d_real_value_fn,
        "additional_conditions": additional_conditions
    }
    return results

# Global function map for SMT specific operations
dreal_function_map = {
    'solver_name': 'dreal',
    'sin': dreal.sin,    # Sine function
    'cos': dreal.cos,    # Cosine function
    'exp': dreal.exp,    # Exponential function
    'tanh': dreal.tanh,  # Hyperbolic tangent
    'abs': abs,           # Absolute value
    'max': dreal.Max,    # Maximum value
    'min': dreal.Min,     # Minimum value
    'and': And,          # Logical AND
    'or': Or,            # Logical OR
    'not': Not,           # Logical NOT
    'solve': check_with_dreal,
    'variable': Variable
}
