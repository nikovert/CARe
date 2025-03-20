import logging
import z3
import sympy
from care.verification.verifier_utils.symbolic import compute_partial_deriv

logger = logging.getLogger(__name__)

def parse_z3_expression(expression):
    return z3.deserialize(expression)

def _check_constraint(constraints) -> z3.Model:
    """Helper function to check constraints with Z3."""
    solver = z3.Solver()
    solver.add(constraints)
    if solver.check() == z3.sat:
        model = solver.model()
        logger.info(f"Z3 found a counterexample: {model}")
        return model
    return None

def check_with_z3(constraint, **kwargs):
    """
    Check a constraint using Z3.
    
    Parameters:
        constraint: Z3 constraint to check
        kwargs: Additional keyword arguments
    
    Returns:
        tuple: (success, counterexample)
    """
    result = _check_constraint(constraint)
    if result:
        counterexample = [float(result[d].as_decimal(10)) for d in result.decls()]
        return counterexample
    return None

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

    symbol_list = [*input_symbols]
    for sym in input_symbols:
        symbol_list.append(sympy.Symbol(f"partial_{sym}"))

    # Compute symbolic partial derivatives
    partials = compute_partial_deriv(final_symbolic_expression[0], input_symbols)

    # Convert SymPy symbols to Z3 variables
    z3_variables = {str(sym): z3.Real(str(sym)) for sym in symbol_list}

    # Convert symbolic partial derivatives to Z3 expressions
    z3_partials = {}
    for var, partial in zip(input_symbols, partials):
        partial_result = sympy_to_z3_converter(z3_variables, partial)
        z3_partials[f"partial_{str(var)}"] = z3.simplify(partial_result)

    # Convert symbolic value function to Z3 expression
    z3_value_fn = sympy_to_z3_converter(z3_variables, final_symbolic_expression[0])

    return {
        "variables": z3_variables,
        "partials": z3_partials,
        "value_fn": z3.simplify(z3_value_fn)
    }

def z3_max(*args):
    # Compute maximum using pairwise comparison
    if not args:
        raise ValueError("z3_max requires at least one argument")
    result = args[0]
    for arg in args[1:]:
        result = z3.If(result >= arg, result, arg)
    return result

def z3_min(*args):
    # Compute minimum using pairwise comparison
    if not args:
        raise ValueError("z3_min requires at least one argument")
    result = args[0]
    for arg in args[1:]:
        result = z3.If(result <= arg, result, arg)
    return result

z3_function_map = {
    'solver_name': 'z3',
    'abs': z3.Abs,
    'max': z3_max,  # replaced z3.max with custom z3_max
    'min': z3_min,  # replaced z3.min with custom z3_min
    'and': z3.And,
    'or': z3.Or, 
    'not': z3.Not,
    'variable': z3.Real
}