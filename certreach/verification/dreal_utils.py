import json
from dreal import Variable
import dreal
import sympy
import sympy
import logging

logger = logging.getLogger(__name__)

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

    # Raise an error if the term is unsupported
    logger.error(f"Unsupported term: {exp} (type: {type(exp)})")
    raise ValueError(f"[Error] Unsupported term: {exp} (type: {type(exp)})")

def extract_dreal_partials(final_symbolic_expression):
    """
    Extracts dReal-compatible variables and partial derivatives 
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

    return {
        "input_symbols": input_symbols,
        "partials": partials,
        "dreal_variables": dreal_variables,
        "dreal_partials": dreal_partials,
        **{f"sympy_partial_{i+1}": partial for i, partial in enumerate(partials)},
        **{f"dreal_partial_{i+1}": dreal_partials[f"partial_{str(input_symbols[i])}"] 
           for i in range(len(input_symbols))}
    }

def sympy_to_serializable(obj):
    """
    Converts SymPy expressions into JSON serializable strings.

    This function ensures that complex SymPy objects such as matrices, dictionaries, 
    and lists are converted into a format that can be saved as JSON. Floats are 
    converted with high precision to avoid rounding errors.

    Args:
        obj: The SymPy object or any nested structure containing SymPy expressions.

    Returns:
        A JSON-compatible representation of the object.
    """

    # Check if the object is a SymPy Basic type (expression or number)
    if isinstance(obj, sympy.Basic):
        # Convert to string with 17 decimal places to preserve precision
        return str(sympy.N(obj, 17))  

    # Check if the object is a SymPy Matrix
    if isinstance(obj, sympy.Matrix):
        # Recursively process each element of the matrix
        return [sympy_to_serializable(obj[i]) for i in range(obj.shape[0])]

    # Check if the object is a dictionary
    if isinstance(obj, dict):
        # Recursively process each key-value pair in the dictionary
        return {k: sympy_to_serializable(v) for k, v in obj.items()}

    # Check if the object is a list
    if isinstance(obj, list):
        # Recursively process each element in the list
        return [sympy_to_serializable(v) for v in obj]

    # If the object is not recognized, return it as is
    return obj

def serializable_to_sympy(data):
    """
    Restores serialized strings back into SymPy expressions.

    This function reconstructs SymPy objects from data stored as JSON-compatible
    formats such as strings, lists, or dictionaries. It ensures that serialized 
    symbolic expressions are correctly restored to SymPy types for further computation.

    Args:
        data: The serialized JSON-compatible data (strings, lists, or dictionaries).

    Returns:
        Restored SymPy expression, list, or dictionary of expressions.
    """

    # If the data is a string, attempt to parse it as a SymPy expression
    if isinstance(data, str):
        return sympy.sympify(data, evaluate=False)  # Prevent automatic simplification

    # If the data is a list, process each element recursively
    if isinstance(data, list):
        return [serializable_to_sympy(v) for v in data]

    # If the data is a dictionary, process each key-value pair recursively
    if isinstance(data, dict):
        return {k: serializable_to_sympy(v) for k, v in data.items()}

    # If the data is neither string, list, nor dictionary, return as is
    return data

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


