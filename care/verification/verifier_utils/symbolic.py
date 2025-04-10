import os
import sympy
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Any, Union
import logging
import torch

logger = logging.getLogger(__name__)

def sine_transform(x: sympy.Basic, frequency: float = 30.0) -> sympy.Basic:
    """Helper function for sine transformation."""
    return sympy.sin(frequency * x)

def relu_primitive_transform(x: sympy.Basic) -> sympy.Basic:
    """Helper function for ReLU primitive transformation."""
    return sympy.Max(0, x)**2

def power_transform(x: sympy.Symbol, power: int) -> sympy.Symbol:
    """Helper function for polynomial transformation."""
    return x**power

class SymbolicPolynomialTransform:
    """A picklable class for polynomial transformations."""
    def __init__(self, power: int):
        self.power = power
    
    def __call__(self, x: sympy.Symbol) -> sympy.Symbol:
        return power_transform(x, self.power)

def get_symbolic_layer_output_generalized(state_dict: Dict[str, torch.Tensor], layer_number: int, config: Dict[str, Any]) -> sympy.Matrix:
    """
    Generate symbolic output for the specified layer using state dict directly.

    Args:
        state_dict: The model's state dictionary
        layer_number: The target layer number (1-indexed)
        config: The model's configuration dictionary

    Returns:
        sympy.Matrix: Symbolic output for the specified layer.
    """
    logger.debug(f"Generating symbolic output for layer {layer_number}")

    # Get input features from state dict, accounting for polynomial layer
    in_features = None
    if config.get('use_polynomial', False) and layer_number == 1:
        # For polynomial layer, get input features from the next layer's weight
        for name, param in state_dict.items():
            if f'net.1.0.weight' in name:  # Look at the next layer after polynomial
                in_features = (param.shape[1]) // config.get('poly_degree', 2)
                break
    else:
        # Normal layer feature detection
        for name, param in state_dict.items():
            if f'net.{layer_number-1}.0.weight' in name:
                in_features = param.shape[1]
                break

    if in_features is None:
        raise ValueError(f"Could not determine input features for layer {layer_number}")

    # Initialize symbolic inputs
    input_symbols = sympy.Matrix([
        sympy.symbols(f"x_{layer_number}_{i+1}") for i in range(in_features)
    ])
    current_output = input_symbols

    # Process layer using state dict values
    if config.get('use_polynomial', False) and layer_number == 1:  # Polynomial layer
        # Apply polynomial transformation to input features
        inputs = sympy.Matrix(current_output)
        poly_terms = []
        
        # Apply polynomial transformation only to states
        for i in range(1, config.get('poly_degree', 2) + 1):
            if i == 1:
                poly_terms.append(inputs)  # Linear terms for states
            else:
                transformer = SymbolicPolynomialTransform(i)
                poly_terms.append(inputs.applyfunc(transformer))
                
        current_output = sympy.Matrix.vstack(*poly_terms)
    
    # Get weight and bias from state dict and apply pruning masks if they exist
    weight_key = f'net.{layer_number-1}.0.weight'
    bias_key = f'net.{layer_number-1}.0.bias'
    
    if weight_key in state_dict and bias_key in state_dict:
        weight = state_dict[weight_key].detach().cpu().numpy()
        # Look for mask with new naming format
        mask_key = f'mask_net_{layer_number-1}_0_weight'
        if mask_key in state_dict:
            weight *= state_dict[mask_key].detach().cpu().numpy()
        
        weight = sympy.Matrix(weight)
        bias = sympy.Matrix(state_dict[bias_key].detach().cpu().numpy())
        
        # Add dimension check
        if current_output.shape[0] == 0 or weight.shape[1] != current_output.shape[0]:
            raise ValueError(f"Matrix dimension mismatch: weight matrix has shape {weight.shape}, "
                           f"but current output has shape {current_output.shape}")
            
        current_output = weight * current_output + bias

        # Check if this is the last layer by looking for next layer's weights
        is_last_layer = True
        for key in state_dict:
            if f'net.{layer_number}.0.weight' in key:
                is_last_layer = False
                break

        # Apply activation if not the last layer
        if not is_last_layer:
            act_type = config.get('activation_type')
            if act_type == 'sine':
                frequency = config.get('sine_frequency', 30.0)
                current_output = current_output.applyfunc(lambda x: sine_transform(x, frequency))
            elif act_type == 'relu':
                current_output = current_output.applyfunc(lambda x: sympy.Max(0, x))
            elif act_type == 'relu_primitive':
                current_output = current_output.applyfunc(relu_primitive_transform)
            elif act_type == 'gelu':
                # Approximate GELU symbolically using its definition
                # GELU(x) ≈ 0.5x(1 + tanh(sqrt(2/π)(x + 0.044715x^3)))
                current_output = current_output.applyfunc(
                    lambda x: 0.5 * x * (1 + sympy.tanh(
                        sympy.sqrt(2/sympy.pi) * (x + 0.044715 * x**3)
                    ))
                )

    logger.info(f"Successfully generated symbolic output for layer {layer_number}")
    return current_output

def compute_layer(state_dict: Dict[str, torch.Tensor], config: Dict[str, Any], layer_number: int) -> sympy.Matrix:
    """Compute symbolic output for a specific layer using state dict."""
    return get_symbolic_layer_output_generalized(
        state_dict, 
        layer_number,
        config
    )

def parallel_substitution_task(args) -> sympy.Expr:
    """
    Perform parallel symbolic substitution for a single expression.
    Args:
        args (tuple): (expression, substitution_map)

    Returns:
        sympy.Expr: The substituted expression.
    """
    expr, substitution_map = args
    # Using xreplace here for fast, dictionary-based substitution
    return expr.xreplace(substitution_map)

def combine_all_layers_parallelized(state_dict: Dict[str, torch.Tensor], config: Dict[str, Any], simplify: bool = False) -> sympy.Matrix:
    """
    Combine all layers using state dict directly.

    Optimizations:
    1. Batch substitution for faster replacements.
    2. Parallel computation for symbolic substitutions using ProcessPoolExecutor.
    3. Simplify symbolic expressions incrementally (optional).
    
    Args:
        state_dict: The model's state dictionary
        config: The model's configuration dictionary
        simplify: Whether to simplify expressions incrementally (default: False)

    Returns:
        sympy.Matrix: Combined symbolic representation of all layers
    """
    
    
    # Determine number of layers from state dict
    layer_indices = {int(key.split('.')[1]) for key in state_dict if '.weight' in key}
    num_layers = max(layer_indices) + 1

    symbolic_outputs = [None] * num_layers

    # Compute symbolic outputs in parallel
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(compute_layer, state_dict, config, i + 1) for i in range(num_layers)]
        for i, future in enumerate(futures):
            symbolic_outputs[i] = future.result()

    # Validate outputs
    for i, symbolic_output in enumerate(symbolic_outputs, start=1):
        if symbolic_output is None:
            raise ValueError(f"Failed to generate symbolic output for layer {i}.")
        logger.debug(f"Layer {i} symbolic output generated.")

    # Combine all symbolic outputs
    combined_symbolic_model = symbolic_outputs[0]

    for layer_number in range(2, num_layers + 1):
        symbolic_output_next = symbolic_outputs[layer_number - 1]
        
        num_subs = len(combined_symbolic_model)
        # Generate input symbols for the next layer
        input_symbols_next_layer = [
            sympy.symbols(f"x_{layer_number}_{i + 1}") 
            for i in range(num_subs)
        ]

        # Create substitution map and prepare parallel processing
        substitution_map = dict(zip(input_symbols_next_layer, combined_symbolic_model))
        substituted_outputs = [expr.xreplace(substitution_map) for expr in symbolic_output_next]

        combined_symbolic_model = sympy.Matrix(substituted_outputs)

        if simplify:
            combined_symbolic_model = sympy.simplify(combined_symbolic_model)

    logger.info(f"All {num_layers} layers combined successfully.")
    return combined_symbolic_model

def extract_symbolic_model(state_dict: Dict[str, torch.Tensor], config: Dict[str, Any], save_path: str) -> str:
    """
    Extracts a symbolic representation from the model state dictionary and config.

    Args:
        state_dict (Dict[str, torch.Tensor]): The model's state dictionary
        config (Dict[str, Any]): The model's configuration dictionary
        save_path (str): Path to save the symbolic model

    Returns:
        str: The extracted symbolic model as a string
    """
    logger.info("Starting symbolic model extraction")
    
    # Get and save symbolic expression
    symbolic_expression = combine_all_layers_parallelized(state_dict, config)

    symbolic_file = os.path.join(save_path, "symbolic_model.txt")
    try:
        with open(symbolic_file, "w") as f:
            f.write(str(symbolic_expression))
        logger.info(f"Symbolic model saved to {symbolic_file}")
    except IOError as e:
        logger.error(f"Failed to save symbolic model: {e}")
        raise

    return symbolic_expression

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

def simplify_heaviside_expressions(expr: Union[sympy.Expr, sympy.Matrix]) -> Union[sympy.Expr, sympy.Matrix]:
    """
    Simplifies expressions containing Heaviside functions with redundant patterns.
    
    Common simplifications include:
    - max(0,a)*Heaviside(a) -> max(0,a)
    - Heaviside(a)*max(0,a) -> max(0,a)
    - Heaviside(a)*a when a >= 0 -> a
    
    Args:
        expr: A sympy expression or matrix to simplify
        
    Returns:
        Simplified sympy expression or matrix
    """
    logger.debug("Simplifying expressions with Heaviside functions")
    
    # Helper function to process individual expressions
    def _simplify_expr(e):
        if not isinstance(e, sympy.Expr):
            return e
            
        # Handle matrix expressions
        if isinstance(e, sympy.Matrix):
            return sympy.Matrix([_simplify_expr(item) for item in e])
            
        # Look for multiplication patterns
        if isinstance(e, sympy.Mul):
            args = e.args
            
            # Check for Heaviside(a) * max(0, a) or max(0, a) * Heaviside(a)
            heaviside_args = [arg for arg in args if isinstance(arg, sympy.Heaviside)]
            max_args = [arg for arg in args if isinstance(arg, sympy.Max) and len(arg.args) == 2 and arg.args[0] == 0]
            
            if heaviside_args and max_args:
                for heaviside in heaviside_args:
                    for max_term in max_args:
                        # If Heaviside(a) and Max(0,a) share the same argument
                        if heaviside.args[0] == max_term.args[1]:
                            # Remove the Heaviside term and keep the Max term
                            remaining_args = [arg for arg in args if arg != heaviside]
                            if len(remaining_args) == 1:
                                return remaining_args[0]
                            return sympy.Mul(*remaining_args)
            
            # Check for Heaviside(a) * a
            for heaviside in heaviside_args:
                heaviside_arg = heaviside.args[0]
                if heaviside_arg in args:
                    # Replace Heaviside(a) * a with just max(0, a)
                    remaining_args = [arg for arg in args if arg != heaviside and arg != heaviside_arg]
                    max_term = sympy.Max(0, heaviside_arg)
                    if remaining_args:
                        return sympy.Mul(*remaining_args, max_term)
                    return max_term
        
        # Recursively process only if the expression contains Heaviside, otherwise return as is
        if e.args:
            if e.has(sympy.Heaviside):
                return e.func(*[_simplify_expr(arg) for arg in e.args])
            else:
                return e
        return e
    
    # Apply the helper function to the expression or matrix
    if isinstance(expr, sympy.Matrix):
        return expr.applyfunc(_simplify_expr)
    return _simplify_expr(expr)

def compute_partial_deriv(final_symbolic_expression: sympy.Matrix, input_symbols: list):
    # Compute symbolic partial derivatives
    partials = [final_symbolic_expression.diff(var) for var in input_symbols]

    # Iteratively simplify Heaviside expressions until convergence
    def iterative_simplify(expr):
        while expr.has(sympy.Heaviside):
            expr = simplify_heaviside_expressions(expr)
        return expr

    partials = [iterative_simplify(p) for p in partials]

    return partials