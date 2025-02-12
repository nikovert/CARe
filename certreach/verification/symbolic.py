import os
import sympy
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Any
import logging
import torch

logger = logging.getLogger(__name__)

def sine_transform(x, frequency=30.0):
    """Helper function for sine transformation."""
    return sympy.sin(frequency * x)

def power_transform(x, power):
    """Helper function for polynomial transformation."""
    return x**power

class SymbolicPolynomialTransform:
    """A picklable class for polynomial transformations."""
    def __init__(self, power):
        self.power = power
    
    def __call__(self, x):
        return power_transform(x, self.power)

def get_symbolic_layer_output_generalized(state_dict, layer_number, config):
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
                in_features = param.shape[1] // config.get('poly_degree', 2)
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
    if config.get('use_polynomial', False) and layer_number == 1:  # Polynomial layer is always first if used
        poly_terms = []
        for i in range(1, config.get('poly_degree', 2) + 1):
            transformer = SymbolicPolynomialTransform(i)
            poly_terms.append(current_output.applyfunc(transformer))
        current_output = sympy.Matrix.vstack(*poly_terms)
    
    # Get weight and bias from state dict
    weight_key = f'net.{layer_number-1}.0.weight'
    bias_key = f'net.{layer_number-1}.0.bias'
    
    if weight_key in state_dict and bias_key in state_dict:
        weight = sympy.Matrix(state_dict[weight_key].detach().cpu().numpy())
        bias = sympy.Matrix(state_dict[bias_key].detach().cpu().numpy())
        current_output = weight * current_output + bias

        # Check for activation and use correct frequency
        if any('sine' in k for k in state_dict.keys()):
            frequency = config.get('sine_frequency', 30.0)
            current_output = current_output.applyfunc(lambda x: sine_transform(x, frequency))

    logger.info(f"Successfully generated symbolic output for layer {layer_number}")
    return current_output

def compute_layer(state_dict, config, layer_number):
    """Compute symbolic output for a specific layer using state dict."""
    return get_symbolic_layer_output_generalized(
        state_dict, 
        layer_number,
        config
    )

def parallel_substitution_task(args):
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

def combine_all_layers_parallelized(state_dict, config, simplify=False):
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

        logger.debug(f"Layer {layer_number} combined. Total layers combined so far: {layer_number}.")

    logger.info(f"All {num_layers} layers combined successfully.")
    return combined_symbolic_model

def extract_symbolic_model(state_dict: Dict[str, torch.Tensor], config: Dict[str, Any], save_path: str):
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
    symbolic_expression = combine_all_layers_parallelized(
        state_dict, config, simplify=False
    )

    symbolic_file = os.path.join(save_path, "symbolic_model.txt")
    try:
        with open(symbolic_file, "w") as f:
            f.write(str(symbolic_expression))
        logger.info(f"Symbolic model saved to {symbolic_file}")
    except IOError as e:
        logger.error(f"Failed to save symbolic model: {e}")
        raise

    return symbolic_expression
