import os
import torch
import sympy
from concurrent.futures import ProcessPoolExecutor
import logging

logger = logging.getLogger(__name__)

def sine_transform(x):
    """Helper function for sine transformation."""
    return sympy.sin(30 * x)

def power_transform(x, power):
    """Helper function for polynomial transformation."""
    return x**power

class SymbolicPolynomialTransform:
    """A picklable class for polynomial transformations."""
    def __init__(self, power):
        self.power = power
    
    def __call__(self, x):
        return power_transform(x, self.power)

def get_symbolic_layer_output_generalized(state_dict, layer_number, use_polynomial=False, poly_degree=2):
    """
    Generate symbolic output for the specified layer using state dict directly.

    Args:
        state_dict: The model's state dictionary
        layer_number: The target layer number (1-indexed)
        use_polynomial: Whether the layer uses polynomial features
        poly_degree: Degree of polynomial features if used

    Returns:
        sympy.Matrix: Symbolic output for the specified layer.
    """
    logger.debug(f"Generating symbolic output for layer {layer_number}")

    # Get input features from state dict
    in_features = None
    for name, param in state_dict.items():
        if f'net.{layer_number-1}' in name and 'weight' in name:
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
    if use_polynomial and layer_number == 1:  # Polynomial layer is always first if used
        poly_terms = []
        for i in range(1, poly_degree + 1):
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

        # Check for activation in the layer structure
        if any('sine' in k for k in state_dict.keys()):  # This is a sine network
            current_output = current_output.applyfunc(sine_transform)

    logger.info(f"Successfully generated symbolic output for layer {layer_number}")
    return current_output

def compute_layer(state_dict, config, layer_number):
    """Compute symbolic output for a specific layer using state dict."""
    return get_symbolic_layer_output_generalized(
        state_dict, 
        layer_number,
        use_polynomial=config.get('use_polynomial', False),
        poly_degree=config.get('poly_degree', 2)
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
    return expr.subs(substitution_map, simultaneous=True)

def combine_all_layers_parallelized(state_dict, config, num_layers, simplify=False):
    """
    Combine all layers using state dict directly.
    
    Args:
        state_dict: The model's state dictionary
        config: The model's configuration dictionary
        num_layers: Number of layers in the model
        simplify: Whether to simplify expressions incrementally (default: False)

    Returns:
        sympy.Matrix: Combined symbolic representation of all layers
    """
    symbolic_outputs = [None] * num_layers

    # Compute symbolic outputs in parallel
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(compute_layer, state_dict, config, i + 1): i 
            for i in range(num_layers)
        }
        for future in futures:
            layer_idx = futures[future]
            symbolic_outputs[layer_idx] = future.result()

    # Validate outputs
    for i, symbolic_output in enumerate(symbolic_outputs, start=1):
        if symbolic_output is None:
            raise ValueError(f"Failed to generate symbolic output for layer {i}.")
        logger.debug(f"Layer {i} symbolic output generated.")

    # Combine all symbolic outputs
    combined_symbolic_model = symbolic_outputs[0]

    for layer_number in range(2, num_layers + 1):
        symbolic_output_next = symbolic_outputs[layer_number - 1]
        
        # Generate input symbols for the next layer
        input_symbols_next_layer = [
            sympy.symbols(f"x_{layer_number}_{i + 1}") 
            for i in range(len(combined_symbolic_model))
        ]

        # Create substitution map and prepare parallel processing
        substitution_map = dict(zip(input_symbols_next_layer, combined_symbolic_model))
        substitution_args = [(expr, substitution_map) for expr in symbolic_output_next]

        # Parallel substitution
        with ProcessPoolExecutor() as executor:
            substituted_outputs = list(executor.map(parallel_substitution_task, substitution_args))

        combined_symbolic_model = sympy.Matrix(substituted_outputs)

        if simplify:
            combined_symbolic_model = sympy.simplify(combined_symbolic_model)

        logger.debug(f"Layer {layer_number} combined. Total layers combined so far: {layer_number}.")

    logger.info(f"All {num_layers} layers combined successfully.")
    return combined_symbolic_model

def extract_symbolic_model(model, save_path):
    """
    Extracts a symbolic representation from the trained model and saves it.

    Args:
        model (torch.nn.Module): The trained PyTorch model
        save_path (str): Path to save the symbolic model

    Returns:
        str: The extracted symbolic model as a string
    """
    logger.info("Starting symbolic model extraction")
    
    # Extract necessary components
    state_dict = model.state_dict()
    config = model.config.to_dict()
    
    # Determine number of layers from state dict
    layer_indices = {int(key.split('.')[1]) for key in state_dict if '.weight' in key}
    num_layers = max(layer_indices) + 1
    
    # Get and save symbolic expression
    symbolic_expression = combine_all_layers_parallelized(
        state_dict, config, num_layers, simplify=False
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

