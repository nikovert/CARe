import os
import torch
import sympy
import torch
from concurrent.futures import ProcessPoolExecutor
import sympy
import logging

logger = logging.getLogger(__name__)

def get_symbolic_layer_output_generalized(model, layer_number):
    """
    Generate symbolic output for the specified layer of a model, 
    handling FCBlock, BatchLinear, Sine, and PolynomialLayer.

    Args:
        model (nn.Module): The PyTorch model (e.g., SingleBVPNet).
        layer_number (int): The target layer number (1-indexed).

    Returns:
        sympy.Matrix: Symbolic output for the specified layer.
    """
    logger.debug(f"Generating symbolic output for layer {layer_number}")

    # ========================= #
    # Step 1: Access the FCBlock
    # ========================= #
    if not hasattr(model, "net") or not hasattr(model.net, "net"):
        logger.error("Model does not have the expected structure")
        raise ValueError("Model does not have the expected structure (FCBlock with Sequential).")
    fc_block = model.net.net

    # Ensure FCBlock is a Sequential Module
    if not isinstance(fc_block, torch.nn.Sequential):
        raise ValueError("FCBlock does not contain a Sequential module.")
    main_sequential = fc_block

    # Check if the layer number is valid
    if layer_number - 1 >= len(main_sequential):
        raise ValueError(f"Layer number {layer_number} exceeds the number of layers in the Sequential.")
    
    # ========================= #
    # Step 2: Access Target Layer
    # ========================= #
    target_layer = main_sequential[layer_number - 1]

    # If the target layer is a Sequential, use its sublayers
    if isinstance(target_layer, torch.nn.Sequential):
        sublayers = list(target_layer.children())
    else:
        sublayers = [target_layer]  # Direct Layer Access (e.g., PolynomialLayer)

    # ========================= #
    # Step 3: Extract Input Features
    # ========================= #
    in_features = None
    for sublayer in sublayers:
        # Extract features from BatchLinear or Linear
        if hasattr(sublayer, "weight"):
            in_features = sublayer.weight.shape[1]
            break
        # Extract features from PolynomialLayer
        elif "PolynomialLayer" in type(sublayer).__name__:
            in_features = sublayer.in_features
            break

    if in_features is None:
        raise ValueError(f"No input features found in layer {layer_number}.")

    # Initialize Symbolic Inputs
    input_symbols = sympy.Matrix([
        sympy.symbols(f"x_{layer_number}_{i+1}") for i in range(in_features)
    ])
    current_output = input_symbols

    # ========================= #
    # Step 4: Process Sublayers
    # ========================= #
    for sublayer in sublayers:

        # Handle BatchLinear Layer
        if hasattr(sublayer, "weight") and hasattr(sublayer, "bias"):
            weight = sympy.Matrix(sublayer.weight.detach().cpu().numpy())
            bias = sympy.Matrix(sublayer.bias.detach().cpu().numpy())
            current_output = weight * current_output + bias

        # Handle Sine Activation
        elif "Sine" in type(sublayer).__name__:
            current_output = current_output.applyfunc(lambda x: sympy.sin(30 * x))

        # Handle Polynomial Layer
        elif "PolynomialLayer" in type(sublayer).__name__:
            degree = sublayer.degree

            # Apply elementwise exponentiation
            poly_terms = [current_output.applyfunc(lambda x: x**i) for i in range(1, degree + 1)]
            
            # Concatenate polynomial features
            current_output = sympy.Matrix.vstack(*poly_terms)

        # Handle Unsupported Layers
        else:
            raise ValueError(f"Unsupported layer type: {type(sublayer).__name__}")

    logger.info(f"Successfully generated symbolic output for layer {layer_number}")
    return current_output


def compute_layer(model, layer_number):
    """
    Compute symbolic output for a specific layer of the model.
    Args:
        model (nn.Module): The PyTorch model.
        layer_number (int): The target layer number (1-based index).

    Returns:
        sympy.Matrix: Symbolic representation of the layer's output.
    """
    return get_symbolic_layer_output_generalized(model, layer_number)


def parallel_substitution_task(expr, substitution_map):
    """
    Perform parallel symbolic substitution for a single expression.
    Args:
        expr (sympy.Expr): The symbolic expression to substitute.
        substitution_map (dict): The substitution map.

    Returns:
        sympy.Expr: The substituted expression.
    """
    return expr.subs(substitution_map, simultaneous=True)


def combine_all_layers_parallelized(model, simplify=False):
    """
    Combine all layers in a model into a single symbolic representation with optimizations.
    
    Optimizations:
    1. Batch substitution for faster replacements.
    2. Parallel computation for symbolic substitutions using ProcessPoolExecutor.
    3. Simplify symbolic expressions incrementally (optional).

    Args:
        model (nn.Module): The PyTorch model (e.g., SingleBVPNet).
        simplify (bool): Whether to simplify symbolic expressions incrementally. Default is False.

    Returns:
        sympy.Matrix: Symbolic representation of the entire model.
    """
    # Step 1: Check Total Number of Layers
    if not hasattr(model, "net") or not hasattr(model.net, "net"):
        raise ValueError("Model does not have the expected structure (FCBlock with Sequential).")

    num_layers = len(model.net.net)  # Total number of layers in the Sequential module
    symbolic_outputs = [None] * num_layers

    # Step 2: Compute Symbolic Outputs in Parallel
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(compute_layer, model, i + 1): i for i in range(num_layers)}
        for future in futures:
            layer_idx = futures[future]
            symbolic_outputs[layer_idx] = future.result()

    # Step 3: Validate Outputs
    for i, symbolic_output in enumerate(symbolic_outputs, start=1):
        if symbolic_output is None:
            raise ValueError(f"Failed to generate symbolic output for layer {i}.")
        logger.debug(f"Layer {i} symbolic output generated.")

    # Step 4: Combine All Symbolic Outputs
    combined_symbolic_model = symbolic_outputs[0]  # Start with the first layer's symbolic output

    for layer_number in range(2, num_layers + 1):
        symbolic_output_next = symbolic_outputs[layer_number - 1]

        # Generate input symbols for the next layer
        input_symbols_next_layer = [
            sympy.symbols(f"x_{layer_number}_{i + 1}") for i in range(len(combined_symbolic_model))
        ]

        # Create the substitution map
        substitution_map = {
            symbol: output for symbol, output in zip(input_symbols_next_layer, combined_symbolic_model)
        }

        # Parallel substitution for all terms
        with ProcessPoolExecutor() as executor:
            substituted_outputs = list(
                executor.map(parallel_substitution_task, symbolic_output_next, [substitution_map] * len(symbolic_output_next))
            )

        # Convert substituted outputs back to a SymPy Matrix
        combined_symbolic_model = sympy.Matrix(substituted_outputs)

        # Optional: Simplify combined output incrementally
        if simplify:
            combined_symbolic_model = sympy.simplify(combined_symbolic_model)

        logger.debug(f"Layer {layer_number} combined. Total layers combined so far: {layer_number}.")

    logger.info(f"All {num_layers} layers combined successfully.")
    return combined_symbolic_model


def extract_symbolic_model(model, save_path):
    """
    Extracts a symbolic representation from the trained model and saves it.

    Args:
        model (torch.nn.Module): The trained PyTorch model.
        save_path (str): Path to the experiment folder where the symbolic model will be saved.

    Returns:
        str: The extracted symbolic model as a string.
    """
    logger.info("Starting symbolic model extraction")
    symbolic_expression = combine_all_layers_parallelized(model, simplify=False)

    # Save the symbolic model
    symbolic_file = os.path.join(save_path, "symbolic_model.txt")
    try:
        with open(symbolic_file, "w") as f:
            f.write(str(symbolic_expression))
        logger.info(f"Symbolic model saved to {symbolic_file}")
    except IOError as e:
        logger.error(f"Failed to save symbolic model: {e}")
        raise

    return symbolic_expression

