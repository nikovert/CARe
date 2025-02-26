import torch
import torch.nn as nn
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)

class StandardMLP(nn.Module):
    """
    A standard MLP implementation that can be used for ONNX export.
    This model uses standard PyTorch modules for better compatibility.
    """
    
    def __init__(self, 
                 layers: List[nn.Module],
                 input_size: int,
                 output_size: int,
                 hidden_sizes: List[int],
                 activations: List[nn.Module]) -> None:
        """
        Initialize a standard MLP model.
        
        Args:
            layers: The linear layers to use (weights and biases will be copied from these)
            input_size: Size of input features
            output_size: Size of output features
            hidden_sizes: Sizes of hidden layers
            activations: Activation functions to use after each hidden layer
        """
        super(StandardMLP, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        
        # Create the model structure
        model_layers = []
        
        # Add input layer and first hidden layer
        linear1 = nn.Linear(input_size, hidden_sizes[0])
        model_layers.append(linear1)
        
        if activations and len(activations) > 0:
            model_layers.append(activations[0])
        
        # Add hidden layers
        for i in range(1, len(hidden_sizes)):
            linear = nn.Linear(hidden_sizes[i-1], hidden_sizes[i])
            model_layers.append(linear)
            
            if i < len(activations):
                model_layers.append(activations[i])
        
        # Add output layer
        linear_out = nn.Linear(hidden_sizes[-1], output_size)
        model_layers.append(linear_out)
        
        # Create sequential model
        self.model = nn.Sequential(*model_layers)
        
        # Copy weights and biases from original layers
        self._copy_weights_from_layers(layers)
    
    def _copy_weights_from_layers(self, source_layers: List[nn.Module]) -> None:
        """
        Copy weights and biases from source layers to this model's layers.
        
        Args:
            source_layers: List of source layers to copy weights and biases from
        """
        linear_idx = 0
        layer_idx = 0
        
        # Iterate through model layers
        for module in self.model:
            if isinstance(module, nn.Linear):
                # Find corresponding source layer
                source_linear = None
                while layer_idx < len(source_layers) and source_linear is None:
                    if hasattr(source_layers[layer_idx], 'weight'):
                        source_linear = source_layers[layer_idx]
                    layer_idx += 1
                
                if source_linear is not None:
                    # Copy weights and biases
                    with torch.no_grad():
                        module.weight.copy_(source_linear.weight)
                        if hasattr(source_linear, 'bias') and source_linear.bias is not None:
                            module.bias.copy_(source_linear.bias)
                    
                linear_idx += 1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.
        
        Args:
            x: Input tensor [batch_size, input_size]
            
        Returns:
            Output tensor [batch_size, output_size]
        """
        return self.model(x)

def convert_to_standard_mlp(model: torch.nn.Module, verbose: bool = False) -> StandardMLP:
    """
    Convert a SingleBVPNet to a standard MLP for better ONNX export compatibility.
    
    Args:
        model: The SingleBVPNet model to convert
        verbose: If True, print verbose output
        
    Returns:
        A StandardMLP model with copied weights and biases
    """
    # Check if we're dealing with a SingleBVPNet
    if not hasattr(model, 'net') or not hasattr(model, 'config'):
        raise ValueError("Input model must be a SingleBVPNet with 'net' and 'config' attributes")
    
    # Extract model architecture info
    config = model.config
    in_features = config.in_features
    out_features = config.out_features
    hidden_features = config.hidden_features
    num_layers = config.num_hidden_layers
    
    # Get activation function type
    activation_type = config.activation_type
    
    # Create activation layers
    activation_layers = []
    
    # Map activation types to standard PyTorch activations
    activation_map = {
        'relu': nn.ReLU(),
        'leaky_relu': nn.LeakyReLU(0.01),
        'sigmoid': nn.Sigmoid(),
        'tanh': nn.Tanh(),
        'gelu': nn.GELU(),
        'softplus': nn.Softplus(),
        'elu': nn.ELU(),
        'selu': nn.SELU(),
    }
    
    # Get appropriate activation
    activation = activation_map.get(activation_type.lower(), nn.ReLU())
    
    # Create hidden layer activations
    for _ in range(num_layers + 1):  # +1 for input layer
        activation_layers.append(activation)
    
    # Extract all linear layers (collect weights and biases)
    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or hasattr(module, 'weight'):
            linear_layers.append(module)
    
    # Create hidden sizes list
    hidden_sizes = [hidden_features] * (num_layers + 1)
    
    if verbose:
        print(f"Converting model with: input={in_features}, output={out_features}")
        print(f"Hidden sizes: {hidden_sizes}")
        print(f"Activation: {activation_type}")
        print(f"Found {len(linear_layers)} linear layers")
    
    # Create standard MLP
    standard_mlp = StandardMLP(
        layers=linear_layers,
        input_size=in_features,
        output_size=out_features,
        hidden_sizes=hidden_sizes,
        activations=activation_layers
    )
    
    # Verify that the converted model produces the same output as the original
    device = next(model.parameters()).device
    standard_mlp.to(device)
    model.eval()
    standard_mlp.eval()
    
    with torch.no_grad():
        test_input = torch.randn(1, in_features, device=device)
        
        # Get output from original SingleBVPNet model (handles the standard interface)
        original_output = model({'coords': test_input})['model_out']
        
        # Get output from standard model
        standard_output = standard_mlp(test_input)
        
        # Assert that outputs are very close
        assert torch.allclose(original_output, standard_output, rtol=1e-4, atol=1e-5), \
            "Converted model produces different outputs than the original model"
    
    return standard_mlp

def export_to_onnx(model: torch.nn.Module,
                 file_path: str, 
                 input_shape: Optional[Tuple[int, ...]] = None, 
                 batch_size: int = 1) -> None:
    """
    Export a PyTorch model to ONNX format.
    
    Args:
        model: The PyTorch model to export
        file_path: Path where the ONNX file will be saved
        input_shape: Shape of a single input sample (excluding batch dimension)
        batch_size: Batch size for the dummy input tensor
        input_names: Names of the input nodes in the exported model
        output_names: Names of the output nodes in the exported model
        dynamic_axes: Dictionary specifying dynamic axes for inputs/outputs
        opset_version: ONNX opset version to use for export
        verbose: If True, will print verbose output during export
        input_key: If the model expects dictionary input, this is the key for the input tensor
        output_key: If the model returns dictionary output, this is the key for the output tensor
        
    Returns:
        None
    """
        
    # Determine device
    device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device('cpu')
    
    # Infer input shape if not provided
    if input_shape is None:
        if hasattr(model, 'config') and hasattr(model.config, 'in_features'):
            input_shape = (model.config.in_features,)
        else:
            # Try to determine from the model's parameters
            for name, param in model.named_parameters():
                if 'weight' in name and len(param.shape) >= 2:
                    input_shape = (param.shape[1],)
                    break
            if input_shape is None:
                raise ValueError("Could not infer input shape. Please provide input_shape parameter.")
                
    # Create dummy input tensor
    dummy_input = torch.randn(batch_size, *input_shape, device=device)
    simplified_model = convert_to_standard_mlp(model)
    simplified_model.to(device)
    simplified_model.eval()

    # Handle file extension
    if not file_path.endswith('.onnx'):
        file_path += '.onnx'
    
    
    # Export using dynamo_export
    exported_model = torch.onnx.dynamo_export(
        simplified_model, 
        dummy_input
    )
    
    # Save the model
    exported_model.save(file_path)

