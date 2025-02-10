import torch
from collections import OrderedDict
import math
from pathlib import Path
from dataclasses import dataclass, fields
from typing import Optional, Literal, Dict, Any, Union, Tuple
from ray import tune
import torch.nn.init as init

@dataclass
class NetworkConfig:
    """Configuration for neural network architecture and training."""
    in_features: int
    out_features: int = 1
    hidden_features: Union[int, tune.grid_search, tune.choice] = 32
    num_hidden_layers: Union[int, tune.grid_search, tune.choice] = 3
    activation_type: Union[Literal['sine', 'relu', 'sigmoid', 'tanh', 'selu', 'softplus', 'elu'], 
                         tune.grid_search, tune.choice] = 'sine'
    mode: str = 'mlp'
    use_polynomial: Union[bool, tune.grid_search, tune.choice] = False
    poly_degree: Union[int, tune.grid_search, tune.choice] = 2
    initialization_scale: Union[float, tune.uniform, tune.loguniform] = 1.0
    first_layer_initialization_scale: Union[float, tune.uniform, tune.loguniform] = 30.0
    dropout_rate: Union[float, tune.uniform] = 0.0
    use_batch_norm: Union[bool, tune.grid_search, tune.choice] = False
    sine_frequency: Union[float, tune.uniform] = 30.0  # New parameter
    
    def __init__(self, **kwargs):
        """Initialize NetworkConfig with flexible keyword arguments."""
        # Filter out unknown arguments
        known_fields = set(f.name for f in fields(self))
        valid_kwargs = {k: v for k, v in kwargs.items() if k in known_fields}
        
        # Initialize with filtered arguments
        for name, value in valid_kwargs.items():
            setattr(self, name, value)
            
        # Set required fields that weren't provided
        if 'in_features' not in valid_kwargs:
            raise ValueError("in_features is required")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items()}
    
    @staticmethod
    def get_tune_config():
        """Get a configuration space for Ray Tune."""
        return {
            "hidden_features": tune.choice([16, 32, 64, 128]),
            "num_hidden_layers": tune.choice([2, 3, 4, 5]),
            "activation_type": tune.choice(['sine', 'relu', 'tanh']),
            "use_polynomial": tune.choice([True, False]),
            "poly_degree": tune.choice([2, 3]),
            "initialization_scale": tune.loguniform(0.1, 10.0),
            "first_layer_initialization_scale": tune.loguniform(1.0, 100.0),
            "dropout_rate": tune.uniform(0.0, 0.5),
            "use_batch_norm": tune.choice([True, False])
        }

class BatchLinear(torch.nn.Linear):
    '''A linear layer'''
    __doc__ = torch.nn.Linear.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        bias = params.get('bias', None)
        weight = params['weight']

        output = input.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))
        output += bias.unsqueeze(-2)
        return output

class Sine(torch.nn.Module):
    def __init__(self, frequency=30.0):
        super().__init__()
        self.frequency = frequency

    def forward(self, input):
        """
        See Sitzmann et al. (2020) "Implicit Neural Representations with Periodic Activation Functions"
        The frequency factor allows for tuning of the periodic activation.
        """
        return torch.sin(self.frequency * input)

class PolynomialFunction(torch.autograd.Function):
    """Custom autograd function for polynomial computation."""
    
    @staticmethod
    def forward(ctx, input, degree):
        """
        Compute polynomial terms up to the specified degree.
        
        Args:
            ctx: Context object to save variables for backward
            input (torch.Tensor): Input tensor
            degree (int): Highest degree of polynomial
        """
        ctx.degree = degree
        ctx.save_for_backward(input)
        
        # Compute polynomial terms [x, x^2, ..., x^degree]
        results = [input**i for i in range(1, degree + 1)]
        return torch.cat(results, dim=-1)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Compute gradient of the loss with respect to the input.
        
        Args:
            ctx: Context object with saved variables
            grad_output: Gradient of the loss with respect to the output
        """
        input, = ctx.saved_tensors
        degree = ctx.degree
        
        # Split grad_output into chunks corresponding to each polynomial term
        chunks = grad_output.chunk(degree, dim=-1)
        
        # Compute gradient for each term: d(x^n)/dx = n * x^(n-1)
        grad_input = torch.zeros_like(input)
        for i, grad in enumerate(chunks):
            power = i + 1  # polynomial term degree (1-based)
            grad_input += power * (input ** (power - 1)) * grad
            
        return grad_input, None  # None for degree parameter

class PolynomialLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, degree=2):
        """
        A custom layer that computes polynomial features using custom backward propagation.

        Args:
            in_features (int): Number of input features
            out_features (int): Number of output features (in_features * degree)
            degree (int): Highest degree of polynomial terms
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        
        if out_features != in_features * degree:
            raise ValueError(f"out_features must be in_features * degree. "
                           f"Got {out_features} != {in_features} * {degree}")

    def forward(self, x):
        """
        Applies polynomial transformation using custom autograd function.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features)

        Returns:
            torch.Tensor: Concatenated tensor with shape (batch_size, in_features * degree)
        """
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Expected input with {self.in_features} features, got {x.shape[-1]}")
            
        return PolynomialFunction.apply(x, self.degree)

# Shared activation configurations
ACTIVATION_CONFIGS: Dict[str, Tuple[torch.nn.Module, callable, Optional[callable]]] = {
    'sine': (
        lambda config: Sine(frequency=config.sine_frequency),
        lambda w: init.uniform_(w, 
            -torch.sqrt(torch.tensor(6.0 / w.size(-1))) / 30,
            torch.sqrt(torch.tensor(6.0 / w.size(-1))) / 30),
        lambda w: init.uniform_(w, -1/w.size(-1), 1/w.size(-1))
    ),
    'relu': (
        torch.nn.ReLU(inplace=True),
        lambda w: init.kaiming_normal_(w, nonlinearity='relu'),
        None
    ),
    'tanh': (
        torch.nn.Tanh(),
        lambda w: init.xavier_normal_(w),
        None
    ),
    'sigmoid': (
        torch.nn.Sigmoid(),
        lambda w: init.xavier_normal_(w),
        None
    ),
    'selu': (
        torch.nn.SELU(inplace=True),
        lambda w: init.normal_(w, std=1/math.sqrt(w.size(1))),
        None
    ),
    'elu': (
        torch.nn.ELU(inplace=True),
        lambda w: init.normal_(w, std=math.sqrt(1.5505188080679277)/math.sqrt(w.size(1))),
        None
    ),
    'softplus': (
        torch.nn.Softplus(),
        lambda w: init.kaiming_normal_(w, nonlinearity='relu'),
        None
    )
}

def initialize_weights(module, weight_init):
    """Helper function to initialize weights only for layers with parameters."""
    if hasattr(module, 'weight') and isinstance(module.weight, torch.nn.Parameter):
        weight_init(module.weight)

class SingleBVPNet(torch.nn.Module):
    '''A canonical representation network for a BVP.'''

    def __init__(self, config: Optional[Union[NetworkConfig, Dict[str, Any]]] = None, **kwargs) -> None:
        super().__init__()
        
        try:
            # Convert dict to NetworkConfig if needed
            if isinstance(config, dict):
                config = NetworkConfig(**config)
            elif config is None:
                config = NetworkConfig(**kwargs)
            elif not isinstance(config, NetworkConfig):
                raise ValueError(f"config must be NetworkConfig, dict, or None, got {type(config)}")
                
            self.config = config
            
            # Use shared activation configs
            nl = ACTIVATION_CONFIGS[config.activation_type][0](config) if callable(ACTIVATION_CONFIGS[config.activation_type][0]) else ACTIVATION_CONFIGS[config.activation_type][0]
            init_fn, first_layer_init = ACTIVATION_CONFIGS[config.activation_type][1:]
            self.weight_init = init_fn
            
            # Build network
            layers = []
            current_dim = config.in_features
            
            # Optional polynomial layer
            if config.use_polynomial:
                layers.append(PolynomialLayer(
                    in_features=current_dim,
                    out_features=current_dim * config.poly_degree,
                    degree=config.poly_degree
                ))
                current_dim *= config.poly_degree

            # Input layer
            layers.append(self._create_layer_block(
                current_dim, 
                config.hidden_features,
                nl,
                is_first=True
            ))
            current_dim = config.hidden_features

            # Hidden layers
            for _ in range(config.num_hidden_layers):
                layers.append(self._create_layer_block(
                    current_dim,
                    config.hidden_features,
                    nl
                ))

            # Output layer
            layers.append(self._create_layer_block(
                current_dim,
                config.out_features,
                activation=None,  # No activation for output layer
                is_output=True
            ))

            self.net = torch.nn.Sequential(*layers)

            # Apply initialization
            if self.weight_init is not None:
                for module in self.net.modules():
                    initialize_weights(module, self.weight_init)
            if first_layer_init is not None:
                # Find first linear layer and initialize it
                for module in self.net.modules():
                    if isinstance(module, BatchLinear):
                        initialize_weights(module, first_layer_init)
                        break
            
        except Exception as e:
            raise TypeError(f"Failed to initialize network: {str(e)}")

    def _create_layer_block(self, in_features: int, out_features: int, 
                          activation: Optional[torch.nn.Module] = None,
                          is_first: bool = False,
                          is_output: bool = False) -> torch.nn.Sequential:
        """Create a block of layers including optional BatchNorm and Dropout."""
        layers = []

        # Linear layer
        layers.append(BatchLinear(in_features, out_features))

        # Optional BatchNorm (not on output layer)
        if self.config.use_batch_norm and not is_output:
            layers.append(torch.nn.BatchNorm1d(out_features))

        # Activation (if provided)
        if activation is not None:
            layers.append(activation)

        # Optional Dropout (not on first or output layer)
        if self.config.dropout_rate > 0 and not is_first and not is_output:
            layers.append(torch.nn.Dropout(self.config.dropout_rate))

        return torch.nn.Sequential(*layers)

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration for reconstruction"""
        return self.config.to_dict()

    @property
    def checkpoint_dir(self):
        """Get the checkpoint directory"""
        return self._checkpoint_dir

    @checkpoint_dir.setter
    def checkpoint_dir(self, path):
        """Set the checkpoint directory"""
        self._checkpoint_dir = Path(path)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, name, optimizer=None, epoch=None, **kwargs):
        """Save checkpoint with specified name in the model's checkpoint directory"""
        if self._checkpoint_dir is None:
            raise ValueError("Checkpoint directory not set. Set model.checkpoint_dir first.")
            
        path = self._checkpoint_dir / f"{name}.pth"
        self._save_checkpoint_file(path, optimizer=optimizer, epoch=epoch, **kwargs)
        return path

    def _save_checkpoint_file(self, path, optimizer=None, epoch=None, **kwargs):
        """Internal method to save checkpoint to a specific file"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_config': self.get_config(),
            **kwargs
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch
            
        torch.save(checkpoint, path)

    @staticmethod
    def load_checkpoint(path, device='cpu', eval_mode=True):
        """ 
        Load model checkpoint with proper evaluation mode setting
        
        Args:
            path (str): Path to checkpoint file
            device (str): Device to load model to ('cpu' or 'cuda')
            eval_mode (bool): Whether to set model to evaluation mode
            
        Returns:
            tuple: (model, checkpoint_dict)
        """
        try:
            checkpoint = torch.load(path, map_location=device)
            
            # Set default configuration
            default_config = {
                'mode': 'mlp',
                'activation_type': 'sine',
                'use_polynomial': False,
                'poly_degree': 2
            }
            
            if 'model_config' not in checkpoint:
                # Infer configuration from state dict
                state_dict = checkpoint['model_state_dict']
                input_layer = next(key for key in state_dict.keys() if 'net.0.0.weight' in key)
                output_layer = next(key for key in state_dict.keys() if key.endswith('.weight'))
                in_features = state_dict[input_layer].shape[1]
                out_features = state_dict[output_layer].shape[0]
                hidden_features = state_dict[input_layer].shape[0]
                # Count hidden layers by counting unique layer indices in state dict
                layer_indices = set()
                for key in state_dict.keys():
                    if '.weight' in key:
                        layer_idx = int(key.split('.')[1])
                        layer_indices.add(layer_idx)
                num_hidden_layers = len(layer_indices) - 2  # subtract input and output layers
                model_config = {
                    **default_config,
                    'in_features': in_features,
                    'out_features': out_features,
                    'hidden_features': hidden_features,
                    'num_hidden_layers': max(0, num_hidden_layers)
                }
            else:
                model_config = {**default_config, **checkpoint['model_config']}
            
            # Create and configure model
            model = SingleBVPNet(**model_config)
            model.to(device)
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            
            if eval_mode:
                model.eval()
                
            return model, checkpoint
            
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {str(e)}")
            
    def load_weights(self, state_dict, eval_mode=True):
        """ 
        Load just the model weights
        
        Args:
            state_dict (dict): Model state dictionary
            eval_mode (bool): Whether to set model to evaluation mode
        """
        self.load_state_dict(state_dict, strict=True)
        if eval_mode:
            self.eval()

    def forward(self, model_input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        # Enables us to compute gradients w.r.t. coordinates
        coords_org = model_input['coords'].clone().detach().requires_grad_(True)
        coords = coords_org
        output = self.net(coords)
        return {'model_in': coords_org, 'model_out': output}

    def print_network_info(self):
        """Print network architecture and parameter information."""
        print(f"\nNetwork Architecture:")
        print("=" * 50)
        print(self)
        print("\nParameter Information:")
        print("=" * 50)
        total_params = 0
        for name, param in self.named_parameters():
            print(f"{name}:")
            print(f"Shape: {param.shape}")
            print(f"Data:\n{param.data}")
            print("-" * 50)
            total_params += param.numel()
        print(f"Total Parameters: {total_params}")

