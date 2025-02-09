import torch
from collections import OrderedDict
import math
from pathlib import Path

def set_known_weights_and_print(model):
    """
    Set known weights and biases for the model's layers, including BatchLinear and PolynomialLayer.
    
    Assign unique, deterministic values for easy verification.
    
    Args:
        model (torch.nn.Module): The PyTorch model to adjust.

    Returns:
        List[Tuple[str, torch.Tensor]]: Stored weights and biases for verification.
    """
    layer_idx = 0
    weights_and_biases = []  # To store weights and biases for saving
    device = next(model.parameters()).device  # Get model's device

    # Iterate through all named modules in the network
    for name, module in model.net.named_modules():
        
        # Handle BatchLinear Layers
        if isinstance(module, torch.nn.Linear) or "BatchLinear" in type(module).__name__:
            # Assign unique weights and biases
            weight = torch.tensor(
                [[0.1 * (layer_idx + i + j + 1) for j in range(module.weight.shape[1])]
                 for i in range(module.weight.shape[0])],
                dtype=torch.float32,
                device=device  # Use model's device
            )
            bias = torch.tensor(
                [0.05 * (layer_idx + i + 1) for i in range(module.bias.shape[0])],
                dtype=torch.float32,
                device=device  # Use model's device
            )

            # Set weights and biases in the layer
            module.weight.data = weight
            module.bias.data = bias

            # Save for reference
            weights_and_biases.append((f"Layer {layer_idx + 1} Weights", weight.clone()))
            weights_and_biases.append((f"Layer {layer_idx + 1} Biases", bias.clone()))

            layer_idx += 1
        
        # Handle Polynomial Layers
        elif "PolynomialLayer" in type(module).__name__:
            # Save Polynomial Layer Info
            poly_info = f"Polynomial Layer {layer_idx + 1}: Degree={module.degree}, In={module.in_features}, Out={module.out_features}"
            weights_and_biases.append((poly_info, "No Trainable Parameters"))
            layer_idx += 1
    
    # Print assigned weights and biases
    for name, tensor in weights_and_biases:
        print(f"{name}:")
        print(tensor)
        print("-" * 50)

    return weights_and_biases

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
    def __init__(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)

class PolynomialLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, degree=2):
        """
        A custom layer that computes polynomial features based on input features.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features (ignored for now).
            degree (int): Highest degree of polynomial terms.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree

    def forward(self, x):
        """
        Computes concatenated polynomial features.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Concatenated tensor with shape (batch_size, in_features * degree).
        """
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Expected input with {self.in_features} features, got {x.shape[-1]}")

        # Compute polynomial terms
        poly_features = [x**i for i in range(1, self.degree + 1)]
        return torch.cat(poly_features, dim=-1)

class FCBlock(torch.nn.Module):
    '''A fully connected neural network.
    '''

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='sine', weight_init=None,
                 use_polynomial=False, poly_degree=2):
        super().__init__()

        self.first_layer_init = None

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {'sine':(Sine(), sine_init, first_layer_sine_init),
                         'relu':(torch.nn.ReLU(inplace=True), init_weights_normal, None),
                         'sigmoid':(torch.nn.Sigmoid(), init_weights_xavier, None),
                         'tanh':(torch.nn.Tanh(), init_weights_xavier, None),
                         'selu':(torch.nn.SELU(inplace=True), init_weights_selu, None),
                         'softplus':(torch.nn.Softplus(), init_weights_normal, None),
                         'elu':(torch.nn.ELU(inplace=True), init_weights_elu, None)}

        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        self.net = []

        # Optionally add PolynomialLayer
        if use_polynomial:
            self.net.append(PolynomialLayer(
                in_features=in_features, 
                out_features=in_features * poly_degree, 
                degree=poly_degree
            ))
            in_features = in_features * poly_degree

        self.net.append(torch.nn.Sequential(
            BatchLinear(in_features, hidden_features), nl
        ))

        for i in range(num_hidden_layers):
            self.net.append(torch.nn.Sequential(
                BatchLinear(hidden_features, hidden_features), nl
            ))

        if outermost_linear:
            self.net.append(torch.nn.Sequential(BatchLinear(hidden_features, out_features)))
        else:
            self.net.append(torch.nn.Sequential(
                BatchLinear(hidden_features, out_features), nl
            ))

        self.net = torch.nn.Sequential(*self.net)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        if first_layer_init is not None: # Apply special initialization to first layer, if applicable.
            self.net[0].apply(first_layer_init)

    def forward(self, coords, params=None, **kwargs):
        if params is None:
            params = OrderedDict(self.named_parameters())

        output = self.net(coords)
        return output

class SingleBVPNet(torch.nn.Module):
    '''A canonical representation network for a BVP.'''

    def __init__(self, 
out_features: int = 1,
type: str = 'sine',
in_features=2,
                 mode='mlp', 
hidden_features=32, 
num_hidden_layers=3,
                 use_polynomial=False, 
poly_degree=2, 
**kwargs) -> None:
        super().__init__()
        self.mode = mode
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.num_hidden_layers = num_hidden_layers
        self.activation_type = type
        self.use_polynomial = use_polynomial
        self.poly_degree = poly_degree
        
        self.net = FCBlock(in_features=in_features, out_features=out_features, 
                          num_hidden_layers=num_hidden_layers,
                          hidden_features=hidden_features, outermost_linear=True, 
                          nonlinearity=type, use_polynomial=use_polynomial, 
                          poly_degree=poly_degree)
        print(self)
        self._checkpoint_dir = None

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

    def get_config(self):
        """Return model configuration for reconstruction"""
        return {
            'mode': self.mode,
            'in_features': self.in_features,
            'out_features': self.out_features,
            'hidden_features': self.hidden_features,
            'num_hidden_layers': self.num_hidden_layers,
            'type': self.activation_type,
            'use_polynomial': self.use_polynomial,
            'poly_degree': self.poly_degree
        }

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
                'type': 'sine',
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

########################
# Initialization methods
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # For PINNet, Raissi et al. 2019
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

def init_weights_trunc_normal(m):
    # For PINNet, Raissi et al. 2019
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    if type(m) == BatchLinear or type(m) == torch.nn.Linear:
        if hasattr(m, 'weight'):
            fan_in = m.weight.size(1)
            fan_out = m.weight.size(0)
            std = math.sqrt(2.0 / float(fan_in + fan_out))
            mean = 0.
            # initialize with the same behavior as tf.truncated_normal
            # "The generated values follow a normal distribution with specified mean and
            # standard deviation, except that values whose magnitude is more than 2
            # standard deviations from the mean are dropped and re-picked."
            _no_grad_trunc_normal_(m.weight, mean, std, -2 * std, 2 * std)

def init_weights_normal(m):
    if type(m) == BatchLinear or type(m) == torch.nn.Linear:
        if hasattr(m, 'weight'):
            torch.nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')

def init_weights_selu(m):
    if type(m) == BatchLinear or type(m) == torch.nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            torch.nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))

def init_weights_elu(m):
    if type(m) == BatchLinear or type(m) == torch.nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            torch.nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input))

def init_weights_xavier(m):
    if type(m) == BatchLinear or type(m) == torch.nn.Linear:
        if hasattr(m, 'weight'):
            torch.nn.init.xavier_normal_(m.weight)

def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            bound = torch.sqrt(torch.tensor(6.0 / num_input)) / 30
            m.weight.uniform_(-bound, bound)

def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)

