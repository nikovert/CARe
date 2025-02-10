import numpy as np
from scipy.io import loadmat
import torch
from typing import Dict, Tuple, Any
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def construct_grid(grid_min: np.ndarray, grid_max: np.ndarray, N: np.ndarray) -> np.ndarray:
    """
    Construct grid points from MATLAB grid parameters.
    
    Args:
        grid_min: Lower corner of computation domain
        grid_max: Upper corner of computation domain
        N: Number of grid points per dimension
        
    Returns:
        Array of grid points
    """
    dimensions = len(N)
    grid_vectors = [np.linspace(grid_min[i], grid_max[i], N[i]) for i in range(dimensions)]
    mesh_grid = np.meshgrid(*grid_vectors, indexing='ij')
    return np.column_stack([x.flatten() for x in mesh_grid])

def load_matlab_data(file_path: str) -> Dict[str, np.ndarray]:
    """
    Load value function and grid data from a MATLAB .mat file.
    
    Args:
        file_path: Path to the .mat file
        
    Returns:
        Dictionary containing the value function and grid data
    """
    try:
        mat_data = loadmat(file_path)
        
        # Extract grid parameters
        grid_min = mat_data['grid_min'].flatten()
        grid_max = mat_data['grid_max'].flatten()
        N = mat_data['N'].flatten()
        
        # Construct grid points
        grid_points = construct_grid(grid_min, grid_max, N)
        
        # Get value function data (last time step)
        value_function = mat_data['data']
        if len(value_function.shape) > 2:
            value_function = value_function[:, :, -1]  # Take last time step
        value_function = value_function.flatten()
        
        return {
            'grid': grid_points,
            'value': value_function,
            'time': mat_data['tau'].flatten(),
            'grid_min': grid_min,
            'grid_max': grid_max,
            'N': N
        }
    except Exception as e:
        raise ValueError(f"Error loading MATLAB file: {e}")

def plot_comparison(grid_points: np.ndarray, true_values: np.ndarray, 
                   nn_predictions: np.ndarray, difference: np.ndarray) -> None:
    """
    Plot the comparison between true values and neural network predictions.
    
    Args:
        grid_points: Input grid points
        true_values: Ground truth values from MATLAB
        nn_predictions: Predictions from neural network
        difference: Difference between predictions and true values
    """
    dim = grid_points.shape[1]
    if dim > 2:
        raise ValueError("Plotting only supported for 1D or 2D data")
        
    fig = plt.figure(figsize=(15, 5))
    
    if dim == 1:
        # 1D case: Line plots
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        
        ax1.plot(grid_points, true_values, 'b-', label='True Values')
        ax1.set_title('True Values')
        
        ax2.plot(grid_points, nn_predictions, 'r-', label='NN Predictions')
        ax2.set_title('NN Predictions')
        
        ax3.plot(grid_points, difference, 'g-', label='Difference')
        ax3.set_title('Difference')
        
    else:  # dim == 2
        # 2D case: Surface plots
        ax1 = fig.add_subplot(131, projection='3d')
        ax2 = fig.add_subplot(132, projection='3d')
        ax3 = fig.add_subplot(133, projection='3d')
        
        x = grid_points[:, 0].reshape(-1)
        y = grid_points[:, 1].reshape(-1)
        
        # Determine grid size for reshaping
        x_unique = np.unique(x)
        y_unique = np.unique(y)
        X, Y = np.meshgrid(x_unique, y_unique)
        
        Z_true = true_values.reshape(X.shape)
        Z_pred = nn_predictions.reshape(X.shape)
        Z_diff = difference.reshape(X.shape)
        
        ax1.plot_surface(X, Y, Z_true, cmap='viridis')
        ax1.set_title('True Values')
        
        ax2.plot_surface(X, Y, Z_pred, cmap='viridis')
        ax2.set_title('NN Predictions')
        
        ax3.plot_surface(X, Y, Z_diff, cmap='viridis')
        ax3.set_title('Difference')
    
    plt.tight_layout()
    plt.show()

def compare_with_nn(
    nn_model: torch.nn.Module,
    matlab_data: Dict[str, np.ndarray],
    value_key: str = 'value',
    grid_key: str = 'grid',
    visualize: bool = False
) -> Tuple[np.ndarray, float]:
    """
    Compare neural network predictions with MATLAB ground truth data.
    
    Args:
        nn_model: Trained neural network model
        matlab_data: Dictionary containing MATLAB data
        value_key: Key for value function in matlab_data
        grid_key: Key for grid points in matlab_data
        visualize: If True, plot the comparison
        
    Returns:
        Tuple of (difference array, mean squared error)
    """
    if value_key not in matlab_data or grid_key not in matlab_data:
        raise ValueError(f"Required keys {value_key} or {grid_key} not found in MATLAB data")
    
    # Get the device from the model
    device = next(nn_model.parameters()).device
    
    # Move input to the same device as the model
    grid_points = torch.tensor(matlab_data[grid_key], dtype=torch.float32, device=device)
    true_values = matlab_data[value_key]
    
    # Format input as dictionary for neural network
    model_input = {'coords': grid_points}
    
    with torch.no_grad():
        output_dict = nn_model(model_input)
        # Extract value from output dictionary, convert to numpy and reshape
        nn_predictions = output_dict['model_out'].cpu().numpy().reshape(-1)
    
    difference = nn_predictions - true_values
    mse = np.mean(difference ** 2)
    
    if visualize:
        plot_comparison(matlab_data[grid_key], true_values, nn_predictions, difference)
    
    return difference, mse
