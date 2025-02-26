import numpy as np
from scipy.io import loadmat
import torch
from typing import Dict, Tuple
import matplotlib
matplotlib.use('Agg')  # Add this before other matplotlib imports
import matplotlib.pyplot as plt

def construct_grid(grid_min: np.ndarray, grid_max: np.ndarray, N: np.ndarray, time_points: np.ndarray = None) -> np.ndarray:
    """
    Construct grid points from MATLAB grid parameters.
    
    Args:
        grid_min: Lower corner of computation domain
        grid_max: Upper corner of computation domain
        N: Number of grid points per dimension
        time_points: Time points to include in grid
        
    Returns:
        Array of grid points with shape (num_total_points, num_dimensions)
    """
    # Create spatial grid vectors
    spatial_vectors = [np.linspace(grid_min[i], grid_max[i], N[i]) for i in range(len(N))]
    
    if time_points is not None:
        # Create meshgrid including time dimension
        grid_vectors = [time_points] + spatial_vectors
    else:
        grid_vectors = spatial_vectors
    
    # Create meshgrid with 'ij' indexing
    mesh_grid = np.meshgrid(*grid_vectors, indexing='ij')
    
    # Reshape to (N, d) where N is total number of points and d is dimension
    return np.column_stack([x.flatten() for x in mesh_grid])

def load_matlab_data(file_path: str) -> Dict[str, np.ndarray]:
    """
    Load value function and grid data from a MATLAB .mat file.
    
    Args:
        file_path: Path to the .mat file
        
    Returns:
        Dictionary containing the value function and grid data with time as first dimension
    """
    try:
        mat_data = loadmat(file_path)
        
        # Extract grid parameters
        grid_min = mat_data['grid_min'].flatten()
        grid_max = mat_data['grid_max'].flatten()
        N = mat_data['N'].flatten()
        time_points = mat_data['tau'].flatten()
        
        # Construct full grid including time dimension
        grid_points = construct_grid(grid_min, grid_max, N, time_points)
        
        # Keep value function data and transpose from (x1, x2, t) to (t, x1, x2)
        value_function = np.moveaxis(mat_data['data'], -1, 0)
        
        return {
            'grid': grid_points,
            'value': value_function,
            'time': time_points,
            'grid_min': grid_min,
            'grid_max': grid_max,
            'N': N,
            'shape': value_function.shape
        }
    except Exception as e:
        raise ValueError(f"Error loading MATLAB file: {e}")

def plot_comparison(grid_points: np.ndarray, true_values: np.ndarray, 
                   nn_predictions: np.ndarray, difference: np.ndarray,
                   save_path: str = 'comparison.png') -> None:
    """
    Plot the comparison between true values and neural network predictions at three time slices.
    Creates smooth surface plots by reconstructing the grid at each time slice.
    
    Args:
        grid_points: Input grid points (time as first dimension)
        true_values: Ground truth values from MATLAB (time as first dimension)
        nn_predictions: Predictions from neural network (time as first dimension)
        difference: Difference between predictions and true values (time as first dimension)
        save_path: Path to save the comparison plot
    """
    # Get time points from first dimension of grid points
    time_points = np.unique(grid_points[:, 0])
    t_start = time_points[0]
    t_mid = time_points[len(time_points)//2]
    t_end = time_points[-1]
    time_slices = [t_start, t_mid, t_end]
    
    # Create figure with three rows (one for each time) and three columns (true, pred, diff)
    fig = plt.figure(figsize=(15, 15))
    
    # Get unique x and y coordinates for grid reconstruction
    x_unique = np.unique(grid_points[:, 1])
    y_unique = np.unique(grid_points[:, 2])
    X, Y = np.meshgrid(x_unique, y_unique, indexing='ij')
    
    for i, t in enumerate(time_slices):
        # Get index for this time slice
        t_idx = np.where(time_points == t)[0][0]
        
        # Extract 2D slices at this time point - already in correct grid shape
        true_slice = true_values[t_idx]
        pred_slice = nn_predictions[t_idx]
        diff_slice = difference[t_idx]
        
        # Create subplots
        ax1 = fig.add_subplot(3, 3, i*3 + 1, projection='3d')
        ax2 = fig.add_subplot(3, 3, i*3 + 2, projection='3d')
        ax3 = fig.add_subplot(3, 3, i*3 + 3, projection='3d')
        
        # Plot smooth surfaces
        surf1 = ax1.plot_surface(X, Y, true_slice, cmap='viridis')
        ax1.set_title(f't = {t:.2f}: True Values')
        ax1.set_xlabel('x1')
        ax1.set_ylabel('x2')
        fig.colorbar(surf1, ax=ax1)
        
        surf2 = ax2.plot_surface(X, Y, pred_slice, cmap='viridis')
        ax2.set_title(f't = {t:.2f}: NN Predictions')
        ax2.set_xlabel('x1')
        ax2.set_ylabel('x2')
        fig.colorbar(surf2, ax=ax2)
        
        surf3 = ax3.plot_surface(X, Y, diff_slice, cmap='viridis')
        ax3.set_title(f't = {t:.2f}: Difference')
        ax3.set_xlabel('x1')
        ax3.set_ylabel('x2')
        fig.colorbar(surf3, ax=ax3)
        
        # Set consistent view angle for all plots in this row
        for ax in [ax1, ax2, ax3]:
            ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def compare_with_nn(
    nn_model: torch.nn.Module,
    matlab_data: Dict[str, np.ndarray],
    value_key: str = 'value',
    grid_key: str = 'grid',
    visualize: bool = False,
    save_path: str = 'comparison.png'
) -> Tuple[np.ndarray, float]:
    """
    Compare neural network predictions with MATLAB ground truth data.
    
    Args:
        nn_model: Trained neural network model
        matlab_data: Dictionary containing MATLAB data
        value_key: Key for value function in matlab_data
        grid_key: Key for grid points in matlab_data
        visualize: If True, plot the comparison (only works for 2D spatial systems)
        save_path: Path to save the comparison plot
        
    Returns:
        Tuple of (difference array, mean squared error)
    """
    if value_key not in matlab_data or grid_key not in matlab_data:
        raise ValueError(f"Required keys {value_key} or {grid_key} not found in MATLAB data")
    
    # Get the device from the model
    device = next(nn_model.parameters()).device
    
    # Move input to the same device as the model and ensure correct formatting
    grid_points = torch.tensor(matlab_data[grid_key], dtype=torch.float32, device=device)
    
    # Ensure grid_points has at least 2 dimensions (time + at least 1 spatial dimension)
    if grid_points.dim() != 2 or grid_points.shape[1] < 2:
        raise ValueError(f"Expected grid points with shape [N, d+1] where d>=1 is spatial dims, got {grid_points.shape}")
    
    # Format input as dictionary with 'coords' key as expected by SingleBVPNet
    model_input = {'coords': grid_points}
    true_values = matlab_data[value_key]
    
    # Run model in evaluation mode
    nn_model.eval()
    with torch.no_grad():
        output_dict = nn_model(model_input)
        # Extract 'model_out' from the output dictionary and reshape to match MATLAB data
        nn_predictions = output_dict['model_out'].cpu().numpy()
        
        # Reshape to match original MATLAB data shape [time, x1, x2, ..., xn]
        nn_predictions = nn_predictions.reshape(matlab_data['shape'])
    
    difference = nn_predictions - true_values
    mse = np.mean(difference ** 2)
    
    if visualize:
        if grid_points.shape[1] != 3:  # time + 2 spatial dimensions
            print("Warning: Visualization only supported for 2D spatial systems. Skipping plot.")
        else:
            plot_comparison(matlab_data[grid_key], true_values, nn_predictions, difference, save_path)
    
    return difference, mse
