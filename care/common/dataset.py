import torch
from torch.utils.data import Dataset
import logging
import matplotlib.pyplot as plt
import matplotlib.pyplot as mplot
import random
import os

logger = logging.getLogger(__name__)

class ReachabilityDataset(Dataset):
    """
    Base class for reachability analysis datasets.
    Implements common functionality for curriculum learning and data generation.
    """
    def __init__(self, batch_size, t_min=0.0, t_max=1.0, 
                 seed=0, device=None,
                 counterexamples=None, percentage_in_counterexample=10,
                 percentage_at_t0=10, epsilon_radius=0.1,
                 num_states=None, compute_boundary_values=None):
        """
        Initialize base dataset parameters.

        Args:
            batch_size (int): Number of points to sample per batch
            t_min (float): Minimum time value
            t_max (float): Maximum time value
            seed (int): Random seed for reproducibility
            device (torch.device): Device to store tensors
            counterexamples (torch.Tensor, optional): Counterexample points [n, state_dim]
            percentage_in_counterexample (float): Percentage of points near counterexample
            percentage_at_t0 (float): Percentage of points at t=0
            epsilon_radius (float): Radius around counterexample points to sample
            num_states (int): Number of state dimensions
            compute_boundary_values (callable): Function that computes boundary values
        """
        super().__init__()
        torch.manual_seed(seed)
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        logger.debug(f"Initializing dataset with batch size {batch_size}")

        self.batch_size = batch_size
        self.t_min = t_min
        self.t_max = t_max

        self.state_min = -1.01
        self.state_max = 1.01

        if num_states is None:
            raise ValueError("num_states must be specified")
        self.num_states = num_states

        if compute_boundary_values is None:
            raise ValueError("compute_boundary_values function must be specified")
        self.compute_boundary_values = compute_boundary_values

        # Counterexample parameters
        self.counterexamples = counterexamples if counterexamples is not None else None
        self.percentage_in_counterexample = percentage_in_counterexample
        self.percentage_at_t0 = percentage_at_t0
        self.epsilon_radius = epsilon_radius

        # Pre-allocate tensors for sampling to avoid repeated memory allocations
        # Make sure these don't require gradients
        self._time_tensor = torch.empty(self.batch_size, 1, device=self.device, requires_grad=False)
        self._random_states_tensor = torch.empty(self.batch_size, self.num_states, device=self.device, requires_grad=False)
        self._t0_states_tensor = None  # Will be initialized based on percentage_at_t0
        self._counter_points_tensor = None  # Will be initialized if needed
        self._noise_tensor = None  # Will be initialized if needed
        
        # Pre-compute bound tensors for clamping operations
        self._min_bounds = torch.tensor([self.t_min] + [self.state_min] * self.num_states, device=self.device, requires_grad=False)
        self._max_bounds = torch.tensor([self.t_max] + [self.state_max] * self.num_states, device=self.device, requires_grad=False)
        
        # Compute sizes once based on percentages
        self.n_t0 = int(self.batch_size * self.percentage_at_t0 / 100)
        self.n_counter = int(self.batch_size * self.percentage_in_counterexample / 100) if counterexamples is not None else 0
        self.n_random = self.batch_size - self.n_counter - self.n_t0
        
        # Initialize t0 tensor now that we know its size
        if self.n_t0 > 0:
            self._t0_states_tensor = torch.empty(self.n_t0, self.num_states, device=self.device, requires_grad=False)
            self._t0_time_tensor = torch.full((self.n_t0, 1), self.t_min, device=self.device, requires_grad=False)
        
        # Initialize counter tensors if needed
        if self.n_counter > 0 and counterexamples is not None:
            self._counter_idx_tensor = torch.empty(self.n_counter, dtype=torch.long, device=self.device, requires_grad=False)
            self._counter_points_tensor = torch.empty(self.n_counter, self.num_states + 1, device=self.device, requires_grad=False)
            self._noise_tensor = torch.empty(self.n_counter, self.num_states + 1, device=self.device, requires_grad=False)
        
        # Pre-allocate output tensors
        self._coords = torch.empty(self.batch_size, 1, self.num_states + 1, device=self.device, requires_grad=False)
        self._boundary_values = torch.empty(self.batch_size, 1, 1, device=self.device, requires_grad=False)
        self._dirichlet_mask = torch.empty(self.batch_size, 1, dtype=torch.bool, device=self.device, requires_grad=False)

    def __len__(self):
        """Dataset length is always 1 as we generate data dynamically."""
        return 1

    def update_time_range(self, t_min: float, t_max: float) -> None:
        """Update the time range for sampling."""
        self.t_min = t_min
        self.t_max = t_max

    def _get_time_samples(self):
        """Generate time samples using pre-allocated tensor."""
        # Fill pre-allocated tensor with uniform values
        self._time_tensor.uniform_(self.t_min, self.t_max)
        
        # Set t=0 for designated samples
        if self.n_t0 > 0:
            self._time_tensor[-self.n_t0:, 0] = self.t_min
        
        return self._time_tensor, self.t_min

    def _sample_near_counterexample(self):
        """Optimized sampling near counterexamples with pre-allocated tensors."""
        if self.counterexamples is None or self.counterexamples.shape[0] == 0 or self.n_counter == 0:
            return torch.empty(0, self.num_states + 1, device=self.device)
            
        # Sample indices into pre-allocated tensor
        self._counter_idx_tensor.random_(0, self.counterexamples.shape[0])
        
        # Copy counterexample points to pre-allocated tensor
        self._counter_points_tensor.copy_(self.counterexamples[self._counter_idx_tensor])
        
        # Generate noise into pre-allocated tensor
        self._noise_tensor.randn_().mul_(self.epsilon_radius)
        
        # Add noise and clamp in-place
        self._counter_points_tensor.add_(self._noise_tensor).clamp_(
            min=self._min_bounds,
            max=self._max_bounds
        )
        
        return self._counter_points_tensor

    def _sample_state_space(self, tensor):
        """
        Fill pre-allocated tensor with uniform samples.
        
        Args:
            tensor (torch.Tensor): Pre-allocated tensor to fill
        """
        tensor.uniform_(self.state_min, self.state_max)
        return tensor

    def __getitem__(self, idx):
        """Get samples using current time range and pre-allocated tensors."""
        # We'll completely detach from computation graph at the end, no need for no_grad during generation
        time, start_time = self._get_time_samples()
        
        # Sample remaining points into pre-allocated tensors
        self._sample_state_space(self._random_states_tensor[:self.n_random])
        
        if self.n_t0 > 0:
            self._sample_state_space(self._t0_states_tensor)
        
        # Get counterexample points if needed
        if self.n_counter > 0 and self.counterexamples is not None:
            counter_coords = self._sample_near_counterexample()
        
        # Fill the coords tensor directly (avoiding concatenation)
        offset = 0
        
        # Fill random coords section
        self._coords[:self.n_random, 0, 0] = time[:self.n_random, 0]
        self._coords[:self.n_random, 0, 1:] = self._random_states_tensor[:self.n_random]
        offset += self.n_random
        
        # Fill counterexample section if needed
        if self.n_counter > 0 and self.counterexamples is not None:
            self._coords[offset:offset+self.n_counter, 0, :] = counter_coords
            offset += self.n_counter
        
        # Fill t0 section if needed
        if self.n_t0 > 0:
            self._coords[offset:, 0, 0] = self.t_min
            self._coords[offset:, 0, 1:] = self._t0_states_tensor
        
        # Compute boundary values without squeezing/unsqueezing
        boundary_values = self.compute_boundary_values(self._coords[..., 1:].squeeze(1))
        self._boundary_values[:, 0, 0] = boundary_values.squeeze(1)
        
        # Create Dirichlet mask directly
        self._dirichlet_mask[:, 0] = (self._coords[:, 0, 0] == start_time)

        # Create output dictionary with tensor views, but detach to prevent gradient tracking
        # This is much more efficient than cloning
        return {
            'coords': self._coords.detach()
        }, {
            'source_boundary_values': self._boundary_values.detach(),
            'dirichlet_mask': self._dirichlet_mask.detach()
        }

    def add_counterexample(self, counterexample: torch.Tensor):
        """Add new counterexample points to the existing dataset.
        
        Args:
            counterexample (torch.Tensor): New counterexample points [n, state_dim]
        """
        if not isinstance(counterexample, torch.Tensor):
            raise TypeError("counterexample must be a torch.Tensor")
        
        # Ensure counterexample has correct shape
        if counterexample.dim() == 1:
            counterexample = counterexample.unsqueeze(0)
        
        if counterexample.size(1) == self.num_states:  # If only state variables provided
            # Add time dimension initialized to t_max
            time_dim = torch.full((counterexample.size(0), 1), self.t_max, device=self.device)
            counterexample = torch.cat([time_dim, counterexample], dim=1)
        
        # Add to existing counterexample if it exists, otherwise create new
        if self.counterexamples is None:
            self.counterexamples = counterexample.to(self.device)
        else:
            self.counterexamples = torch.cat([self.counterexamples, counterexample.to(self.device)], dim=0)
        
        logger.debug(f"Added {counterexample.size(0)} counterexample points. Total: {self.counterexamples.size(0)}")

    def get_batch(self):
        """Generate a new batch of data directly without using indexing."""
        return self.__getitem__(0)
    
    def _plot_samples(self, coords, percentage=1):
        """
        Visualize the distribution of sampled points in state space and time.
        
        Args:
            coords (torch.Tensor): Sampled coordinates of shape (batch_size, 1, num_states + 1)
        """
        # Extract t, x, and y coordinates from the sampled points
        t = coords[:, 0, 0].cpu().numpy()
        x = coords[:, 0, 1].cpu().numpy()
        y = coords[:, 0, 2].cpu().numpy()
        
        # Subsample 1% of the points for plotting
        indices = random.sample(range(len(t)), int(len(t) * percentage/100))
        t_sampled = t[indices]
        x_sampled = x[indices]
        y_sampled = y[indices]
        
        # Create a 3D scatter plot
        fig = mplot.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_sampled, y_sampled, t_sampled, alpha=0.5)
        
        ax.set_xlabel("State Variable 1")
        ax.set_ylabel("State Variable 2")
        ax.set_zlabel("Time")
        ax.set_title("Distribution of Sampled Points in State-Time Space")
        
        ax.set_xlim(self.state_min, self.state_max)
        ax.set_ylim(self.state_min, self.state_max)
        ax.set_zlim(self.t_min, self.t_max)
        
        # Save the figure to a file
        if not os.path.exists('plots'):
            os.makedirs('plots')
        
        filename = f"plots/sample_distribution_{random.randint(0, 1000)}.png"
        mplot.savefig(filename)
        plt.close(fig)  # Close the figure to release memory
        logger.debug(f"Saved plot to {filename}")

