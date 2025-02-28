import torch
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)

class ReachabilityDataset(Dataset):
    """
    Base class for reachability analysis datasets.
    Implements common functionality for curriculum learning and data generation.
    """
    def __init__(self, batch_size, t_min=0.0, t_max=1.0, 
                 seed=0, device=None,
                 counterexamples=None, percentage_in_counterexample=20,
                 percentage_at_t0=20, epsilon_radius=0.1,
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

    def __len__(self):
        """Dataset length is always 1 as we generate data dynamically."""
        return 1

    def update_time_range(self, t_min: float, t_max: float) -> None:
        """Update the time range for sampling."""
        self.t_min = t_min
        self.t_max = t_max

    def _get_time_samples(self):
        """Generate time samples using current time range."""
        # Regular sampling between t_min and t_max
        time = torch.zeros(self.batch_size, 1, device=self.device).uniform_(self.t_min, self.t_max)
        
        # Ensure some samples at t=0 based on percentage_at_t0
        n_t0 = int(self.batch_size * self.percentage_at_t0 / 100)
        time[-n_t0:, 0] = self.t_min
        
        return time, self.t_min

    def _sample_near_counterexample(self, num_points):
        """
        Sample points near the counterexample points including time dimension.
        
        Args:
            num_points (int): Number of points to sample
            
        Returns:
            torch.Tensor: Sampled points of shape (num_points, num_states + 1)
        """
        if self.counterexamples is None:
            return torch.empty(0.0, self.num_states, device=self.device)
            
        counter_idx = torch.randint(0, self.counterexamples.shape[0], (num_points,))
        # Create a new tensor that requires gradients
        counter_points = self.counterexamples[counter_idx].detach().clone()
        noise = torch.randn_like(counter_points) * self.epsilon_radius
        return counter_points + noise

    def _sample_state_space(self, num_points):
        """
        Sample points in state space uniformly.
        
        Args:
            num_points (int): Number of points to sample
        
        Returns:
            torch.Tensor: Sampled points of shape (num_points, num_states)
        """
        if self.num_states is None:
            raise ValueError("Child class must set self.num_states in __init__")
        
        return torch.zeros(num_points, self.num_states, device=self.device).uniform_(-1, 1)

    def __getitem__(self, idx):
        """Get samples using current time range."""
        time, start_time = self._get_time_samples()
        
        # Calculate number of points for each category
        n_t0 = int(self.batch_size * self.percentage_at_t0 / 100)
        
        if self.counterexamples is not None:
            n_counter = int(self.batch_size * self.percentage_in_counterexample / 100)
            n_random = self.batch_size - n_counter - n_t0
            
            # Sample counterexample points (includes time)
            counter_coords = self._sample_near_counterexample(n_counter)
            
        else:
            n_random = self.batch_size - n_t0
            
        # Sample remaining points
        random_states = self._sample_state_space(n_random)
        t0_states = self._sample_state_space(n_t0)
        
        # Combine with respective time coordinates
        random_coords = torch.cat([time[:n_random], random_states], dim=1)
        t0_coords = torch.cat([torch.zeros(n_t0, 1, device=self.device), t0_states], dim=1)
        
        # Combine all coordinates
        if self.counterexamples is not None:
            coords = torch.cat([counter_coords, random_coords, t0_coords], dim=0)
        else:
            coords = torch.cat([random_coords, t0_coords], dim=0)
            
        # Add observation dimension
        coords = coords.unsqueeze(1)  # [batch_size, 1, num_dims]

        # Compute boundary values and add observation dimension, only pass states, not time
        boundary_values = self.compute_boundary_values(coords[...,1:].squeeze(1)).unsqueeze(1)
        
        # Create Dirichlet mask with observation dimension
        dirichlet_mask = (coords[:, :, 0] == start_time)

        return {'coords': coords}, {
            'source_boundary_values': boundary_values,
            'dirichlet_mask': dirichlet_mask.bool()
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

