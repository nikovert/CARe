import torch
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)

class ReachabilityDataset(Dataset):
    """
    Base class for reachability analysis datasets.
    Implements common functionality for curriculum learning and data generation.
    """
    def __init__(self, batch_size, tMin=0.0, tMax=1.0, 
                 seed=0, device=None,
                 counterexamples=None, percentage_in_counterexample=10,
                 percentage_at_t0=20, epsilon_radius=0.1,
                 num_states=None, compute_boundary_values=None,
                 fixed_grid: bool = True):
        """
        Initialize base dataset parameters.

        Args:
            batch_size (int): Number of points to sample per batch
            tMin (float): Minimum time value
            tMax (float): Maximum time value
            seed (int): Random seed for reproducibility
            device (torch.device): Device to store tensors
            counterexample (torch.Tensor, optional): Counterexample points [n, state_dim]
            percentage_in_counterexample (float): Percentage of points near counterexample
            percentage_at_t0 (float): Percentage of points at t=0
            epsilon_radius (float): Radius around counterexample points to sample
            num_states (int): Number of state dimensions
            compute_boundary_values (callable): Function that computes boundary values
                                            signature: f(coords: torch.Tensor) -> torch.Tensor
            fixed_grid (bool): Whether to use a fixed grid of samples
        """
        super().__init__()
        torch.manual_seed(seed)
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        logger.debug(f"Initializing dataset with batch size {batch_size}")

        self.batch_size = batch_size
        self.tMin = tMin
        self.tMax = tMax

        if num_states is None:
            raise ValueError("num_states must be specified")
        self.num_states = num_states

        if compute_boundary_values is None:
            raise ValueError("compute_boundary_values function must be specified")
        self.compute_boundary_values = compute_boundary_values

        # Counterexample parameters
        self.counterexamples = counterexamples.to(self.device) if counterexamples is not None else None
        self.percentage_in_counterexample = percentage_in_counterexample
        self.percentage_at_t0 = percentage_at_t0
        self.epsilon_radius = epsilon_radius

        self.fixed_grid = fixed_grid
        # Precompute time samples cache:
        self._precomputed_time = None
        self._start_time = None
        if self.fixed_grid:
            self.fixed_states = self._generate_fixed_grid_states()
            self.fixed_state_sample = self._compute_fixed_state_sample()
            # Precompute boundary values for fixed state sample for speed
            self.fixed_boundary_values = self.compute_boundary_values(self.fixed_state_sample).unsqueeze(1)

    def _generate_fixed_grid_states(self):
        """Generate a fixed grid of state space samples only."""
        m = max(1, round(self.batch_size ** (1 / self.num_states)))
        lins = [torch.linspace(-1, 1, steps=m, device=self.device) for _ in range(self.num_states)]
        grid = torch.cartesian_prod(*lins)  # Shape: [m^(num_states), num_states]
        return grid

    def _compute_fixed_state_sample(self):
        """Compute and cache fixed state samples from the grid for fast __getitem__.
           Precompute a random permutation of the fixed grid to cover the state space."""
        total_points = self.fixed_states.size(0)
        perm = torch.randperm(total_points)
        if total_points >= self.batch_size:
            state_sample = self.fixed_states[perm]
        else:
            state_sample = self.fixed_states[perm]
            extra = self.batch_size - total_points
            extra_states = self._sample_state_space(extra)
            state_sample = torch.cat([state_sample, extra_states], dim=0)
        return state_sample

    def __len__(self):
        """Dataset length is always 1 as we generate data dynamically."""
        return 1

    def update_time_range(self, tMin: float, tMax: float) -> None:
        """Update the time range for sampling and precompute time samples."""
        self.tMin = tMin
        self.tMax = tMax
        # Precompute time samples with the new time range.
        self._precomputed_time, self._start_time = self._get_time_samples()
        # No grid regeneration needed for fixed state space

    def _get_time_samples(self):
        """Generate time samples using current time range."""
        # Use torch.empty instead of torch.zeros to avoid unneeded zero initialization.
        time = torch.empty(self.batch_size, 1, device=self.device).uniform_(self.tMin, self.tMax)
        
        n_t0 = int(self.batch_size * self.percentage_at_t0 / 100)
        time[-n_t0:, 0] = self.tMin
        
        return time, self.tMin

    def _sample_near_counterexample(self, num_points):
        """
        Sample points near the counterexample points including time dimension.
        
        Args:
            num_points (int): Number of points to sample
            
        Returns:
            torch.Tensor: Sampled points of shape (num_points, num_states + 1)
        """
        if self.counterexamples is None:
            return None
            
        counter_idx = torch.randint(0, self.counterexamples.shape[0], (num_points,))
        # Create a new tensor that requires gradients
        counter_points = self.counterexamples[counter_idx].detach().clone()
        counter_points.requires_grad_(True)
        noise = torch.randn_like(counter_points) * self.epsilon_radius
        noise.requires_grad_(True)
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
        
        # Use torch.empty instead of torch.zeros before filling uniformly.
        return torch.empty(num_points, self.num_states, device=self.device).uniform_(-1, 1)

    def __getitem__(self, idx):
        # Use precomputed time samples if available.
        if self._precomputed_time is None or self._start_time is None:
            time, start_time = self._get_time_samples()
        else:
            time, start_time = self._precomputed_time, self._start_time
        n_t0 = int(self.batch_size * self.percentage_at_t0 / 100)
        
        if self.fixed_grid:
            # Fixed grid branch
            if self.counterexamples is not None:
                n_counter = int(self.batch_size * self.percentage_in_counterexample / 100)
                n_random = self.batch_size - n_counter - n_t0
                counter_coords = self._sample_near_counterexample(n_counter)
                counter_bvals = self.compute_boundary_values(counter_coords[...,1:]).unsqueeze(1)
                fixed_random_states = self.fixed_state_sample[:n_random]
                fixed_random_bvals = self.fixed_boundary_values[:n_random]
            else:
                n_counter = 0
                n_random = self.batch_size - n_t0
                fixed_random_states = self.fixed_state_sample[:n_random]
                fixed_random_bvals = self.fixed_boundary_values[:n_random]
            
            random_coords = torch.cat([time[:n_random], fixed_random_states], dim=1) if n_random > 0 else None
            t0_states = self._sample_state_space(n_t0)
            t0_coords = torch.cat([torch.zeros(n_t0, 1, device=self.device), t0_states], dim=1)
            t0_bvals = self.compute_boundary_values(t0_states).unsqueeze(1)
            
            if n_counter > 0:
                coords = torch.cat([counter_coords, random_coords, t0_coords], dim=0)
                bvals = torch.cat([counter_bvals, fixed_random_bvals, t0_bvals], dim=0)
            else:
                coords = torch.cat([random_coords, t0_coords], dim=0)
                bvals = torch.cat([fixed_random_bvals, t0_bvals], dim=0)
        else:
            # Non-fixed grid branch
            if self.counterexamples is not None:
                n_counter = int(self.batch_size * self.percentage_in_counterexample / 100)
                n_random = self.batch_size - n_counter - n_t0
                counter_coords = self._sample_near_counterexample(n_counter)
            else:
                n_counter = 0
                n_random = self.batch_size - n_t0
            
            random_states = self._sample_state_space(n_random)
            t0_states = self._sample_state_space(n_t0)
            random_coords = torch.cat([time[:n_random], random_states], dim=1)
            t0_coords = torch.cat([torch.zeros(n_t0, 1, device=self.device), t0_states], dim=1)
            
            if self.counterexamples is not None:
                coords = torch.cat([counter_coords, random_coords, t0_coords], dim=0)
            else:
                coords = torch.cat([random_coords, t0_coords], dim=0)
            bvals = self.compute_boundary_values(coords[...,1:]).unsqueeze(1)
        
        # Common postprocessing: add observation dim and create Dirichlet mask
        coords = coords.unsqueeze(1)
        dirichlet_mask = (coords[:, :, 0] == start_time)
        return {'coords': coords}, {'source_boundary_values': bvals, 'dirichlet_mask': dirichlet_mask}

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
            # Add time dimension initialized to tMax
            time_dim = torch.full((counterexample.size(0), 1), self.tMax, device=self.device)
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

