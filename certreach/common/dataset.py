import torch
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)

class ReachabilityDataset(Dataset):
    """
    Base class for reachability analysis datasets.
    Implements common functionality for curriculum learning and data generation.
    """
    def __init__(self, numpoints, tMin=0.0, tMax=1.0, 
                 pretrain=False, pretrain_iters=2000,
                 counter_start=0, counter_end=100e3,
                 num_src_samples=1000, seed=0, device=None,
                 counterexample=None, percentage_in_counterexample=20,
                 percentage_at_t0=20, epsilon_radius=0.1,
                 num_states=None, compute_boundary_values=None):
        """
        Initialize base dataset parameters.

        Args:
            numpoints (int): Number of points to sample per batch
            tMin (float): Minimum time value
            tMax (float): Maximum time value
            pretrain (bool): Whether to use pretraining
            pretrain_iters (int): Number of pretraining iterations
            counter_start (int): Start value for curriculum counter
            counter_end (int): End value for curriculum counter
            num_src_samples (int): Number of source (t=0) samples
            seed (int): Random seed for reproducibility
            device (torch.device): Device to store tensors
            counterexample (torch.Tensor, optional): Counterexample points [n, state_dim]
            percentage_in_counterexample (float): Percentage of points near counterexample
            percentage_at_t0 (float): Percentage of points at t=0
            epsilon_radius (float): Radius around counterexample points to sample
            num_states (int): Number of state dimensions
            compute_boundary_values (callable): Function that computes boundary values
                                            signature: f(coords: torch.Tensor) -> torch.Tensor
        """
        super().__init__()
        torch.manual_seed(seed)
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        logger.debug(f"Initializing dataset with {numpoints} points")

        self.numpoints = numpoints
        self.tMin = tMin
        self.tMax = tMax
        
        self.pretrain = pretrain
        self.pretrain_iters = pretrain_iters
        self.pretrain_counter = 0

        self.counter = counter_start
        self.full_count = counter_end
        self.N_src_samples = num_src_samples

        if num_states is None:
            raise ValueError("num_states must be specified")
        self.num_states = num_states

        if compute_boundary_values is None:
            # Default spherical boundary
            def default_boundary(coords):
                pos = coords[:, 1:1+num_states]  # Extract state variables
                boundary_values = torch.norm(pos, dim=1, keepdim=True)
                return boundary_values - 0.25  # Default radius
            
            self.compute_boundary_values = default_boundary
        else:
            self.compute_boundary_values = compute_boundary_values

        # Counterexample parameters
        self.counterexample = counterexample.to(self.device) if counterexample is not None else None
        self.percentage_in_counterexample = percentage_in_counterexample
        self.percentage_at_t0 = percentage_at_t0
        self.epsilon_radius = epsilon_radius

    def __len__(self):
        """Dataset length is always 1 as we generate data dynamically."""
        return 1

    def _get_time_samples(self):
        """Generate time samples based on training phase."""
        start_time = 0.0

        if self.pretrain:
            # During pretraining, all samples are at t=0
            time = torch.ones(self.numpoints, 1, device=self.device) * start_time
        else:
            # Progressive sampling during training
            time = self.tMin + torch.zeros(self.numpoints, 1, device=self.device).uniform_(
                0, (self.tMax - self.tMin) * (self.counter / self.full_count))
            # Ensure some samples at t=0
            time[-self.N_src_samples:, 0] = start_time

        return time, start_time

    def _update_counters(self):
        """Update curriculum learning counters."""
        if self.pretrain and self.pretrain_counter == self.pretrain_iters - 1:
            logger.info("Pretraining phase completed")
        if self.pretrain:
            self.pretrain_counter += 1
            if self.pretrain_counter == self.pretrain_iters:
                self.pretrain = False
        elif self.counter < self.full_count:
            self.counter += 1

    def _sample_near_counterexample(self, num_points):
        """
        Sample points near the counterexample points including time dimension.
        
        Args:
            num_points (int): Number of points to sample
            
        Returns:
            torch.Tensor: Sampled points of shape (num_points, num_states + 1)
        """
        if self.counterexample is None:
            return None
            
        counter_idx = torch.randint(0, self.counterexample.shape[0], (num_points,))
        # Create a new tensor that requires gradients
        counter_points = self.counterexample[counter_idx].detach().clone()
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
        
        return torch.zeros(num_points, self.num_states, device=self.device).uniform_(-1, 1)

    def __getitem__(self, idx):
        """Base implementation for getting samples."""
        # Get time samples first
        time, start_time = self._get_time_samples()
        
        # Calculate number of points for each category
        n_t0 = int(self.numpoints * self.percentage_at_t0 / 100)
        
        if self.counterexample is not None:
            n_counter = int(self.numpoints * self.percentage_in_counterexample / 100)
            n_random = self.numpoints - n_counter - n_t0
            
            # Sample counterexample points (includes time)
            counter_coords = self._sample_near_counterexample(n_counter)
            
        else:
            n_random = self.numpoints - n_t0
            
        # Sample remaining points
        random_states = self._sample_state_space(n_random)
        t0_states = self._sample_state_space(n_t0)
        
        # Combine with respective time coordinates
        random_coords = torch.cat([time[:n_random], random_states], dim=1)
        t0_coords = torch.cat([torch.zeros(n_t0, 1, device=self.device), t0_states], dim=1)
        
        # Combine all coordinates
        if self.counterexample is not None:
            coords = torch.cat([counter_coords, random_coords, t0_coords], dim=0)
        else:
            coords = torch.cat([random_coords, t0_coords], dim=0)

        # Compute boundary values
        boundary_values = self.compute_boundary_values(coords)

        # Create Dirichlet mask
        if self.pretrain:
            dirichlet_mask = torch.ones(coords.shape[0], 1, device=self.device) > 0
        else:
            dirichlet_mask = (coords[:, 0, None] == start_time)

        # Update curriculum learning counters
        self._update_counters()

        return {'coords': coords}, {
            'source_boundary_values': boundary_values,
            'dirichlet_mask': dirichlet_mask
        }

