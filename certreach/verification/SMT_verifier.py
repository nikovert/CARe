import time
import logging
import torch
from typing import Dict, Any, Tuple, Callable, Optional
from certreach.verification.verifier_utils.symbolic import extract_symbolic_model
from certreach.verification.verifier_utils.dreal_utils import (
    extract_dreal_partials,
    verify_with_dreal
)
from certreach.verification.verifier_utils.z3_utils import (
    extract_z3_partials,
    verify_with_z3
)

logger = logging.getLogger(__name__)

class SMTVerifier:
    NAME = "SMTVerifier"
    
    def __init__(self, device='cpu', solver_preference='auto'):
        """
        Initialize the SMT verifier.
        
        Args:
            device: Torch device to use
            solver_preference: Preferred solver ('z3', 'dreal', 'marabou', or 'auto')
        Raises:
            ValueError: If an invalid solver is selected.
        """
        self.device = device
        self.solver_preference = solver_preference
        self.delta = None
        
    def _select_solver(self, symbolic_model):
        """
        Select appropriate solver based on model characteristics and user preference.
        
        Args:
            symbolic_model: The symbolic model to analyze
            
        Returns:
            str: Selected solver name ('z3', 'dreal', or 'marabou')
        """
            
        # Auto-select based on model properties
        # Check if model has trigonometric functions (better with dReal)
        has_trig = (str(str(symbolic_model)).find('sin') >= 0) or (str(str(symbolic_model)).find('cos') >= 0)
        
        if self.solver_preference in ['z3', 'marabou'] and has_trig:
            logger.info("Model contains trigonometric functions: Using dReal for verification")
            self.delta = 0.01
            return 'dreal'  # Use dReal for models with trigonometric functions
        elif self.solver_preference == 'auto':
            if has_trig:
                logger.info("Model contains trigonometric functions: Using dReal for verification")
                self.delta = 0.01  # Default delta for dReal verification
                return 'dreal'
            else:
                return 'z3'
        else:
            return self.solver_preference

    def validate_counterexample(
        self,
        counterexample: torch.Tensor,
        loss_fn: Callable,
        compute_boundary: Callable,
        epsilon: float,
        model: torch.nn.Module
    ) -> Dict[str, Any]:
        """
        Validate if a counterexample actually violates the HJ PDE or boundary conditions.
        
        Args:
            counterexample: Tensor representing the counterexample point
            loss_fn: Loss function to compute the PDE residual
            compute_boundary: Function to compute boundary conditions
            epsilon: Verification tolerance
            model: Pre-loaded model instance
            
        Returns:
            Dict[str, Any]: Validation results including violation details
        """
        logger.info(f"Validating counterexample: {counterexample.cpu().numpy()}")
        
        # Initialize validation result
        result = {
            'is_valid_ce': False,
            'violation_type': None,
            'violation_amount': 0.0,
            'details': {},
            'counterexample': counterexample.cpu().numpy()
        }
        
        model.eval()
        
        # Ensure counterexample is on the correct device
        if counterexample.device != self.device:
            counterexample = counterexample.to(self.device)
        
        # Convert counterexample to required format for model input and ensure it requires gradients
        ce_input = {'coords': counterexample.clone().detach().requires_grad_(True).unsqueeze(0)}
        ce_states = counterexample[1:].unsqueeze(0)
        
        # Get model output
        model_out = model(ce_input)
        boundary_value = compute_boundary(ce_states)
        
        gt = {'source_boundary_values': boundary_value, 'dirichlet_mask': counterexample[0] == 0.0}
        loss = loss_fn(model_out, gt)  # Compute loss to get gradients
        
        pde_residual = loss['diff_constraint_hom'].item()
        if counterexample[0] == 0.0: 
            boundary_diff = loss['dirichlet'].item()
        else:
            boundary_diff = None
        
        result['details']['pde_residual'] = pde_residual
        
        if self.delta:
            delta = self.delta
            logger.info(f"Using delta={delta} for validation")
        else:
            delta = 0

        # Determine which condition is violated based on epsilon
        if boundary_diff is not None and boundary_diff > epsilon-delta:
            result['is_valid_ce'] = True
            result['violation_type'] = 'boundary'
            result['violation_amount'] = boundary_diff
            logger.info(f"Valid counterexample: Boundary condition violated by {boundary_diff:.6f} > {epsilon-delta}")
        
        elif pde_residual > epsilon-delta:
            result['is_valid_ce'] = True
            result['violation_type'] = 'pde'
            result['violation_amount'] = pde_residual
            logger.info(f"Valid counterexample: PDE violated by {pde_residual:.6f} > {epsilon-delta}")
        
        else:
            boundary_info = f"Boundary diff: {boundary_diff:.6f}, " if boundary_diff is not None else ""
            logger.warning(f"{boundary_info}PDE residual: {pde_residual:.6f}")
            raise ValueError(f"Invalid counterexample: No violation exceeds epsilon={epsilon}")
        
        # Return detailed validation results
        return result

    def verify_system(
        self,
        model_state: Dict[str, torch.Tensor],
        model_config: Dict[str, Any],
        system_specifics: Dict[str, Any],
        compute_hamiltonian: Callable,
        compute_boundary: Callable,
        epsilon: float,
    ) -> Tuple[bool, Optional[torch.Tensor], Dict[str, float]]:
        """
        Verify a trained model using dReal/Z3.
        
        Args:
            model_state: The model's state dictionary
            model_config: The model's configuration (including architecture details)
            system_specifics: System-specific information including paths and parameters
            compute_hamiltonian: Function to compute the Hamiltonian
            compute_boundary: Function to compute the boundary conditions
            epsilon: Verification tolerance
            
        Returns:
            Tuple[bool, Optional[torch.Tensor], Dict[str, float]]: 
                (Success flag, Counterexample if any, Timing information)
        """
        timing_info = {}
        logger.info("Starting %s verification", system_specifics.get('name', 'system'))
        
        # Generate symbolic model
        t_symbolic_start = time.time()
        symbolic_model = extract_symbolic_model(model_state, model_config, system_specifics['root_path'])
        timing_info['symbolic_time'] = time.time() - t_symbolic_start
        logger.debug("Symbolic model extracted successfully")
        
        # Select solver
        solver = self._select_solver(symbolic_model)
        logger.info(f"Selected {solver} solver for verification")
        
        # Time verification setup and execution
        t_verify_start = time.time()
        
        if solver == 'dreal':
            # Use dReal for verification
            result = extract_dreal_partials(symbolic_model)
            self.delta = min(epsilon / 10, 0.01)
            success, counterexample = verify_with_dreal(
                d_real_value_fn=result["d_real_value_fn"],
                dreal_partials=result["dreal_partials"],
                dreal_variables=result["dreal_variables"],
                compute_hamiltonian=compute_hamiltonian,
                compute_boundary=compute_boundary,
                epsilon=epsilon,
                delta=self.delta,
                reach_mode=system_specifics.get('reach_mode', 'forward'),
                min_with=system_specifics.get('min_with', 'none'),
                set_type=system_specifics.get('set_type', 'set'),
                save_directory=system_specifics['root_path'],
                execution_mode="parallel",  # Use sequential for better timing info
                additional_constraints=system_specifics.get('additional_constraints', None)
            )
        
            # Convert counterexample to tensor format if found
            if not success and counterexample:
                ce_list = []
                for key in sorted(counterexample.keys()):
                    if key.startswith('x_'):  # Only include state variables
                        interval = counterexample[key]
                        # Take midpoint of interval as the counterexample point
                        ce_list.append((interval[0] + interval[1]) / 2)
                
                if ce_list:
                    counterexample = torch.tensor(ce_list, device=self.device)
                    logger.info(f"Counterexample found: {counterexample}")
            else:
                counterexample = None
        
        elif solver == 'z3':
            # Use Z3 for verification
            result = extract_z3_partials(symbolic_model)
            success, counterexample = verify_with_z3(
                z3_value_fn=result["z3_value_fn"],
                z3_partials=result["z3_partials"],
                z3_variables=result["z3_variables"],
                compute_hamiltonian=compute_hamiltonian,
                compute_boundary=compute_boundary,
                epsilon=epsilon,
                reach_mode=system_specifics.get('reach_mode', 'forward'),
                min_with=system_specifics.get('min_with', 'none'),
                set_type=system_specifics.get('set_type', 'set'),
                save_directory=system_specifics['root_path'],
                additional_constraints=system_specifics.get('additional_constraints', None)
            )
            counterexample = torch.tensor(counterexample, device=self.device) if counterexample else None
        
        elif solver == 'marabou':
            # Use Marabou for verification
            raise NotImplementedError("Marabou verification is not yet supported")
        else:
            raise ValueError(f"Invalid solver selected: {solver}")
            
        timing_info['verification_time'] = time.time() - t_verify_start
        
        logger.info(f"Verification completed: {'successful' if success else 'failed'}")
        logger.debug(f"Symbolic generation: {timing_info.get('symbolic_time', 0):.2f}s, " 
                    f"Verification: {timing_info['verification_time']:.2f}s")
        
        return success, counterexample, timing_info