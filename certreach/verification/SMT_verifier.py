import time
import logging
import torch
from typing import Dict, Any, Tuple, Callable, List, Optional, Union
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
            solver_preference: Preferred solver ('z3', 'dreal', or 'auto')
        """
        self.device = device
        self.solver_preference = solver_preference
        
    def _select_solver(self, symbolic_model):
        """
        Select appropriate solver based on model characteristics and user preference.
        
        Args:
            symbolic_model: The symbolic model to analyze
            
        Returns:
            str: Selected solver name ('z3' or 'dreal')
        """
        # If user explicitly specified a solver, use that
        if self.solver_preference in ['z3', 'dreal']:
            return self.solver_preference
            
        # Auto-select based on model properties
        # Check if model has trigonometric functions (better with dReal)
        has_trig = (str(str(symbolic_model)).find('sin') >= 0)  or (str(str(symbolic_model)).find('cos') >= 0)
        
        if has_trig:
            return 'dreal'
        else:
            return 'z3'  # Default to Z3 for simpler models

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
        Verify a trained model using dReal/Z3 with option to reuse symbolic model.
        
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
        try:
            symbolic_model = extract_symbolic_model(model_state, model_config, system_specifics['root_path'])
            timing_info['symbolic_time'] = time.time() - t_symbolic_start
            logger.debug("Symbolic model extracted successfully")
        except Exception as e:
            logger.error("Failed to extract symbolic model: %s", str(e))
            return False, None, {'error': 'symbolic_extraction_failed', 'time': time.time() - t_symbolic_start}
        
        # Select solver
        solver = self._select_solver(symbolic_model)
        logger.info(f"Selected {solver} solver for verification")
        
        try:
            # Time verification setup and execution
            t_verify_start = time.time()
            
            if solver == 'dreal':
                # Use dReal for verification
                result = extract_dreal_partials(symbolic_model)
                success, counterexample = verify_with_dreal(
                    d_real_value_fn=result["d_real_value_fn"],
                    dreal_partials=result["dreal_partials"],
                    dreal_variables=result["dreal_variables"],
                    compute_hamiltonian=compute_hamiltonian,
                    compute_boundary=compute_boundary,
                    epsilon=epsilon,
                    reachMode=system_specifics.get('reachMode', 'forward'),
                    setType=system_specifics.get('setType', 'set'),
                    save_directory=system_specifics['root_path'],
                    execution_mode="sequential",  # Use sequential for better timing info
                    additional_constraints=system_specifics.get('additional_constraints', None)
                )
            else:
                # Use Z3 for verification
                result = extract_z3_partials(symbolic_model)
                success, counterexample = verify_with_z3(
                    z3_value_fn=result["z3_value_fn"],
                    z3_partials=result["z3_partials"],
                    z3_variables=result["z3_variables"],
                    compute_hamiltonian=compute_hamiltonian,
                    compute_boundary=compute_boundary,
                    epsilon=epsilon,
                    reachMode=system_specifics.get('reachMode', 'forward'),
                    setType=system_specifics.get('setType', 'set'),
                    save_directory=system_specifics['root_path'],
                    additional_constraints=system_specifics.get('additional_constraints', None)
                )
                
            timing_info['verification_time'] = time.time() - t_verify_start
            
            # Convert counterexample to tensor format if found
            ce_tensor = None
            if not success and counterexample:
                ce_list = []
                for key in sorted(counterexample.keys()):
                    if key.startswith('x_'):  # Only include state variables
                        interval = counterexample[key]
                        # Take midpoint of interval as the counterexample point
                        ce_list.append((interval[0] + interval[1]) / 2)
                
                if ce_list:
                    ce_tensor = torch.tensor(ce_list, device=self.device)
                    logger.info(f"Counterexample found: {ce_tensor}")
            
            logger.info(f"Verification completed: {'successful' if success else 'failed'}")
            logger.debug(f"Symbolic generation: {timing_info.get('symbolic_time', 0):.2f}s, " 
                       f"Verification: {timing_info['verification_time']:.2f}s")
            
            return success, ce_tensor, timing_info
            
        except Exception as e:
            logger.error("Verification failed with error: %s", str(e))
            timing_info['verification_time'] = time.time() - t_verify_start
            timing_info['error'] = str(e)
            return False, None, timing_info