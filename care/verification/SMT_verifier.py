import time
import logging
import torch
import json
from typing import Dict, Any, Tuple, Callable, Optional
import multiprocessing as mp
from care.verification.verifier_utils.symbolic import extract_symbolic_model
from care.verification.verifier_utils.dreal_utils import extract_dreal_partials
try:
    import z3
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
if Z3_AVAILABLE:
    from care.verification.verifier_utils.z3_utils import extract_z3_partials
from care.verification.verifier_utils.constraint_builder import (
    prepare_constraint_data_batch,
    process_check_advanced,
    serialize_expression,
    parse_counterexample as parse_ce,
    function_maps,
    create_constraint_data
)

logger = logging.getLogger(__name__)

def verify_with_SMT(value_fn, partials_variables, variables, compute_hamiltonian, compute_boundary, solver_name, epsilon=0.5, epsilon_ratio=0.05, delta = 0.001,
                      reach_mode='forward', min_with='none', set_type='set', save_directory="./", execution_mode="parallel", additional_constraints=None):
    """
    Verifies if the HJB equation holds using dReal for a double integrator system.
    
    Parameters:
      ...
      reach_mode (str): 'forward' (default) or 'backward' for reach set computation
      min_with (str): Specifies minimum value computation method ('none', or 'target')
      set_type (str): 'set' (default) or 'tube' for target set type
      execution_mode (str): "parallel" (default) runs constraint checks concurrently,
                           "sequential" runs the boundary and derivative checks in sequence while timing each.
    """
    
    
    # Extract state variables and partial derivatives dynamically
    state_vars = []
    time_vars = []
    partial_vars = []
    for key, value in variables.items():
        if key.startswith("x_1_"):
            if key.endswith("_1"):
                time_vars.append(value)
            else:
                state_vars.append(value)
        elif key.startswith("partial_x_1_"):
            partial_vars.append(value)

    func_map = function_maps[solver_name]

    # Use class method for Hamiltonian computation and other setup
    use_partial_expression = True
    if use_partial_expression:
        hamiltonian_value = compute_hamiltonian(state_vars, [value for key, value in partials_variables.items() if not key.endswith("_1")], func_map)
    else:
        hamiltonian_value = compute_hamiltonian(state_vars, partial_vars[1:], func_map)

    if set_type == 'tube': 
        hamiltonian_value = func_map['max'](hamiltonian_value, 0)

    if reach_mode == 'backward':
        hamiltonian_value = -hamiltonian_value

    boundary_value = compute_boundary(state_vars)

    # Serialize expressions
    hamiltonian_expr = serialize_expression(hamiltonian_value, solver_name)
    value_fn_expr = serialize_expression(value_fn, solver_name)
    partials_expr = {key: serialize_expression(val, solver_name) for key, val in partials_variables.items()}
    boundary_expr = serialize_expression(boundary_value, solver_name)

    result = None
    timing_info = {}
    
    if execution_mode == "parallel":
        logger.info("Starting parallel constraint checks with multiprocessing.Pool...")
        
        # Prepare data for parallel execution
        state_dim = len(state_vars)
        
        # Use the constraint builder to create serializable constraint data
        constraint_data_batch = prepare_constraint_data_batch(
            state_dim=state_dim,
            epsilon=epsilon,
            epsilon_ratio=epsilon_ratio,
            delta=delta,
            min_with=min_with,
            reach_mode=reach_mode,
            set_type=set_type
        )
        
        # Create a multiprocessing pool
        pool = mp.Pool()
        
        # Track async results
        async_results = []
        
        # Submit all tasks asynchronously
        for constraint_data in constraint_data_batch:
            async_result = pool.apply_async(
                process_check_advanced,
                args=(
                    solver_name if constraint_data['constraint_type'] not in ['boundary_1', 'boundary_2', 'target_1', 'target_3'] else 'dreal',
                    constraint_data,
                    hamiltonian_expr,
                    value_fn_expr,
                    boundary_expr,
                    partials_expr
                )
            )

            async_results.append(async_result)
        
        try:
            # Wait for results and process them as they arrive
            remaining_results = list(range(len(async_results)))  # Track indices of remaining results
            
            while remaining_results and not result:
                # Check each remaining result without modifying the list during iteration
                i = 0
                while i < len(remaining_results):
                    idx = remaining_results[i]
                    async_result = async_results[idx]
                    
                    if async_result.ready():
                        try:
                            constraint_id, constraint_result = async_result.get(0)  # Non-blocking
                            
                            # Remove this result from remaining_results
                            remaining_results.pop(i)
                            
                            if constraint_result and not constraint_result.startswith("Error"):
                                # We found a counterexample
                                result = constraint_result
                                logger.info(f"Found counterexample in constraint {constraint_id}")
                                break
                            elif constraint_result and constraint_result.startswith("Error"):
                                logger.error(f"Error in constraint {constraint_id}: {constraint_result}")
                        except Exception as e:
                            logger.error(f"Error getting result: {e}")
                            # Still remove this result even if there was an error
                            remaining_results.pop(i)
                    else:
                        # Move to next index only if we didn't remove the current one
                        i += 1
                
                # If we found a counterexample or all tasks are done, break
                if result or not remaining_results:
                    break
                
                # Sleep briefly to avoid high CPU usage
                time.sleep(5)
                
        finally:
            # Immediately terminate the pool if a counterexample was found
            if result:
                logger.info("Terminating process pool due to counterexample found")
                pool.terminate()
            else:
                logger.info("Closing process pool normally")
                pool.close()
                
            # Always join the pool to clean up resources properly
            pool.join()
            
        logger.info("Parallel constraint checks completed.")
    
    elif execution_mode == "sequential":
        constraint_data = create_constraint_data(1, 'derivative_boundary', False, len(state_vars), epsilon, delta, 
                                      reach_mode, set_type, (0.0, 1.0))
        result = process_check_advanced(
                    solver_name,
                    constraint_data,
                    hamiltonian_expr,
                    value_fn_expr,
                    boundary_expr,
                    partials_expr
                )
    else:
        logger.error(f"Unknown execution_mode: {execution_mode}.")

    if not result:
        success = True  # HJB Equation is satisfied
        logger.info("No counterexamples found in checks.")
        verification_result = {
            "epsilon": epsilon,
            "result": "HJB Equation Satisfied",
            "counterexample": None,
            "timing": timing_info
        }
    else:
        success = False  # HJB Equation is not satisfied
        verification_result = {
            "epsilon": epsilon,
            "result": "HJB Equation Not Satisfied",
            "counterexample": parse_ce(str(result)),
            "timing": timing_info
        }
    
    # Optionally save result to file
    result_file = f"{save_directory}/result.json"
    with open(result_file, "w") as f:
        json.dump(verification_result, f, indent=4)
    logger.debug(f"Saved result to {result_file}")

    counterexample = parse_ce(str(result)) if not success else None

    return success, counterexample

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
        self._solver = None
        self.delta = None
        
    def _select_solver(self, symbolic_model, epsilon):
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
        self.delta = 0.001  # Default delta for dReal verification

        if self.solver_preference in ['z3', 'marabou'] and has_trig:
            logger.info("Model contains trigonometric functions: Using dReal for verification")
            self._solver =  'dreal'  # Use dReal for models with trigonometric functions
        elif self.solver_preference == 'auto':
            if has_trig:
                logger.info("Model contains trigonometric functions: Using dReal for verification")
                self._solver =  'dreal'
            else:
                # Need to add delta here for constraints that are solved with delta
                self._solver =  'dreal'
        else:
            if self.solver_preference == 'z3' and not Z3_AVAILABLE:
                logger.warning("Z3 solver not available. Falling back to dReal.")
                self._solver = 'dreal'
            elif self.solver_preference == 'marabou' and not marabou_AVAILABLE:
                logger.warning("Marabou solver not available. Falling back to dReal.")
                self._solver = 'dreal'
            else:
                self._solver = self.solver_preference
        
        return self._solver

    def validate_counterexample(
        self,
        counterexample: torch.Tensor,
        loss_fn: Callable,
        compute_boundary: Callable,
        epsilon_bndry: float,
        epsilon_diff: float,
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
        
        if self._solver == 'dreal':
            pde_result_delta = self.delta
        else:
            pde_result_delta = 0.0

        # Determine which condition is violated based on epsilon
        if boundary_diff is not None and boundary_diff > epsilon_bndry - self.delta:
            result['is_valid_ce'] = True
            result['violation_type'] = 'boundary'
            result['violation_amount'] = boundary_diff
            logger.info(f"Valid counterexample: Boundary condition violated by {boundary_diff:.6f} > {epsilon_bndry - self.delta}")
        
        elif pde_residual > epsilon_diff - pde_result_delta:
            result['is_valid_ce'] = True
            result['violation_type'] = 'pde'
            result['violation_amount'] = pde_residual
            logger.info(f"Valid counterexample: PDE violated by {pde_residual:.6f} > {epsilon_diff - pde_result_delta}")
        
        else:
            boundary_info = f"Boundary diff: {boundary_diff:.6f}, " if boundary_diff is not None else ""
            logger.warning(f"{boundary_info}PDE residual: {pde_residual:.6f}")
            raise ValueError(f"Invalid counterexample: No violation exceeds epsilon={epsilon_diff}")
        
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
        epsilon_ratio: float = 0.05
    ) -> Tuple[bool, Optional[torch.Tensor], Dict[str, float]]:
        """
        Verify a trained model using dReal/Z3/Marabou.
        
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
        solver_name = self._select_solver(symbolic_model, epsilon)
        logger.info(f"Selected {solver_name} solver for verification")
        
        # Time verification setup and execution
        t_verify_start = time.time()
        
        if solver_name == 'z3':
            result = extract_z3_partials(symbolic_model)
        elif solver_name in ['dreal', 'marabou']:
            result = extract_dreal_partials(symbolic_model)
        else:
            NotImplementedError(f"Solver {solver_name} is not yet supported.")

        success, counterexample = verify_with_SMT(
            value_fn=result["value_fn"],
            partials_variables=result["partials"],
            variables=result["variables"],
            compute_hamiltonian=compute_hamiltonian,
            compute_boundary=compute_boundary,
            epsilon=epsilon,
            epsilon_ratio=epsilon_ratio,
            delta=self.delta,
            solver_name = solver_name,
            reach_mode=system_specifics.get('reach_mode', 'forward'),
            min_with=system_specifics.get('min_with', 'none'),
            set_type=system_specifics.get('set_type', 'set'),
            save_directory=system_specifics['root_path'],
            additional_constraints=system_specifics.get('additional_constraints', None)
        )
    
        # Convert counterexample to tensor format if found
        if not success and counterexample:
            if type(counterexample) is dict:
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
                counterexample = torch.tensor(counterexample, device=self.device)
        else:
            counterexample = None
        
        timing_info['verification_time'] = time.time() - t_verify_start
        
        logger.info(f"Verification completed: {'successful' if success else 'failed'}")
        logger.debug(f"Symbolic generation: {timing_info.get('symbolic_time', 0):.2f}s, " 
                    f"Verification: {timing_info['verification_time']:.2f}s")
        
        return success, counterexample, timing_info