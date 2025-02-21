import time
import logging
import torch
from typing import Dict, Any, Tuple, Optional, Callable
from certreach.verification.symbolic import extract_symbolic_model
from certreach.verification.dreal_utils import (
    extract_dreal_partials,
    verify_with_dreal
)
from dreal import And, Not, CheckSatisfiability

logger = logging.getLogger(__name__)

def verify_system(
    model_state: Dict[str, torch.Tensor],
    model_config: Dict[str, Any],
    system_specifics: Dict[str, Any],
    compute_hamiltonian: Callable,
    compute_boundary: Callable,
    epsilon: float,
    symbolic_model: Optional[Any] = None
) -> Tuple[Dict[str, Any], Dict[str, float], Any]:
    """
    Verify a trained model using dReal with option to reuse symbolic model.
    
    Args:
        model_state: The model's state dictionary
        model_config: The model's configuration (including architecture details)
        root_path: Path to save verification results
        system_type: Type of system to verify
        epsilon: Verification tolerance
        symbolic_model: Optional precomputed symbolic model
        
    Returns:
        Tuple[dict, dict, Any]: (Verification results, Timing information, Symbolic model)
    """
    timing_info = {}
    logger.info("Starting %s verification", system_specifics['name'])
    
    try:
        # Time symbolic model generation
        if symbolic_model is None:
            t_symbolic_start = time.time()
            symbolic_model = extract_symbolic_model(model_state, model_config, system_specifics['root_path'])
            timing_info['symbolic_time'] = time.time() - t_symbolic_start
            logger.debug("Symbolic model saved to %s", system_specifics['root_path'])
        else:
            timing_info['symbolic_time'] = 0.0
            logger.debug("Reusing existing symbolic model")

        # Time dReal verification setup and execution
        t_verify_start = time.time()
        result = extract_dreal_partials(symbolic_model)
        
        logger.info("Running dReal verification")
        verification_result = verify_with_dreal(
            d_real_value_fn=result["d_real_value_fn"],
            dreal_partials=result["dreal_partials"],
            dreal_variables=result["dreal_variables"],
            compute_hamiltonian=compute_hamiltonian,
            compute_boundary=compute_boundary,
            epsilon=epsilon,
            reachMode='forward',
            setType='set',
            save_directory=system_specifics['root_path'],
            additional_constraints=result["additional_conditions"] if "additional_conditions" in result else None
        )
        timing_info['verification_time'] = time.time() - t_verify_start
        
        logger.debug("Symbolic generation took: %.2fs", timing_info.get('symbolic_time', 0))
        logger.debug("Verification took: %.2fs", timing_info['verification_time'])
        
        logger.info("Verification completed with result: %s", verification_result.get('result', 'Unknown'))
        
        return verification_result, timing_info, symbolic_model
        
    except Exception as e:
        logger.error("Verification failed: %s", str(e))
        raise