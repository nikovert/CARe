import time
import logging
from typing import Dict, Any, Tuple
from certreach.verification.symbolic import extract_symbolic_model
from certreach.verification.dreal_utils import (
    dreal_double_integrator_BRS,
    dreal_triple_integrator_BRS,
    dreal_three_state_system_BRS,
    extract_dreal_partials,
    process_dreal_result
)

logger = logging.getLogger(__name__)

def verify_system(model, root_path, system_type='double_integrator', epsilon=0.35) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Verify a trained model using dReal.
    
    Args:
        model: The trained PyTorch model
        root_path: Path to save verification results
        system_type: Type of system to verify ('double_integrator', 'triple_integrator', 'three_state')
        epsilon: Verification tolerance
        
    Returns:
        Tuple[dict, dict]: (Verification results, Timing information)
    """
    timing_info = {}
    logger.info(f"Starting {system_type} verification")
    model = model.cpu()
    
    logger.info("Extracting symbolic model")
    try:
        # Time symbolic model generation
        t_symbolic_start = time.time()
        symbolic_model = extract_symbolic_model(model, root_path)
        timing_info['symbolic_time'] = time.time() - t_symbolic_start

        # Time dReal verification setup and execution
        t_verify_start = time.time()
        in_features = 3 if system_type == 'double_integrator' else 4
        result = extract_dreal_partials(symbolic_model, in_features=in_features)
        
        logger.info("Running dReal verification")
        verification_fn = {
            'double_integrator': dreal_double_integrator_BRS,
            'triple_integrator': dreal_triple_integrator_BRS,
            'three_state': dreal_three_state_system_BRS
        }.get(system_type)
        
        if not verification_fn:
            raise ValueError(f"Unsupported system type: {system_type}")
        
        verification_result = verification_fn(
            dreal_partials=result["dreal_partials"],
            dreal_variables=result["dreal_variables"],
            epsilon=epsilon,
            reachMode='forward',
            reachAim='reach',
            setType='set',
            save_directory=root_path
        )
        timing_info['verification_time'] = time.time() - t_verify_start
        
        logger.info("Processing verification results")
        logger.info(f"Symbolic generation took: {timing_info['symbolic_time']:.2f}s")
        logger.info(f"Verification took: {timing_info['verification_time']:.2f}s")
        
        return process_dreal_result(f"{root_path}/dreal_result.json"), timing_info
        
    except Exception as e:
        logger.error(f"Verification failed: {str(e)}")
        raise
