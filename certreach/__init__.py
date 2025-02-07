# Core modules
from .learning.training import train
from .learning import loss_functions
from .common import dataio
from .verification import symbolic
from .verification import dreal_utils

# Classes and functions that should be directly accessible
from .learning.networks import SingleBVPNet, BatchLinear, Sine, FCBlock
from .common.dataio import (
    DoubleIntegratorDataset,
    get_experiment_folder,
    save_experiment_details,
    check_existing_experiment
)
from .verification.dreal_utils import (
    extract_dreal_partials,
    process_dreal_result,
    serializable_to_sympy,
    sympy_to_serializable,
    CounterexampleDataset
)
from .verification.symbolic import (
    get_symbolic_layer_output_generalized,
    compute_layer,
    combine_all_layers_parallelized,
    extract_symbolic_model
)

# Version info
__version__ = '0.1.0'
