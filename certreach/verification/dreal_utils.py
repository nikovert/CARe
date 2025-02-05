import json
from dreal import And, Not, CheckSatisfiability, Variable
import dreal
import sympy
import torch
import sympy
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)

def convert_symbols_to_dreal(input_symbols):
    """
    Convert SymPy symbols to dReal variables.

    Args:
        input_symbols (sympy.Matrix): Matrix of SymPy symbols.

    Returns:
        dict: A dictionary mapping SymPy symbol names to corresponding dReal variables.
    """
    logger = logging.getLogger(__name__)
    # Create a dictionary mapping SymPy symbols to dReal variables
    dreal_variables = {str(sym): Variable(str(sym)) for sym in input_symbols}

    # Print the mapping for debugging and verification
    logger.debug("Conversion from SymPy to dReal:")
    for sym, dreal_var in dreal_variables.items():
        logger.debug(f"{sym} -> {dreal_var}")

    return dreal_variables

def sympy_to_dreal_converter(syms: dict, exp: sympy.Expr, to_number=lambda x: float(x), expand_pow=True):
    """
    Convert a SymPy expression to a dReal-compatible expression.

    Args:
        syms (dict): Dictionary mapping SymPy symbols to dReal variables.
        exp (sympy.Expr): The SymPy expression to be converted.
        to_number (callable): Function for numeric conversion (default: float).
        expand_pow (bool): Whether to expand powers manually (default: True).

    Returns:
        dReal expression: The equivalent expression in dReal format.
    
    Raises:
        ValueError: If the expression cannot be converted.
    """

    # Handle symbols: Look up corresponding dReal variable
    if isinstance(exp, sympy.Symbol):
        exp_str = str(exp)
        if exp_str not in syms:
            raise ValueError(f"[Error] Symbol '{exp_str}' not found in provided symbols dictionary. "
                             f"Available symbols: {list(syms.keys())}")
        return syms[exp_str]

    # Handle numeric constants
    elif isinstance(exp, sympy.Number):
        try:
            return to_number(exp)
        except Exception:
            return sympy.Float(exp, len(str(exp)))

    # Handle addition: Sum all terms
    elif isinstance(exp, sympy.Add):
        terms = [sympy_to_dreal_converter(syms, arg, to_number, expand_pow) for arg in exp.args]
        return sum(terms)

    # Handle multiplication: Multiply all terms
    elif isinstance(exp, sympy.Mul):
        terms = [sympy_to_dreal_converter(syms, arg, to_number, expand_pow) for arg in exp.args]
        result = terms[0]
        for term in terms[1:]:
            result *= term
        return result

    # Handle exponentiation (powers)
    elif isinstance(exp, sympy.Pow):
        base = sympy_to_dreal_converter(syms, exp.args[0], to_number, expand_pow)
        exponent = sympy_to_dreal_converter(syms, exp.args[1], to_number, expand_pow)

        # Expand integer powers if requested
        if expand_pow:
            try:
                exp_val = float(exponent)
                if exp_val.is_integer():
                    exp_val = int(exp_val)
                    result = base
                    for _ in range(exp_val - 1):
                        result *= base
                    return result
            except Exception:
                pass

        # Use the power operator if expansion fails
        return base**exponent

    # Handle standard mathematical functions
    elif isinstance(exp, sympy.Function):
        func_map = {
            sympy.sin: dreal.sin,    # Sine function
            sympy.cos: dreal.cos,    # Cosine function
            sympy.exp: dreal.exp,    # Exponential function
            sympy.tanh: dreal.tanh,  # Hyperbolic tangent
        }

        for sympy_func, dreal_func in func_map.items():
            if isinstance(exp, sympy_func):
                # Convert the argument and apply the corresponding dReal function
                arg = sympy_to_dreal_converter(syms, exp.args[0], to_number, expand_pow)
                return dreal_func(arg)

    # Raise an error if the term is unsupported
    logger.error(f"Unsupported term: {exp} (type: {type(exp)})")
    raise ValueError(f"[Error] Unsupported term: {exp} (type: {type(exp)})")

def extract_dreal_partials(final_symbolic_expression, in_features):
    """
    Extracts dReal-compatible variables and partial derivatives 
    from a given symbolic expression.

    Args:
        final_symbolic_expression (sympy.Matrix): The symbolic expression from the neural network.
        in_features (int): The number of input features in the model.

    Returns:
        dict: A dictionary containing dReal variables and their partial derivatives.
    """
    # Define SymPy input symbols
    input_symbols = sympy.Matrix([sympy.symbols(f"x_1_{i+1}") for i in range(in_features)])

    # Compute symbolic partial derivatives
    partials = [final_symbolic_expression[0].diff(var) for var in input_symbols]

    # Convert SymPy symbols to dReal variables
    dreal_variables = convert_symbols_to_dreal(input_symbols)

    # Convert symbolic partial derivatives to dReal expressions
    dreal_partials = {
        f"partial_x_1_{i+1}": sympy_to_dreal_converter(dreal_variables, partial)
        for i, partial in enumerate(partials)
    }

    # Return all relevant variables and partial derivatives
    return {
        "input_symbols": input_symbols,
        "partials": partials,
        "dreal_variables": dreal_variables,
        "dreal_partials": dreal_partials,
        **{f"sympy_partial_{i+1}": partial for i, partial in enumerate(partials)},
        **{f"dreal_partial_{i+1}": dreal_partials[f"partial_x_1_{i+1}"] for i in range(in_features)}
    }

def sympy_to_serializable(obj):
    """
    Converts SymPy expressions into JSON serializable strings.

    This function ensures that complex SymPy objects such as matrices, dictionaries, 
    and lists are converted into a format that can be saved as JSON. Floats are 
    converted with high precision to avoid rounding errors.

    Args:
        obj: The SymPy object or any nested structure containing SymPy expressions.

    Returns:
        A JSON-compatible representation of the object.
    """

    # Check if the object is a SymPy Basic type (expression or number)
    if isinstance(obj, sympy.Basic):
        # Convert to string with 17 decimal places to preserve precision
        return str(sympy.N(obj, 17))  

    # Check if the object is a SymPy Matrix
    if isinstance(obj, sympy.Matrix):
        # Recursively process each element of the matrix
        return [sympy_to_serializable(obj[i]) for i in range(obj.shape[0])]

    # Check if the object is a dictionary
    if isinstance(obj, dict):
        # Recursively process each key-value pair in the dictionary
        return {k: sympy_to_serializable(v) for k, v in obj.items()}

    # Check if the object is a list
    if isinstance(obj, list):
        # Recursively process each element in the list
        return [sympy_to_serializable(v) for v in obj]

    # If the object is not recognized, return it as is
    return obj

def serializable_to_sympy(data):
    """
    Restores serialized strings back into SymPy expressions.

    This function reconstructs SymPy objects from data stored as JSON-compatible
    formats such as strings, lists, or dictionaries. It ensures that serialized 
    symbolic expressions are correctly restored to SymPy types for further computation.

    Args:
        data: The serialized JSON-compatible data (strings, lists, or dictionaries).

    Returns:
        Restored SymPy expression, list, or dictionary of expressions.
    """

    # If the data is a string, attempt to parse it as a SymPy expression
    if isinstance(data, str):
        return sympy.sympify(data, evaluate=False)  # Prevent automatic simplification

    # If the data is a list, process each element recursively
    if isinstance(data, list):
        return [serializable_to_sympy(v) for v in data]

    # If the data is a dictionary, process each key-value pair recursively
    if isinstance(data, dict):
        return {k: serializable_to_sympy(v) for k, v in data.items()}

    # If the data is neither string, list, nor dictionary, return as is
    return data

def process_dreal_result(json_path):
    """
    Process the dReal result from a JSON file to determine whether the HJB Equation is satisfied,
    display the epsilon value, and optionally return the counterexample if verification is not satisfied.

    Args:
        json_path (str): Path to the JSON file containing dReal results.

    Returns:
        dict: Parsed dReal result including epsilon, result details, and counterexample range if applicable.
    """
    logger = logging.getLogger(__name__)
    def parse_counterexample(result_str):
        """
        Parse the counterexample from the dReal result string.

        Args:
            result_str (str): The result string from the dReal output.

        Returns:
            dict: Parsed counterexample as a dictionary of variable ranges.
        """
        counterexample = {}
        try:
            for line in result_str.strip().split('\n'):
                # Parse each variable and its range
                variable, value_range = line.split(':')
                lower, upper = map(float, value_range.strip('[] ').split(','))
                counterexample[variable.strip()] = (lower, upper)
        except Exception as e:
            logger.error(f"Failed to parse counterexample: {e}")
            return None
        return counterexample

    try:
        # Load the JSON result from the specified file
        with open(json_path, "r") as file:
            dreal_result = json.load(file)

        # Extract epsilon value and result
        epsilon = dreal_result.get("epsilon", "Unknown")
        result = dreal_result.get("result", "Unknown")
        logger.info(f"Epsilon: {epsilon}")

        # Check and process the result
        if "HJB Equation Satisfied" in result:
            logger.info("dReal verification satisfied. HJB Equation is satisfied.")
            logger.info(f"Reachable Set: {dreal_result.get('set', 'Unknown')}")
            return {"epsilon": epsilon, "result": result, "counterexample": None}
        else:
            logger.info("dReal verification NOT satisfied. Counterexample found:")
            counterexample = parse_counterexample(result)
            if counterexample:
                for variable, (lower, upper) in counterexample.items():
                    logger.debug(f"  {variable}: [{lower}, {upper}]")
            return {"epsilon": epsilon, "result": result, "counterexample": counterexample}

    except FileNotFoundError:
        logger.error(f"File not found at {json_path}.")
        return {"error": "FileNotFound"}
    except json.JSONDecodeError:
        logger.error(f"Unable to parse JSON from the file at {json_path}.")
        return {"error": "JSONDecodeError"}
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return {"error": str(e)}



def dreal_double_integrator_BRS(dreal_partials, dreal_variables, epsilon=0.5, reachMode='forward', reachAim='reach', setType='set', save_directory="./"):
    """
    Verifies if the HJB equation holds using dReal for a double integrator system and saves the results.

    Args:
        dreal_partials (dict): Partial derivatives in dReal format.
        dreal_variables (dict): State variables in dReal format.
        epsilon (float): Tolerance for constraint checking.
        reachMode (str): Reachability mode, either "forward" or "backward".
        reachAim (str): Aim of reachability, either "reach" or "avoid".
        setType (str): Constraint combination type, e.g., "tube".
        save_directory (str): Directory where results will be saved.

    Returns:
        dict: dReal's result if constraints are satisfiable; otherwise None.
    """

    # Extract state and partial derivative variables
    t = dreal_variables["x_1_1"]  # Time variable
    x_1 = dreal_variables["x_1_2"]  # State variable x_1
    x_2 = dreal_variables["x_1_3"]  # State variable x_2

    dv_dt = dreal_partials["partial_x_1_1"]  # Time derivative of V
    p_1 = dreal_partials["partial_x_1_2"]    # Costate for x_1
    p_2 = dreal_partials["partial_x_1_3"]    # Costate for x_2

    # Define Hamiltonian
    hamiltonian = p_1 * x_2

    # Modify the Hamiltonian based on reachAim
    if reachAim == 'avoid':
        hamiltonian -= abs(p_2) # for now we assume that input is [-1,1] Maximize input for avoidance
    elif reachAim == 'reach':
        hamiltonian += abs(p_2) # for now we assume that input is [-1,1] Minimize input for reachability

    # Apply backward reachability logic if specified
    if reachMode == 'backward':
        hamiltonian = -hamiltonian

    # Define constraints
    condition_1 = abs(dv_dt + hamiltonian) <= epsilon # HJB equation
    condition_2 = abs(dv_dt) <= epsilon               # Time derivative constraint

    # Combine constraints based on setType
    if setType=='tube':
        final_condition = Not(And(condition_1, condition_2))
    else:
        final_condition = Not(condition_1)

    # State constraints (bounds for time and states)
    state_constraints = And(
        t >= 0, t <= 1,       # Time bounds
        x_1 >= -1, x_1 <= 1,  # State bounds for x_1
        x_2 >= -1, x_2 <= 1   # State bounds for x_2
    )

    # Combine all constraints
    all_constraints = And(final_condition, state_constraints)

    # Check constraints with tolerance delta
    delta = 1e-5
    result = CheckSatisfiability(all_constraints, delta)

    # Prepare result data
    result_data = {
        "epsilon": epsilon,
        "set": f"{reachMode}_{reachAim}_{setType}", 
        "result": str(result) if result else "HJB Equation Satisfied"
    }

    # Save result to JSON
    result_file = f"{save_directory}/dreal_result.json"
    with open(result_file, "w") as f:
        json.dump(result_data, f, indent=4)

    logger.info(f"Saved result to {result_file}")
    return result_data


class BaseCounterexampleDataset(Dataset):
    """
    Base dataset class for counterexample-based sampling with uniform sampling logic.
    """

    def __init__(self, counterexample, numpoints, percentage_in_counterexample, 
                 percentage_at_t0, tMin=0.0, tMax=1.0, input_max=1.0, 
                 seed=0, collision_radius=0.25, epsilon_radius=0.1):
        """
        Initialize the base dataset.

        Args:
            counterexample (dict): A dictionary with variable ranges.
            numpoints (int): Total number of samples per batch.
            percentage_in_counterexample (float): Percentage of samples from counterexample (10-90).
            percentage_at_t0 (float): Percentage of samples with t = 0 (0-100).
            tMin (float): Minimum time value.
            tMax (float): Maximum time value.
            input_max (float): Maximum control input.
            seed (int): Random seed for reproducibility.
            collision_radius (float): Radius for defining boundary conditions.
            epsilon_radius (float): Amount to expand counterexample range.
        """
        super().__init__()
        if not (10 <= percentage_in_counterexample <= 90):
            raise ValueError("percentage_in_counterexample must be between 10 and 90")
        if not (0 <= percentage_at_t0 <= 100):
            raise ValueError("percentage_at_t0 must be between 0 and 100")

        torch.manual_seed(seed)

        # Extract and expand counterexample ranges
        self.counterexample = {
            variable: (
                max(lower - epsilon_radius, -1),
                min(upper + epsilon_radius, 1)
            )
            for variable, (lower, upper) in counterexample.items()
        }

        # Initialize parameters
        self.numpoints = numpoints
        self.percentage_in_counterexample = percentage_in_counterexample / 100.0
        self.percentage_at_t0 = percentage_at_t0 / 100.0
        self.input_max = input_max
        self.tMin = tMin
        self.tMax = tMax
        self.collision_radius = collision_radius

        # Calculate sample distributions
        self.n_counterexample = int(self.percentage_in_counterexample * numpoints)
        self.n_uniform = numpoints - self.n_counterexample
        self.n_counterexample_t0 = int(self.percentage_at_t0 * self.n_counterexample)
        self.n_counterexample_remaining = self.n_counterexample - self.n_counterexample_t0
        self.n_uniform_t0 = int(self.percentage_at_t0 * self.n_uniform)
        self.n_uniform_remaining = self.n_uniform - self.n_uniform_t0

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        coords = torch.zeros(self.numpoints, len(self.counterexample))

        # Generate samples
        for i, (variable, (lower, upper)) in enumerate(self.counterexample.items()):
            if variable == "x_1_1":  # Time variable
                coords = self._sample_time_dimension(coords, i, lower, upper)
            else:
                coords = self._sample_state_dimension(coords, i, lower, upper)

        # Compute boundary values
        boundary_values = self._compute_boundary_values(coords)
        dirichlet_mask = (coords[:, 0:1] == 0.0)

        return {'coords': coords}, {
            'source_boundary_values': boundary_values,
            'dirichlet_mask': dirichlet_mask
        }

    def _sample_time_dimension(self, coords, dim_idx, lower, upper):
        """Sample the time dimension with the specified distribution."""
        coords[:self.n_counterexample_t0, dim_idx] = 0.0
        coords[self.n_counterexample_t0:self.n_counterexample, dim_idx] = torch.zeros(
            self.n_counterexample_remaining
        ).uniform_(lower, upper)
        coords[self.n_counterexample:self.n_counterexample + self.n_uniform_t0, dim_idx] = 0.0
        coords[self.n_counterexample + self.n_uniform_t0:, dim_idx] = torch.zeros(
            self.n_uniform_remaining
        ).uniform_(self.tMin, self.tMax)
        return coords

    def _sample_state_dimension(self, coords, dim_idx, lower, upper):
        """Sample a state dimension with the specified distribution."""
        coords[:self.n_counterexample, dim_idx] = torch.zeros(
            self.n_counterexample
        ).uniform_(lower, upper)
        coords[self.n_counterexample:, dim_idx] = torch.zeros(
            self.n_uniform
        ).uniform_(-1, 1)
        return coords

    def _compute_boundary_values(self, coords):
        """Compute boundary values for the given coordinates."""
        return torch.norm(coords[:, 1:], dim=1, keepdim=True) - self.collision_radius


class CounterexampleDatasetDoubleIntegrator(BaseCounterexampleDataset):
    """Dataset class for the Double Integrator system."""
    pass  # Inherits all functionality from base class


class CounterexampleDatasetThreeStateSystem(BaseCounterexampleDataset):
    """Dataset class for the 3-state system."""
    pass  # Inherits all functionality from base class


class CounterexampleDatasetTripleIntegrator(BaseCounterexampleDataset):
    """Dataset class for the Triple Integrator system."""
    pass  # Inherits all functionality from base class


def dreal_three_state_system_BRS(dreal_partials, dreal_variables, epsilon=0.5, reachMode='forward', reachAim='reach', setType='set', save_directory="./", k1=1.0, k2=1.0, c1=0.5, c2=0.5, u_max=1.0):
    """
    Verifies if the HJB equation holds using dReal for a 3-state system and saves the results.

    Args:
        dreal_partials (dict): Partial derivatives in dReal format.
        dreal_variables (dict): State variables in dReal format.
        epsilon (float): Tolerance for constraint checking.
        reachMode (str): Reachability mode, either "forward" or "backward".
        reachAim (str): Aim of reachability, either "reach" or "avoid".
        setType (str): Constraint combination type, e.g., "tube".
        save_directory (str): Directory where results will be saved.
        k1, k2 (float): Stiffness constants.
        c1, c2 (float): Coupling coefficients.
        u_max (float): Control bounds.
    """
    # Extract state and partial derivative variables
    t = dreal_variables["x_1_1"]  # Time variable
    x_1 = dreal_variables["x_1_2"]  # State variable x1 (position)
    x_2 = dreal_variables["x_1_3"]  # State variable x2 (velocity)
    x_3 = dreal_variables["x_1_4"]  # State variable x3 (auxiliary)

    dv_dt = dreal_partials["partial_x_1_1"]  # Time derivative of V
    p_1 = dreal_partials["partial_x_1_2"]    # Costate for x1
    p_2 = dreal_partials["partial_x_1_3"]    # Costate for x2
    p_3 = dreal_partials["partial_x_1_4"]    # Costate for x3

    # Define the Hamiltonian
    if reachAim == 'avoid':
        # Maximize control for avoidance
        hamiltonian = (p_1 * x_2
                       - k1 * p_2 * x_1
                       - c1 * p_2 * x_2
                       + abs(p_2) * u_max
                       - k2 * p_3 * x_3
                       + c2 * p_3 * x_1)
    elif reachAim == 'reach':
        # Minimize control for reachability
        hamiltonian = (p_1 * x_2
                       - k1 * p_2 * x_1
                       - c1 * p_2 * x_2
                       - abs(p_2) * u_max
                       - k2 * p_3 * x_3
                       + c2 * p_3 * x_1)

    # Apply backward reachability logic if specified
    if reachMode == 'backward':
        hamiltonian = -hamiltonian

    # Define constraints
    condition_1 = abs(dv_dt + hamiltonian) <= epsilon  # HJB equation
    condition_2 = abs(dv_dt) <= epsilon               # Time derivative constraint

    # Combine constraints based on setType
    if setType == 'tube':
        final_condition = Not(And(condition_1, condition_2))
    else:
        final_condition = Not(condition_1)

    # State constraints (bounds for time and states)
    state_constraints = And(
        t >= 0, t <= 1,       # Time bounds
        x_1 >= -1, x_1 <= 1,  # State bounds for x1
        x_2 >= -1, x_2 <= 1,  # State bounds for x2
        x_3 >= -1, x_3 <= 1   # State bounds for x3
    )

    # Combine all constraints
    all_constraints = And(final_condition, state_constraints)

    # Check constraints with tolerance delta
    delta = 1e-2 # make this large to increase speed (Doesn't increase by much though)
    result = CheckSatisfiability(all_constraints, delta)

    # Prepare result data
    result_data = {
        "epsilon": epsilon,
        "set": f"{reachMode}_{reachAim}_{setType}",
        "result": str(result) if result else "HJB Equation Satisfied"
    }

    # Save result to JSON
    result_file = f"{save_directory}/dreal_result.json"
    with open(result_file, "w") as f:
        json.dump(result_data, f, indent=4)

    logger.info(f"Saved result to {result_file}")
    return result_data


def dreal_triple_integrator_BRS(dreal_partials, dreal_variables, epsilon=0.5, reachMode='forward', reachAim='reach', setType='set', save_directory="./"):
    """
    Verifies if the HJB equation holds using dReal for a triple integrator system and saves the results.

    Args:
        dreal_partials (dict): Partial derivatives in dReal format.
        dreal_variables (dict): State variables in dReal format.
        epsilon (float): Tolerance for constraint checking.
        reachMode (str): Reachability mode, either "forward" or "backward".
        reachAim (str): Aim of reachability, either "reach" or "avoid".
        setType (str): Constraint combination type, e.g., "tube".
        save_directory (str): Directory where results will be saved.

    Returns:
        dict: dReal's result if constraints are satisfiable; otherwise None.
    """
    logger.info("Starting triple integrator verification")

    # Extract state and partial derivative variables
    t = dreal_variables["x_1_1"]  # Time variable
    x_1 = dreal_variables["x_1_2"]  # Position (x1)
    x_2 = dreal_variables["x_1_3"]  # Velocity (x2)
    x_3 = dreal_variables["x_1_4"]  # Acceleration (x3)

    dv_dt = dreal_partials["partial_x_1_1"]  # Time derivative of V
    p_1 = dreal_partials["partial_x_1_2"]    # Costate for x_1 (dV/dx1)
    p_2 = dreal_partials["partial_x_1_3"]    # Costate for x_2 (dV/dx2)
    p_3 = dreal_partials["partial_x_1_4"]    # Costate for x_3 (dV/dx3)

    # Define Hamiltonian
    hamiltonian = p_1 * x_2 + p_2 * x_3  # Contribution from position and velocity

    # Modify the Hamiltonian based on reachAim
    if reachAim == 'avoid':
        hamiltonian -= abs(p_3)  # Maximize control input for avoidance
    elif reachAim == 'reach':
        hamiltonian += abs(p_3)  # Minimize control input for reachability

    # Apply backward reachability logic if specified
    if reachMode == 'backward':
        hamiltonian = -hamiltonian

    # Define constraints
    condition_1 = abs(dv_dt + hamiltonian) <= epsilon  # HJB equation
    condition_2 = abs(dv_dt) <= epsilon               # Time derivative constraint

    # Combine constraints based on setType
    if setType == 'tube':
        final_condition = Not(And(condition_1, condition_2))
    else:
        final_condition = Not(condition_1)

    # State constraints (bounds for time and states)
    state_constraints = And(
        t >= 0, t <= 1,       # Time bounds
        x_1 >= -1, x_1 <= 1,  # State bounds for x_1 (position)
        x_2 >= -1, x_2 <= 1,  # State bounds for x_2 (velocity)
        x_3 >= -1, x_3 <= 1   # State bounds for x_3 (acceleration)
    )

    # Combine all constraints
    all_constraints = And(final_condition, state_constraints)

    # Check constraints with tolerance delta
    delta = 1e-5
    result = CheckSatisfiability(all_constraints, delta)

    # Prepare result data
    result_data = {
        "epsilon": epsilon,
        "set": f"{reachMode}_{reachAim}_{setType}",
        "result": str(result) if result else "HJB Equation Satisfied"
    }

    # Save result to JSON
    result_file = f"{save_directory}/dreal_result.json"
    with open(result_file, "w") as f:
        json.dump(result_data, f, indent=4)

    logger.info(f"Saved result to {result_file}")
    return result_data


