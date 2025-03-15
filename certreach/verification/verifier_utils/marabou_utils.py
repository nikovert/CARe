import logging
from typing import Dict, Tuple, Optional
from maraboupy import MarabouCore
import re

logger = logging.getLogger(__name__)

class MarabouExpressionParser:
    """Parse symbolic expressions into Marabou input queries with ReLU support."""
    
    large_constant = 1e6

    def __init__(self):
        """Initialize the expression parser."""
        self.variables = {}  # Maps var names to Marabou variable indices
        self.next_var_idx = 0
        self.equations = []  # Stores Marabou equations
        self.relu_pairs = []  # Stores pairs of variables for ReLU constraints
        self.abs_pairs = []  # Stores pairs of variables for Abs constraints
    
    def get_variable(self, var_name):
        """Get existing or create new Marabou variable index."""
        if var_name not in self.variables:
            self.variables[var_name] = self.next_var_idx
            self.next_var_idx += 1
        return self.variables[var_name]
    
    def create_auxiliary_variable(self):
        """Create a new auxiliary variable."""
        var_idx = self.next_var_idx
        self.next_var_idx += 1
        return var_idx
    
    def parse_expression(self, expr_str):
        """
        Parse a symbolic expression into Marabou constraints.
        Handles expressions with ReLU/Max/Abs functions.
        
        Args:
            expr_str: Expression string like "0.5*max(0, x_1_1) + 0.3*abs(x_1_2)"
            
        Returns:
            An output variable index representing this expression
        """        
        if expr_str.startswith('(') and expr_str.endswith(')'):
            expr_str = expr_str[1:-1]

        # Tokenize and parse the expression
        if "max(" in expr_str or "abs(" in expr_str:
            # Handle expressions with Max/ReLU/Abs
            return self._parse_expression_with_nonlinear(expr_str)
        else:
            # Handle linear expressions
            return self._parse_linear_expression(expr_str)
    
    def _parse_expression_with_nonlinear(self, expr_str):
        """Parse expression containing ReLU/Max/Abs terms."""
        # This handles expressions with nonlinear functions (max, abs)
        
        output_var = self.create_auxiliary_variable()
        
        # Split by addition/subtraction operations, respecting parentheses
        terms = []
        current_term = ""
        paren_count = 0

        for char in expr_str:
            if char in ('+', '-') and paren_count == 0:
                if current_term:
                    terms.append(current_term)
                    current_term = ""
                if char == '-':
                    current_term = '-'
                # + is just a separator, don't add to current_term
            else:
                current_term += char
                if char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1
        
        if current_term:
            terms.append(current_term)
        
        # Process each term
        output_terms = []
        for term in terms:
            term = term.strip()
            if "max(" in term:
                # Handle Max/ReLU term
                max_var = self._parse_max_term(term)
                output_terms.append(max_var)
            elif "abs(" in term:
                # Handle Abs term
                abs_var = self._parse_abs_term(term)
                output_terms.append(abs_var)
            else:
                # Handle linear term
                linear_var = self._parse_linear_term(term)
                output_terms.append(linear_var)
        
        # Create equation connecting output_var to sum of all terms
        output_eq = MarabouCore.Equation(MarabouCore.Equation.EQ)
        
        # Output variable coefficient is -1
        output_eq.addAddend(-1, output_var)
        
        # Add each term with coefficient 1
        for term_var in output_terms:
            if term_var is not None:
                output_eq.addAddend(1, term_var)
        
        output_eq.setScalar(0)  # Set right side to 0
        self.equations.append(output_eq)
        
        return output_var
    
    def _parse_max_term(self, term):
        """Parse a term containing max(0, ...)."""
        # Extract coefficient and Max expression
        if '*' in term and "max(" in term.split('*')[1]:
            # Term is in the form "coef*max(...)"
            coef_str, max_expr = term.split('*', 1)
            coef_str = coef_str.replace(" ", "")
            coef = float(coef_str.strip())
        elif term.strip().startswith("max("):
            # Term is just "max(...)" with implicit coefficient 1
            max_expr = term
            coef = 1.0
        elif term.strip().startswith("-max("):
            # Term is "-max(...)" with implicit coefficient -1
            max_expr = term[1:].strip()
            coef = -1.0
        else:
            raise ValueError(f"Cannot parse Max term: {term}")
        
        # Extract the inner linear expression
        inner_expr = max_expr.strip()[4:-1]  # Remove "max(" and ")"
        if inner_expr.startswith("0, "):
            # This is a ReLU: max(0, expr)
            inner_expr = inner_expr[3:]  # Remove "0, "
            input_var = self.parse_expression(inner_expr)
            
            # Create output variable for the ReLU
            output_var = self.create_auxiliary_variable()
            
            # Add ReLU relation
            self.relu_pairs.append((input_var, output_var))
            
            # If coefficient is not 1, we need another variable and equation
            if coef != 1.0:
                final_var = self.create_auxiliary_variable()
                eq = MarabouCore.Equation(MarabouCore.Equation.EQ)
                eq.addAddend(-1, final_var)
                eq.addAddend(coef, output_var)
                eq.setScalar(0)
                self.equations.append(eq)
                return final_var
            else:
                return output_var
        else:
            raise ValueError(f"Expected ReLU max(0, expr), got: {max_expr}")
    
    def _parse_abs_term(self, term):
        """Parse a term containing abs(...)."""
        # Extract coefficient and Abs expression
        if '*' in term and "abs(" in term.split('*')[1]:
            # Term is in the form "coef*abs(...)"
            coef_str, abs_expr = term.split('*', 1)
            coef_str = coef_str.replace(" ", "")
            coef = float(coef_str.strip())
        elif term.strip().startswith("abs("):
            # Term is just "abs(...)" with implicit coefficient 1
            abs_expr = term
            coef = 1.0
        elif term.strip().startswith("-abs("):
            # Term is "-abs(...)" with implicit coefficient -1
            abs_expr = term[1:].strip()
            coef = -1.0
        else:
            raise ValueError(f"Cannot parse Abs term: {term}")
        
        # Extract the inner linear expression
        inner_expr = abs_expr.strip()[4:-1]  # Remove "abs(" and ")"
        input_var = self.parse_expression(inner_expr)
        
        # Create output variable for the Abs
        output_var = self.create_auxiliary_variable()
        
        # Add Abs relation
        self.abs_pairs.append((input_var, output_var))
        
        # If coefficient is not 1, we need another variable and equation
        if coef != 1.0:
            final_var = self.create_auxiliary_variable()
            eq = MarabouCore.Equation(MarabouCore.Equation.EQ)
            eq.addAddend(-1, final_var)
            eq.addAddend(coef, output_var)
            eq.setScalar(0)
            self.equations.append(eq)
            return final_var
        else:
            return output_var
    
    def _parse_linear_term(self, term):
        """Parse a linear term without Max/Abs functions."""
        if not term:
            return None
        
        # Handle parentheses removal for expressions like "- (x_1_3 * partial_x_1_2)"
        term = term.strip()
        negative_multiplier = 1.0
        if term.startswith('-'):
            term = term[1:].strip()
            negative_multiplier = -1.0
        
        # Remove outer parentheses if present
        if term.startswith('(') and term.endswith(')'):
            term = term[1:-1].strip()
        
        # Create a variable to represent this term
        term_var = self.create_auxiliary_variable()
        
        # Create equation: term_var = linear combination
        eq = MarabouCore.Equation(MarabouCore.Equation.EQ)
        eq.addAddend(-1, term_var)
        
        # Handle multiplication of variables: x_1_3 * partial_x_1_2
        if '*' in term:
            parts = term.split('*')
            if len(parts) == 2:  # Simple form: coef * var or var1 * var2
                part1 = parts[0].strip()
                part2 = parts[1].strip()
                
                try:
                    # Check if part1 is a numeric coefficient
                    coef = float(part1)
                    var_idx = self.get_variable(part2)
                    eq.addAddend(coef * negative_multiplier, var_idx)
                except ValueError:
                    # Both parts are variables - need auxiliary variables and constraints
                    var1_idx = self.get_variable(part1)
                    var2_idx = self.get_variable(part2)
                    
                    # Create a new equation representing the product
                    # We're approximating: term_var = var1 * var2
                    # For now, we'll represent this as a linear equation
                    # If exact multiplication is needed, further constraints would be required
                    
                    # Simple approach: create two constraints to bound the product
                    # This approximation only works if we know the range of both variables
                    # For exact handling, a proper multiplication encoding would be needed
                    logger.warning(f"Approximating multiplication of variables: {part1} * {part2}")
                    
                    # For simplicity, we'll just add both variables with equal weight
                    # This is a crude approximation that should be improved
                    eq.addAddend(0.5 * negative_multiplier, var1_idx)
                    eq.addAddend(0.5 * negative_multiplier, var2_idx)
            else:
                # More complex expression with multiple multiplications
                logger.warning(f"Complex multiplication expression detected: {term} - approximating")
                # Process multiple multiplications with a simplified approach
                # A better solution would encode the constraints more precisely
                for part in parts:
                    try:
                        coef = float(part.strip())
                        # It's a coefficient, store it
                        eq.setScalar(eq.getScalar() + coef * negative_multiplier)
                    except ValueError:
                        # It's a variable
                        var_idx = self.get_variable(part.strip())
                        eq.addAddend(negative_multiplier, var_idx)
        else:
            # Term is just a variable or constant
            try:
                # Try parsing as a constant
                constant = float(term)
                eq.setScalar(constant * negative_multiplier)
            except ValueError:
                # It's a variable with coefficient 1
                var_idx = self.get_variable(term)
                eq.addAddend(negative_multiplier, var_idx)
        
        self.equations.append(eq)
        return term_var
    
    def _parse_linear_expression(self, expr_str):
        """
        Parse a linear expression (no Max functions).
        
        """
        # Create variable for the result
        result_var = self.create_auxiliary_variable()
        
        # Create equation: result_var = linear combination
        eq = MarabouCore.Equation(MarabouCore.Equation.EQ)
        eq.addAddend(-1, result_var)
        
        # Use regex to extract terms properly
        # Match patterns like: "-0.123", "0.456 * x_1_2", "-7.89 * x_1_3"
        pattern = r'([-+]?\s*\d*\.?\d+(?:[eE][-+]?\d+)?)(?:\s*\*\s*([a-zA-Z_][a-zA-Z0-9_]*))?'
        
        # Find all matches
        matches = re.findall(pattern, expr_str)
        constant_term = 0
        
        # Process each match
        for coef_str, var_name in matches:
            # Remove all whitespace from the coefficient string
            coef_str = coef_str.replace(" ", "")
            coeff = float(coef_str)
            
            if var_name:
                # Term with variable
                var_idx = self.get_variable(var_name.strip())
                eq.addAddend(coeff, var_idx)
            else:
                # Constant term
                constant_term += coeff
        
        # Add constant term to equation
        eq.setScalar(constant_term)
        self.equations.append(eq)
        
        return result_var
    
    def create_marabou_query(self, bounds=None):
        """
        Create a Marabou input query from the parsed expressions.
        
        Args:
            bounds: Dictionary mapping variable names to (lower, upper) bounds
            
        Returns:
            MarabouCore.InputQuery: The input query for Marabou
        """
        query = MarabouCore.InputQuery()
        query.setNumberOfVariables(self.next_var_idx-1)

        # Add all equations
        for eq in self.equations:
            query.addEquation(eq)
        
        for var_idx in range(self.next_var_idx-1):
            query.setLowerBound(var_idx, -self.large_constant)
            query.setUpperBound(var_idx, self.large_constant)

        # Set variable bounds
        if bounds:
            for var_name, (lb, ub) in bounds.items():
                if var_name in self.variables:
                    var_idx = self.variables[var_name]
                    query.setLowerBound(var_idx, lb)
                    query.setUpperBound(var_idx, ub)

        # Add all ReLU constraints
        for in_var, out_var in self.relu_pairs:
            MarabouCore.addReluConstraint(query, in_var, out_var)
            
        # Add all Abs constraints
        for in_var, out_var in self.abs_pairs:
            MarabouCore.addAbsConstraint(query, in_var, out_var)
        
        return query

def check_with_marabou(
    constraint_data, 
    partials_expr,
    hamiltonian_expr
) -> Tuple[int, Optional[Dict]]:
    """
    Process a constraint check using Marabou with full ReLU support.
    
    Args:
        constraint_data: Dictionary describing the constraint to check
        partial_derivs: Dictionary of partial derivatives
        hamiltonian_expr: Symbolic expression of the Hamiltonian
    
    Returns:
        Tuple of (constraint_id, counterexample if found or None)
    """
    constraint_id = constraint_data['constraint_id']
    constraint_type = constraint_data['constraint_type']
    epsilon = constraint_data['epsilon']
    is_initial_time = constraint_data['is_initial_time']
    
    # Create parser to handle expressions with ReLU
    parser = MarabouExpressionParser()
    
    # Setup variable bounds
    bounds = {}
    if is_initial_time:
        bounds["x_1_1"] = (0.0, 0.0)  # Fix time at t=0
    else:
        bounds["x_1_1"] = (0.0, 1.0)  # Time variable t
        
    # Add state variable bounds
    space_constraints = constraint_data['space_constraints']
    for i, (lb, ub) in enumerate(space_constraints):
        bounds[f"x_1_{i+2}"] = (lb, ub)
    
    # Parse the value function, partial derivatives, and boundary expressions
    dv_dt_var = None
    if f"partial_x_1_1" in partials_expr:
        dv_dt_var = parser.parse_expression(partials_expr[f"partial_x_1_1"])
    
    hamiltonian_var = parser.parse_expression(hamiltonian_expr)
    
    if constraint_type in ['boundary_1', 'boundary_2', 'target_1', 'target_3']:
        # Value function - boundary value > epsilon
        logger.warning(f"{constraint_type} constraint not recommended for Marabou")
    elif constraint_type == 'derivative_1':
        # dV/dt + H(x, ∇V) < -epsilon
        constraint_eq = MarabouCore.Equation(MarabouCore.Equation.LE)
        constraint_eq.addAddend(1, dv_dt_var)
        constraint_eq.addAddend(1, hamiltonian_var)
        constraint_eq.setScalar(-epsilon)
        
    elif constraint_type == 'derivative_2':
        # dV/dt + H(x, ∇V) > epsilon
        constraint_eq = MarabouCore.Equation(MarabouCore.Equation.GE)
        constraint_eq.addAddend(1, dv_dt_var)
        constraint_eq.addAddend(1, hamiltonian_var)
        constraint_eq.setScalar(epsilon)
        
    elif constraint_type == 'target_2':
        # dV/dt + H(x, ∇V) > epsilon
        constraint_eq = MarabouCore.Equation(MarabouCore.Equation.GE)
        constraint_eq.addAddend(1, dv_dt_var)
        constraint_eq.addAddend(1, hamiltonian_var)
        constraint_eq.setScalar(epsilon)
        
    query = parser.create_marabou_query(bounds)
    query.addEquation(constraint_eq)
    
    options = MarabouCore.Options()
    options._verbosity = 0
    
    # Check constraint
    result = MarabouCore.solve(query, options)
    
    if result[0] == 'unsat':
        return None 
    else:
        # Found a counterexample
        counterexample = {}
        for var_name, var_idx in parser.variables.items():
            if var_name.startswith('x_1_'):
                counterexample[var_name] = result[1][var_idx]
        
        return counterexample
