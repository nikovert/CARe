import numpy as np
import cvxpy as cp
import z3

class BaseModeVerifier:
    def verify_mode(self, conditions: NeuralStochasticCBFConditions, alpha, constraint_rules, mode):
        unsafe_mode, initial_point = self.check_unsafe_mode(conditions, constraint_rules, mode)
        if unsafe_mode:
            # If X_c \cap {x | N(x) <= 0} is empty, then no x in the mode region is safe.
            # Hence, the condition is trivially satisfied and we do not need to check it.
            return True, None

        success, counterexample = self.verify_subset(conditions, constraint_rules, mode, initial_point)
        # if not success:
        return success, counterexample



class Z3ModeVerifier(BaseModeVerifier):

    def verify_subset(self, conditions: NeuralStochasticCBFConditions, constraint_rules, mode, initial_point):
        """
        Verify that for all `x` in `{x | N(x) <= 0}`, we have `s(x) >= 0`.
        :param conditions: A class to represent the necessary components to specify the conditions on a Neural Stochastic CBF.
        :param constraint_rules: A list of constraint rules for the modes (to recover a H-rep of `\\mathcal{X}_c`).
        :param mode: The mode to verify.
        :param initial_point: The initial point to start the optimization from.
        :return: Boolean indicating whether the condition is satisfied, and a counterexample if it is not satisfied.

        The strategy is to encode a constraint of the form `N(x) <= A_c * x + \\overline{b}_c <= 0` for mode `c` and let the minimization objective be `s(x)`.
        Then if `s(x) <= 0` for some x under the constraints of the optimization problem, we have found a violation of `N(x) <= 0 \implies s(x) >= 0`.
        """

        A, b_lower, b_upper = mode.A, mode.b_lower, mode.b_upper
        H, h = mode.constraints(constraint_rules, conditions.system.state_domain)

        solver = z3.Solver()

        x = [z3.Real(f"x{i}") for i in range(conditions.system.state_dim)]

        # Set the state domain as the bounds
        x_lower, x_upper = self.bounding_box(conditions.system.state_domain, H, h)
        solver.add(z3.And([x[i] >= x_lower[i] for i in range(conditions.system.state_dim)]))
        solver.add(z3.And([x[i] <= x_upper[i] for i in range(conditions.system.state_dim)]))

        # Add constraints for the mode and network negativity
        self.network_negativity_constraint(solver, x, A, b_lower, b_upper)
        self.mode_constraints(solver, x, H, h)

        # Minimize the signed distance to the unsafe region
        sx = conditions.system.safe_domain_z3(x)
        solver.add(sx <= 0)

        status = solver.check()
        if status == z3.sat:
            x = self.optimize_subset_counterexample(conditions, H, h, A, b_lower, b_upper)
            return False, CounterExample.unsafe(x)
        elif status == z3.unsat:
            return True, None
        else:
            raise ValueError("Z3 solver failed")

    def network_negativity_constraint(self, solver, x, A, b_lower, b_upper):
        """
        Add the constraint that the network is negative, i.e `N(x) <= A_c * x + \\overline{b}_c <= 0` for mode `c`.
        """
        s = b_lower[0].item()
        for j in range(A.size(1)):
            s = s + A[0, j].item() * x[j]
        solver.add(s <= 0)

        return s

    def mode_constraints(self, solver, x, H, h):
        """
        Add constraints for the set `\\mathcal{X}_c` wrt. mode `c`.
        """
        for i in range(H.size(0)):
            s = h[i].item()
            for j in range(H.size(1)):
                s += H[i, j].item() * x[j]
            solver.add(s >= 0)

    def bounding_box(self, state_domain, H, h):
        # P = {x : H * x + h >= 0}

        # Compute the bounding box of the polytope using
        # the support function of the polytope along each axis
        # Use cvxpy to solve the LP problem

        x_lower = np.zeros(H.size(1))
        x_upper = np.zeros(H.size(1))

        x = cp.Variable(H.size(1))
        constraints = [H.numpy() @ x + h.numpy() >= 0]

        for i in range(H.size(1)):
            objective = cp.Minimize(x[i])
            prob = cp.Problem(objective, constraints)
            prob.solve()

            x_lower[i] = x.value[i]

            objective = cp.Maximize(x[i])
            prob = cp.Problem(objective, constraints)
            prob.solve()

            x_upper[i] = x.value[i]

        x_lower = np.maximum(x_lower, state_domain.lower.numpy())
        x_upper = np.minimum(x_upper, state_domain.upper.numpy())

        return x_lower, x_upper

    def optimize_subset_counterexample(self, conditions: NeuralStochasticCBFConditions, H, h, A, b_lower, b_upper):
        opt = z3.Optimize()

        x = [z3.Real(f"x{i}") for i in range(conditions.system.state_dim)]

        # Set the state domain as the bounds
        x_lower, x_upper = self.bounding_box(conditions.system.state_domain, H, h)
        opt.add(z3.And([x[i] >= x_lower[i] for i in range(conditions.system.state_dim)]))
        opt.add(z3.And([x[i] <= x_upper[i] for i in range(conditions.system.state_dim)]))

        # Add constraints for the mode and network negativity
        nx = self.network_negativity_constraint(opt, x, A, b_lower, b_upper)
        self.mode_constraints(opt, x, H, h)

        # Minimize the signed distance to the unsafe region
        sx = conditions.system.safe_domain_z3(x)
        opt.add(sx <= 0)

        opt.minimize(nx)

        status = opt.check()
        if status != z3.sat:
            raise ValueError(f"Optimization failed with status {status}")

        model = opt.model()

        def to_float(x, precision=10):
            try:
                return float(x.as_fraction())
            except AttributeError:
                return float(x.approx(precision).as_fraction())

        x = np.array([to_float(model[x[i]]) for i in range(conditions.system.state_dim)])

        return x

    def verify_derivatives(self, conditions: NeuralStochasticCBFConditions, alpha, constraint_rules, mode, initial_point):
        raise NotImplementedError("NLoptModeVerifier.verify_derivatives not implemented")
