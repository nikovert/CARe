import queue
import torch
from torch import nn
import cdd
import time
import logging

logger = logging.getLogger(__name__)
from .mode_verifier import Z3ModeVerifier
from .constraint_rules import Relu_LowerModeConstraint, Relu_UpperModeConstraint


class TreeSearchVerifier:
    NAME = "TreeSearchVerifier"

    constraint_rules = [
        Relu_LowerModeConstraint(),
        Relu_UpperModeConstraint()
    ]

    def __init__(self, mode_verification_strategy=Z3ModeVerifier()):
        self.found_modes = 0  # Initialize found_modes
        self.mode_verification_strategy = mode_verification_strategy

    def verify_system(self, conditions, alpha):
        logger.info(f"Starting tree search verification with mode verifier {type(self.mode_verification_strategy).__name__}...")
        modes = ConfigIterator(self.constraint_rules, conditions.network, conditions.system.state_domain)
        start = time.time()
        for mode in modes:
            success, counterexample = self.mode_verification_strategy.verify_mode(
                conditions, alpha, self.constraint_rules, mode)
            if not success:
                logger.debug(f"Total modes found: {modes.num_modes}, pruned branches: {modes.num_pruned}, generated branches: {modes.num_generated}, in {time.time() - start}s")
                logger.info("Tree search verification completed. Result: counterexample found.")
                return success, counterexample

            if modes.num_modes % 1000 == 0:
                logger.debug(f"{modes.num_modes}/{modes.num_pruned}: {time.time() - start}")

        self.found_modes = modes.num_modes  # Set found_modes
        logger.debug(f"Total modes found: {modes.num_modes}, pruned branches: {modes.num_pruned}, generated branches: {modes.num_generated}, in {time.time() - start}s")
        logger.info("Tree search verification completed. Result: verified.")
        success = True
        counterexample = None
        return success, counterexample


class PartialConfiguration:
    def __init__(self, parent, layer: nn.Linear, mode_config: torch.Tensor, deriv_config: torch.Tensor, A, b):
        # The parent configuration in the tree (None if this is the root)
        self.parent = parent
        # The current layer of the neural network being configured
        self.layer = layer
        self.mode_config = mode_config
        self.deriv_config = deriv_config
        self.A, self.b = A, b

    def num_parents(self):
        if self.parent is None:
            return 0
        else:
            return 1 + self.parent.num_parents()

    def num_nodes(self):
        return self.layer.out_features

    def isfull(self, num_layers):
        return num_layers == self.num_parents() + 1 and self.islayerfull()

    @staticmethod
    def empty(layer: nn.Linear, offset):
        # Create an empty configuration for the given layer
        empty_config = torch.zeros((0,), dtype=torch.int32, device=layer.weight.device)
        empty_A = torch.eye(layer.in_features, device=layer.weight.device)
        empty_b = torch.zeros((layer.in_features, 1), device=layer.weight.device)
        return PartialConfiguration(None, layer, empty_config, empty_config, empty_A, empty_b, empty_b, offset)

    def isempty(self):
        return self.mode_config.size(0) == 0

    def islayerfull(self):
        # Check if the current layer is fully configured
        return self.mode_config.size(0) == self.num_nodes()

    def nextlayer(self):
        return self.num_parents() + 1

    def finalize_layer(self, network):
        A = self.mode_config.view(-1, 1) * torch.matmul(self.layer.weight.data, self.A)

        eps_l = torch.matmul(self.layer.weight.data.abs(), (self.b_upper - self.b_lower))

        Wb_lower = torch.matmul(self.layer.weight.data.clamp(min=0), self.b_lower) + \
            torch.matmul(self.layer.weight.data.clamp(max=0), self.b_upper)
        Wb_upper = torch.matmul(self.layer.weight.data.clamp(min=0), self.b_upper) + \
            torch.matmul(self.layer.weight.data.clamp(max=0), self.b_lower)

        b_lower = self.mode_config.view(-1, 1) * (Wb_lower + self.layer.bias.data.view(-1, 1)) - \
            network.epsilon()
        b_upper = self.mode_config.view(-1, 1) * (Wb_upper + self.layer.bias.data.view(-1, 1)) + \
            self.deriv_config.view(-1, 1) * eps_l

        return A, b_lower, b_upper

    def constraints(self, constraint_rules, domain):
        # H-rep: 0 <= A * x  + b
        domain_hrep = domain.tosimplehrep()

        hreps = [domain_hrep]

        partial_config = self
        while partial_config is not None:
            for constraint_rule in constraint_rules:
                A, b = constraint_rule.tohrep(partial_config)
                if b.size(-1) != 0:
                    hreps.append((A, b))

            partial_config = partial_config.parent

        # Stack constraints
        A = torch.cat([A for A, _ in hreps], dim=0)
        b = torch.cat([b for _, b in hreps], dim=0)

        return A, b

    def isintersectionempty(self, constraint_rules, domain):
        A, b = self.constraints(constraint_rules, domain)

        # To LP format
        matrix_data = torch.cat([
            torch.cat((b.view(-1, 1), A), dim=1),
            torch.zeros((1, A.shape[1] + 1))    # Objective function
        ], dim=0)
        lp = cdd.linprog_from_array(matrix_data.cpu().numpy(), obj_type=cdd.LPObjType.MAX)
        cdd.linprog_solve(lp)

        # If the LP is infeasible, then the intersection is empty
        return lp.status != cdd.LPStatusType.OPTIMAL


class FullConfiguration:
    def __init__(self, parent, A, b_lower, b_upper):
        self.parent = parent
        self.A, self.b_lower, self.b_upper = A, b_lower, b_upper

    def constraints(self, constraint_rules, domain):
        return self.parent.constraints(constraint_rules, domain)

    @property
    def mode_config(self):
        mode_config = []

        parent = self.parent
        while parent is not None:
            mode_config.insert(0, parent.mode_config)
            parent = parent.parent

        return mode_config

    @property
    def deriv_config(self):
        deriv_config = []

        parent = self.parent
        while parent is not None:
            deriv_config.insert(0, parent.deriv_config)
            parent = parent.parent

        return deriv_config


class ConfigIterator:
    def __init__(self, constraint_rules, network, domain):
        self.constraint_rules = constraint_rules
        self.network = network

        # Ignore output layer, since it does not have an activation function
        self.linear_layers = [layer for layer in self.network if isinstance(layer, nn.Linear)][:-1]
        self.final_layer = self.network[-1]

        self.domain = domain

        # It is more efficient memory-wise to use a stack instead of a queue,
        # since we will traverse the tree in a depth-first manner and discarding
        # the nodes as we go along.
        self.stack = queue.LifoQueue()

        self.num_modes = 0
        self.num_generated = {type(rule).__name__: 0 for rule in self.constraint_rules}
        self.num_pruned = {type(rule).__name__: 0 for rule in self.constraint_rules}

    def num_layers(self):
        return len(self.linear_layers)

    def __iter__(self):
        self.stack.empty()
        offset = self.network.offset()

        config = PartialConfiguration.empty(self.linear_layers[0], offset)
        self.stack.put(config)

        self.num_modes = 0
        self.num_generated = {type(rule).__name__: 0 for rule in self.constraint_rules}
        self.num_pruned = {type(rule).__name__: 0 for rule in self.constraint_rules}

        return self

    def __next__(self):
        if self.stack.empty():
            raise StopIteration

        partial_config = self.stack.get()
        while not partial_config.isfull(self.num_layers()):  # Is branch, not leaf
            # Generate new configurations based on the current partial configuration
            for config in self.generate_next_partial_configurations(partial_config):
                self.stack.put(config)

            # Get the next partial configuration from the stack until we find a leaf
            partial_config = self.stack.get()

        self.num_modes += 1
        config = self.finalize_config(partial_config)

        return config

    def generate_next_partial_configurations(self, partial_config):
        if partial_config.islayerfull():
            A, b_lower, b_upper = partial_config.finalize_layer(self.network)

            layer = self.linear_layers[partial_config.nextlayer()]
            parent = partial_config
        else:
            A, b_lower, b_upper = partial_config.A, partial_config.b_lower, partial_config.b_upper

            layer = partial_config.layer
            parent = partial_config.parent

        for constraint_rule in self.constraint_rules:
            partial_mode_config, partial_deriv_config = constraint_rule.nextmode(partial_config)

            config = PartialConfiguration(parent, layer, partial_mode_config, partial_deriv_config,
                                          A, b_lower, b_upper, partial_config.offset)
            if config.isintersectionempty(self.constraint_rules, self.domain):
                self.num_pruned[type(constraint_rule).__name__] += 1
            else:
                self.num_generated[type(constraint_rule).__name__] += 1
                yield config

    def finalize_config(self, partial_config):
        # Penultimate layer
        A, b_lower, b_upper = partial_config.finalize_layer(self.network)

        # Last layer
        layer = self.final_layer

        new_A = torch.matmul(layer.weight.data, A)
        new_b_lower = torch.matmul(layer.weight.data.clamp(min=0), b_lower) + \
            torch.matmul(layer.weight.data.clamp(max=0), b_upper) + layer.bias.data.view(-1, 1)
        new_b_upper = torch.matmul(layer.weight.data.clamp(min=0), b_upper) + \
            torch.matmul(layer.weight.data.clamp(max=0), b_lower) + layer.bias.data.view(-1, 1)

        return FullConfiguration(partial_config, new_A, new_b_lower.view(-1), new_b_upper.view(-1))
