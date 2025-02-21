import torch

class BaseModeConstraint:
    def nextmode(self, mode):
        partial_mode_config, partial_deriv_config = self.config(mode)

        if not mode.islayerfull():
            partial_mode_config = torch.cat((mode.mode_config, partial_mode_config))
            partial_deriv_config = torch.cat((mode.deriv_config, partial_deriv_config))

        return partial_mode_config, partial_deriv_config

    def tohrep(self, mode):
        raise NotImplementedError()

    def config(self, mode):
        raise NotImplementedError()
    
class Relu_LowerModeConstraint(BaseModeConstraint):
    # W * A * x + W * b + b <= 0
    def config(self, mode):
        mode_config = torch.tensor([0], device=mode.A.device)
        deriv_config = torch.tensor([0], device=mode.A.device)

        return mode_config, deriv_config

    def tohrep(self, mode):
        weight = mode.layer.weight.data
        bias = mode.layer.bias.data[:mode.mode_config.size(0)]

        WA = torch.matmul(weight, mode.A)[:mode.mode_config.size(0)]
        Wb = torch.matmul(weight, mode.b)[:mode.mode_config.size(0)]

        mask = (mode.mode_config == 0) & (mode.deriv_config == 0)

        # W * A * x + W * b + b <= 0
        A = -WA[mask]
        b = -Wb - bias[mask]

        # Return as -W * A * x - W * b - b >= 0
        return A, b
    
class Relu_UpperModeConstraint(BaseModeConstraint):
    # W * A * x + W * b + b >= 0
    def config(self, mode):
        mode_config = torch.tensor([1], device=mode.A.device)
        deriv_config = torch.tensor([1], device=mode.A.device)

        return mode_config, deriv_config

    def tohrep(self, mode):
        weight = mode.layer.weight.data
        bias = mode.layer.bias.data[:mode.mode_config.size(0)]

        WA = torch.matmul(weight, mode.A)[:mode.mode_config.size(0)]
        Wb = torch.matmul(weight, mode.b)[:mode.mode_config.size(0)]

        mask = (mode.mode_config == 0) & (mode.deriv_config == 0)

        # W * A * x + W * b + b >= 0
        A = WA[mask]
        b = Wb + bias[mask]

        return A, b