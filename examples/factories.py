from examples.systems.double_integrator.double_integrator import DoubleIntegrator

def create_example(name, args):
    """Factory function to create examples"""
    if name == DoubleIntegrator.Name:
        return DoubleIntegrator(args)
    else:
        raise ValueError(f"Unknown example: {name}")

EXAMPLE_NAMES = [DoubleIntegrator.Name]