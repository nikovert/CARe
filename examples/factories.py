import logging
import os
import importlib
from typing import Dict, Type
from pathlib import Path

logger = logging.getLogger(__name__)

# Dictionary to store example classes
EXAMPLE_REGISTRY: Dict[str, Type] = {}

def register_example(cls):
    """Decorator to register example classes"""
    EXAMPLE_REGISTRY[cls.Name] = cls
    return cls

def create_example(name, args):
    """Factory function to create examples"""
    if name not in EXAMPLE_REGISTRY:
        raise ValueError(f"Unknown example: {name}")
    return EXAMPLE_REGISTRY[name](args)

def get_example_names():
    """Get list of registered example names"""
    return list(EXAMPLE_REGISTRY.keys())

def discover_examples():
    """Automatically discover and import example systems"""
    systems_dir = Path(__file__).parent / 'systems'
    if not systems_dir.exists():
        logger.warning(f"Systems directory not found at {systems_dir}")
        return

    # Iterate through all subdirectories in systems
    for system_dir in systems_dir.iterdir():
        if not system_dir.is_dir():
            continue

        # Look for a Python file with the same name as the directory
        main_file = system_dir / f"{system_dir.name}.py"
        if main_file.exists():
            # Convert path to module path
            relative_path = main_file.relative_to(Path(__file__).parent.parent)
            module_path = str(relative_path).replace('/', '.').replace('.py', '')
            
            try:
                # Import the module to trigger the register_example decorator
                importlib.import_module(module_path)
                logger.debug(f"Successfully loaded example system from {module_path}")
            except Exception as e:
                logger.warning(f"Failed to load example system from {module_path}: {e}")

# Discover examples when the module is imported
discover_examples()

# Update EXAMPLE_NAMES after discovery
EXAMPLE_NAMES = get_example_names()