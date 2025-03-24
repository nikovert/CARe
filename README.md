# Certified Approximate Reachability (CARe): Formal Error Bounds on Deep Learning of Reachable Sets

This repository contains the code for "Certified Approximate Reachability (CARe): Formal Error Bounds on Deep Learning of Reachable Sets."

## Overview
Recent approaches to leveraging deep learning for computing reachable sets of continuous-time dynamical systems have gained popularity over traditional level-set methods, as they overcome the curse of dimensionality. However, as with level-set methods, considerable care needs to be taken in limiting approximation errors, particularly since no guarantees are provided during training on the accuracy of the learned reachable set. To address this limitation, we introduce an epsilon-approximate Hamilton-Jacobi Partial Differential Equation (HJ-PDE), which establishes a relationship between training loss and accuracy of the \textit{true} reachable set. To formally certify this approximation, we leverage Satisfiability Modulo Theories (SMT) solvers to bound the residual error of the HJ-based loss function across the domain of interest. Leveraging Counter Example Guided Inductive Synthesis (CEGIS), we close the loop around learning and verification, by fine-tuning the neural network on counterexamples found by the SMT solver, thus improving the accuracy of the learned reachable set. To the best of our knowledge, Certified Approximate Reachability (CARe) is the first approach to provide soundness guarantees on learned reachable sets of continuous dynamical systems.
## Installing dReal

This package requires the dReal SMT solver. To ensure a consistent environment, we recommend using Docker. 

### Using Docker
If you have Docker installed within VS Code, you can simply create a new development container from the Dockerfile, which will set up a reproducible environment with all dependencies installed, including dReal.

## Installation (without Docker)

If you prefer to install the package directly on your system, run:

```bash
pip install -e .
```

This will install the package and its dependencies while allowing you to modify the source code.
