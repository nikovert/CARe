# Certified Approximate Reachability (CARe): Formal Error Bounds on Deep Learning of Reachable Sets

This repository contains the code for "Certified Approximate Reachability (CARe): Formal Error Bounds on Deep Learning of Reachable Sets."
[Check out the paper](https://www.arxiv.org/abs/2503.23912)

## Overview
Recent approaches to leveraging deep learning for computing reachable sets of continuous-time dynamical systems have gained popularity over traditional level-set methods, as they overcome the curse of dimensionality. However, as with level-set methods, considerable care needs to be taken in limiting approximation errors, particularly since no guarantees are provided during training on the accuracy of the learned reachable set. To address this limitation, we introduce an epsilon-approximate Hamilton-Jacobi Partial Differential Equation (HJ-PDE), which establishes a relationship between training loss and accuracy of the \textit{true} reachable set. To formally certify this approximation, we leverage Satisfiability Modulo Theories (SMT) solvers to bound the residual error of the HJ-based loss function across the domain of interest. Leveraging Counter Example Guided Inductive Synthesis (CEGIS), we close the loop around learning and verification, by fine-tuning the neural network on counterexamples found by the SMT solver, thus improving the accuracy of the learned reachable set. To the best of our knowledge, Certified Approximate Reachability (CARe) is the first approach to provide soundness guarantees on learned reachable sets of continuous dynamical systems.
## Installing dReal

This package requires the dReal SMT solver. To ensure a consistent environment, we recommend using Docker. 

## Setting Up the Development Container in VS Code 

If you have Docker installed in VS Code, you can easily create a development container from the provided Dockerfile. This will set up a reproducible environment with all necessary dependencies, including dReal.  

### Steps to Set Up the Development Container  

1. **Open the Repository** – Ensure the repository is open in VS Code.  
2. **Access the Docker Plugin** – In the left sidebar, navigate to the Docker extension.  
3. **Create a New Container** – Click the **plus (+) icon** next to "Containers" and select **"Open Current Folder in Container"**.  
4. **Wait for Setup** – In the bottom right corner of VS Code, you'll see **"Opening Remote..."** This indicates that a new Docker container is being built in the background, which may take a few minutes.  
5. **You're Ready!** – Once the process is complete, your development environment will be fully set up with CARe installed and ready to use.  

## Installation (without Docker)

If you prefer to install the package directly on your system, run:

```bash
pip install -e .
```

This will install the package and its dependencies while allowing you to modify the source code.
