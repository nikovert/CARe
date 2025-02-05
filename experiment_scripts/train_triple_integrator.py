# ================================ #
# Train Triple Integrator System   #
# ================================ #

# Fix Python Import Path for Cross-Module Access
import sys
import os
import shutil
import random # to generate random seed 

# Add project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Standard Imports
import torch
import modules  # Model definition
import certreach.training as training  # Training loop
import loss_functions
from dataio import TripleIntegratorDataset

# Plotting and Argument Parsing Libraries
import matplotlib
matplotlib.use('Agg')  # Disable interactive mode for plotting
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import configargparse

# For model structure and saving models
from modules_beyond import (get_experiment_folder, save_experiment_details,
                            check_existing_experiment, extract_symbolic_model,
                            extract_dreal_partials,serializable_to_sympy,sympy_to_serializable,
                            process_dreal_result)
import torch.multiprocessing as mp
import json
from modules_dreal import dreal_triple_integrator_BRS,CounterexampleDatasetTripleIntegrator


# ================================ #
# Argument Parser Setup            #
# ================================ #

# Create an argument parser for command-line options
p = configargparse.ArgumentParser()

# Logging and Experiment Settings
p.add_argument('--logging_root', type=str, default='./logs', help='Root directory for logging.')
p.add_argument('--experiment_name', type=str, default="Triple_integrator", help='Name of the experiment.')

# Training Settings
p.add_argument('--batch_size', type=int, default=32)
p.add_argument('--lr', type=float, default=2e-5, help='Learning rate.')
p.add_argument('--num_epochs', type=int, default=30000, help='Number of training epochs.')
p.add_argument('--epochs_til_ckpt', type=int, default=1000, help='Checkpoint saving frequency.')
p.add_argument('--steps_til_summary', type=int, default=100, help='Logging summary frequency.')

# Model Settings
p.add_argument('--model', type=str, default='sine', choices=['sine', 'tanh', 'sigmoid', 'relu'], help='Activation type.')
p.add_argument('--mode', type=str, default='mlp', choices=['mlp', 'rbf', 'pinn'], help='Model architecture.')
p.add_argument('--in_features', type=int, default=4, help='Number of input features.')  # Added
p.add_argument('--out_features', type=int, default=1, help='Number of output features.')  # Added
p.add_argument('--tMin', type=float, default=0.0, help='Start time of the simulation.')
p.add_argument('--tMax', type=float, default=1.0, help='End time of the simulation.')
p.add_argument('--num_hl', type=int, default=0, help='Number of hidden layers.')
p.add_argument('--num_nl', type=int, default=16, help='Number of neurons per layer.')
p.add_argument('--minWith', type=str, default='none', choices=['none', 'zero', 'target'], help='Constraint type.')
p.add_argument('--reachMode', type=str, default='forward', choices=['backward', 'forward'], help='reach mode type.')
p.add_argument('--reachAim', type=str, default='reach', choices=['avoid', 'reach'], help='reach mode type.')
p.add_argument('--setType', type=str, default='set', choices=['set', 'tube'], help='set type.')
p.add_argument('--radius', type=float, default=0.25, help='collision / reach radius')

# Polynomial Layer Settings
p.add_argument('--use_polynomial', action='store_true', default=True, help='Enable polynomial layer.')
p.add_argument('--poly_degree', type=int, default=2, help='Polynomial layer degree.')

# Double Integrator Specific Settings
p.add_argument('--input_max', type=float, default=1.0, help='Maximum control input (acceleration).')
p.add_argument('--pretrain', action='store_true', default=True, help='Enable pretraining mode.')
p.add_argument('--pretrain_iters', type=int, default=2000, help='Number of pretraining iterations.')
p.add_argument('--counter_start', type=int, default=0, help='Start of curriculum learning.')
p.add_argument('--counter_end', type=int, default=100e3, help='End of curriculum learning.')
p.add_argument('--num_src_samples', type=int, default=1000, help='Number of samples from initial state.')
p.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility.')

# Dreal Specific Settings
p.add_argument('--epsilon', type=float, default=0.35, help='Error from the actual value function')

# Re-training Settings 
p.add_argument('--iterate', type=bool, default=True, help='maximum number of iterations')
p.add_argument('--epsilon_radius', type=float, default=0.1, help='epsilon radius for data point for re-training')
p.add_argument('--max_iterations', type=int, default=8, help='maximum number of iterations')

# Load Parsed Arguments
opt = p.parse_args()


if __name__ == '__main__':

    mp.set_start_method('spawn', force=True)
    # ================================ #
    # Check for Existing Experiment    #
    # ================================ #

    # Check if a matching experiment exists
    found, experiment_folder = check_existing_experiment(opt.logging_root, vars(opt))

    if found:
        print(f"[INFO] Found matching experiment at {experiment_folder}")

        root_path = experiment_folder
        
    else:
        # If no matching experiment is found, create a new folder
        root_path = get_experiment_folder(opt.logging_root, opt.experiment_name)
        print(f"[INFO] Starting new experiment at: {root_path}")


        # ================================ #
        # Dataset Creation                 #
        # ================================ #

        # Create a dataset for the Double Integrator system
        dataset = TripleIntegratorDataset(
            numpoints=85000, tMin=opt.tMin, tMax=opt.tMax, 
            input_max=opt.input_max, pretrain=opt.pretrain,
            pretrain_iters=opt.pretrain_iters,
            counter_start=opt.counter_start, counter_end=opt.counter_end, 
            num_src_samples=opt.num_src_samples, seed=opt.seed, radius=opt.radius
        )

        # DataLoader Initialization
        dataloader = DataLoader(dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)

        # ================================ #
        # Model Definition                 #
        # ================================ #

        # Initialize the neural network model
        model = modules.SingleBVPNet(
            in_features=opt.in_features, # [time, position, velocity]
            out_features=opt.out_features,
            type=opt.model, 
            mode=opt.mode,
            hidden_features=opt.num_nl,
            num_hidden_layers=opt.num_hl, 
            use_polynomial=opt.use_polynomial, 
            poly_degree=opt.poly_degree,
        ).cuda()

        # Initialize Loss Function from loss_functions.py
        loss_fn = loss_functions.initialize_hji_triple_integrator(dataset, opt.minWith, opt.reachMode, opt.reachAim)



        # ================================ #
        # Experiment Folder Setup          #
        # ================================ #

        # Find or create the experiment folder
        # Setup experiment folder
        experiment_folder = get_experiment_folder(opt.logging_root, opt.experiment_name)
        root_path = experiment_folder


        # ================================ #
        # Training Procedure               #
        # ================================ #

        # Perform training
        try:
            training.train(
                model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
                steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
                model_dir=root_path, loss_fn=loss_fn, clip_grad=False,
                use_lbfgs=False, validation_fn=None,
                start_epoch=0
            )
            # Save the model details
            save_experiment_details(root_path, str(loss_fn), vars(opt))
        except KeyboardInterrupt:
            save_experiment_details(root_path, str(loss_fn), vars(opt)) 

    # Define the path to the model checkpoint
    # This is where the trained model's weights are saved after training
    checkpoint_path = os.path.join(root_path, "checkpoints", "model_final.pth")

    # Initialize the model architecture on CPU
    # Loading the model on CPU avoids CUDA-related multiprocessing issues during symbolic extraction
    model = modules.SingleBVPNet(
        in_features=opt.in_features,        # Number of input features (e.g., time, position, velocity)
        out_features=opt.out_features,      # Number of output features (e.g., value function output)
        type=opt.model,                     # Activation function type (e.g., 'sine', 'relu')
        mode=opt.mode,                      # Network architecture mode (e.g., 'mlp')
        hidden_features=opt.num_nl,         # Number of neurons in each hidden layer
        num_hidden_layers=opt.num_hl,       # Number of hidden layers in the network
        use_polynomial=opt.use_polynomial,  # Whether to use a polynomial layer in the network
        poly_degree=opt.poly_degree,        # Degree of the polynomial layer if enabled
    ).cpu()  # Ensure the model runs on CPU for symbolic processing

    # Load the trained model's weights from the checkpoint file
    # This restores the model to its previously trained state
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    print("[INFO] Loaded model from checkpoint.")
    

    # Define file paths
    symbolic_model_file = os.path.join(root_path, "symbolic_model.json")            


    # ============================= #
    # Load or Calculate the Model   #
    # ============================= #
    if os.path.isfile(symbolic_model_file):
        print("[INFO] Found existing symbolic model. Loading from file...")

        # Load symbolic model from file
        with open(symbolic_model_file, 'r') as f:
            symbolic_model = serializable_to_sympy(json.load(f))

        print("[INFO] Loaded symbolic model successfully.")
    else:
        print("[INFO] Symbolic model not found. Extracting and saving...")

        # Extract symbolic model from the trained neural network
        symbolic_model = extract_symbolic_model(model, root_path)

        # Save symbolic model
        with open(symbolic_model_file, 'w') as f:
            json.dump(sympy_to_serializable(symbolic_model), f, indent=4)
        
        print("[INFO] Saved symbolic model.")


    # ============================= #
    # Extract Partial Derivatives   #
    # ============================= #
    result = extract_dreal_partials(symbolic_model, in_features=opt.in_features)

    # Extract partials
    sympy_partials = result["partials"]
    dreal_partials = result["dreal_partials"]
    dreal_variables = result["dreal_variables"]

    
    # ============================= #
    # Check Dreal condition         #
    # ============================= #

    # Define the file path for dReal result
    dreal_result_path = os.path.join(root_path, "dreal_result.json")


    if not os.path.isfile(dreal_result_path):
        # If the result file does not exist, execute dReal
        print("[INFO] dReal result file not found. Executing dReal verification...")
        epsilon = opt.epsilon
        dreal_triple_integrator_BRS(
            dreal_partials,
            dreal_variables,
            epsilon=epsilon,
            reachMode=opt.reachMode,
            reachAim=opt.reachAim,
            setType=opt.setType,
            save_directory=root_path,
        )
        print("[INFO] dReal verification completed.")
    else:
        # Load the epsilon value from the existing result
        with open(dreal_result_path, "r") as f:
            dreal_result = json.load(f)
        epsilon = dreal_result.get("epsilon", opt.epsilon)  # Default to 0.37 if not found

    # Process and display the result (whether the file existed or was created)
    result = process_dreal_result(dreal_result_path)

    # ============================= #
    # Iterative Update Loop         #
    # ============================= #
    iterate = opt.iterate

    if iterate:

        epsilon_updated = opt.epsilon        # Initialize epsilon
        max_iterations = opt.max_iterations  # Maximum number of iterations
        iteration_count = 0                  # Counter for iterations
        counterexample = result["counterexample"]  # Initial counterexample from the primary run

        if not result["counterexample"]:
            print("[INFO] No Counterexample found. Reducing the epsilon to 0.9 epsilon")
            epsilon_updated *= 0.9 # Reduce epsilon

        # Path to the updated model
        updated_model_path = os.path.join(root_path, "checkpoints", "updated_model.pth")
        model_final_path = os.path.join(root_path, "checkpoints", "model_final.pth")

        while iteration_count < max_iterations:
            print(f"\n[INFO] Starting iteration {iteration_count + 1} with epsilon: {epsilon_updated}")

            # Load the latest model
            if os.path.exists(updated_model_path):
                print("[INFO] Loading updated model for the current iteration...")
                model_update = modules.SingleBVPNet(
                    in_features=opt.in_features,
                    out_features=opt.out_features,
                    type=opt.model,
                    mode=opt.mode,
                    hidden_features=opt.num_nl,
                    num_hidden_layers=opt.num_hl,
                    use_polynomial=opt.use_polynomial,
                    poly_degree=opt.poly_degree,
                ).cuda()
                model_update.load_state_dict(torch.load(updated_model_path))
            else:
                print("[INFO] No updated model found. Loading final model for the current iteration...")
                model_update = modules.SingleBVPNet(
                    in_features=opt.in_features,
                    out_features=opt.out_features,
                    type=opt.model,
                    mode=opt.mode,
                    hidden_features=opt.num_nl,
                    num_hidden_layers=opt.num_hl,
                    use_polynomial=opt.use_polynomial,
                    poly_degree=opt.poly_degree,
                ).cuda()
                model_update.load_state_dict(torch.load(model_final_path))
            print("[INFO] Model loaded successfully.")


            if counterexample:
                print("[INFO] Counterexample found. Creating relearning dataset...")

                # Generate a random seed
                random_seed = random.randint(0, 2**32 - 1)

                # Create relearning dataset
                relearning_dataset = CounterexampleDatasetTripleIntegrator(
                    counterexample=counterexample,
                    numpoints=10000,
                    percentage_in_counterexample=20,
                    percentage_at_t0=20,
                    tMin=opt.tMin,
                    tMax=opt.tMax,
                    input_max=opt.input_max,
                    seed=random_seed, # Use dynamically generated random seed
                    collision_radius=opt.radius,
                    epsilon_radius=opt.epsilon_radius,
                )
                print("[INFO] Relearning dataset created successfully.")

                # Prepare DataLoader
                dataloader = DataLoader(
                    relearning_dataset,
                    shuffle=True,
                    batch_size=opt.batch_size,
                    pin_memory=True,
                    num_workers=0,
                )


                # Define loss function
                loss_fn = loss_functions.initialize_hji_triple_integrator(
                    dataset=relearning_dataset,
                    minWith=opt.minWith,
                    reachMode=opt.reachMode,
                    reachAim=opt.reachAim,
                )

                # Retrain directory setup
                retrain_dir = os.path.join(root_path, f"checkpoints/re_train_iter_{iteration_count + 1}")
                if os.path.exists(retrain_dir):
                    shutil.rmtree(retrain_dir)
                os.makedirs(retrain_dir, exist_ok=True)

                # Retrain the model
                print("[INFO] Retraining model...")
                training.train(
                    model=model_update,
                    train_dataloader=dataloader,
                    epochs=opt.num_epochs,
                    lr=opt.lr,
                    steps_til_summary=opt.steps_til_summary,
                    epochs_til_checkpoint=opt.epochs_til_ckpt,
                    model_dir=retrain_dir,
                    loss_fn=loss_fn,
                    clip_grad=False,
                    use_lbfgs=False,
                    validation_fn=None,
                    start_epoch=0,
                )
                print("[INFO] Retraining completed.")
                torch.save(model_update.state_dict(), updated_model_path)
                print(f"[INFO] Updated model saved at {updated_model_path}.")

            # Run dReal verification
            print("[INFO] Running dReal verification...")
            model_updated = modules.SingleBVPNet(
                in_features=opt.in_features,
                out_features=opt.out_features,
                type=opt.model,
                mode=opt.mode,
                hidden_features=opt.num_nl,
                num_hidden_layers=opt.num_hl,
                use_polynomial=opt.use_polynomial,
                poly_degree=opt.poly_degree,
            ).cpu()
            model_updated.load_state_dict(torch.load(updated_model_path, map_location=torch.device('cpu')))
            symbolic_model_updated = extract_symbolic_model(model_updated, root_path)
            result_updated = extract_dreal_partials(symbolic_model_updated, in_features=opt.in_features)

            dreal_partials_updated = result_updated["dreal_partials"]
            dreal_variables_updated = result_updated["dreal_variables"]

            # Ensure directory for dReal results
            updated_dreal_folder = os.path.join(root_path, f"updated_model_dreal_result_iter_{iteration_count + 1}")
            os.makedirs(updated_dreal_folder, exist_ok=True)
            updated_dreal_result_path = os.path.join(updated_dreal_folder, "dreal_result.json")

            dreal_triple_integrator_BRS(
                dreal_partials=dreal_partials_updated,
                dreal_variables=dreal_variables_updated,
                epsilon=epsilon_updated,
                reachMode=opt.reachMode,
                reachAim=opt.reachAim,
                setType=opt.setType,
                save_directory=updated_dreal_folder,
            )
            print(f"[INFO] dReal verification completed. Result saved to {updated_dreal_result_path}.")

            # Process dReal result
            result_updated = process_dreal_result(updated_dreal_result_path)

            if result_updated["counterexample"]:
                print("[INFO] Counterexample found. Continuing with next iteration...")
                counterexample = result_updated["counterexample"]
            else:
                print("[INFO] HJB Equation satisfied. Reducing epsilon and continuing...")
                epsilon_updated *= 0.9  # Reduce epsilon
                counterexample = None  # Clear counterexample for the next iteration

            iteration_count += 1
            
        print("[INFO] Iterative update loop completed.")

                