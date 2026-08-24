"""
Main entry point for the Black-Scholes PINN project.
"""
import random

import numpy as np
import torch

from src.models.pinn import BlackScholesPINN
import numpy as np
from src.physics.data import (
    generate_collocation_points,
    generate_terminal_points,
    generate_lower_boundary_points
)
from src.training.trainer import PINNTrainer
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def main():
    set_seed(SEED)
    # =========================
    # Black-Scholes parameters
    # =========================

    K = 100.0              # Strike price
    r = 0.05               # Risk-free interest rate
    sigma = 0.20           # Volatility
    dividend = 0.0         # Dividend yield
    T = 1.0                # Maturity

    # =========================
    # Computational domain
    # =========================

    S_min = 0.0
    S_max = 200.0

    t_min = 0.0
    t_max = T

    # =========================
    # Training parameters
    # =========================

    num_collocation = 5000
    num_terminal = 1000
    num_boundary = 1000

    epochs = 5000
    learning_rate = 1e-3

    print("Black-Scholes PINN experiment")
    print(f"Strike price: {K}")
    print(f"Volatility: {sigma}")
    print(f"Risk-free rate: {r}")
    print(f"Dividend yield: {dividend}")
    print(f"Maturity: {T}")
        # =========================
    # Generate training data
    # =========================

    x_collocation = generate_collocation_points(
        num_points=num_collocation,
        S_min=S_min,
        S_max=S_max,
        t_min=t_min,
        t_max=t_max
    )

    x_terminal = generate_terminal_points(
        num_points=num_terminal,
        S_min=S_min,
        S_max=S_max,
        T=T
    )

    x_boundary = generate_lower_boundary_points(
        num_points=num_boundary,
        t_min=t_min,
        t_max=t_max
    )

    print()
    print("Training data generated:")
    print(f"Collocation points: {x_collocation.shape}")
    print(f"Terminal points:    {x_terminal.shape}")
    print(f"Boundary points:    {x_boundary.shape}")
        # =========================
    # Device
    # =========================

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
        # =========================
    # Convert data to tensors
    # =========================

    x_collocation = torch.tensor(
        x_collocation,
        dtype=torch.float32,
        device=device
    )

    x_terminal = torch.tensor(
        x_terminal,
        dtype=torch.float32,
        device=device
    )

    x_boundary = torch.tensor(
        x_boundary,
        dtype=torch.float32,
        device=device
    )
    # =========================
    # Create PINN
    # =========================

    model = BlackScholesPINN(
        input_dim=2,
        hidden_dim=50,
        num_hidden_layers=9
    ).to(device)
        # =========================
    # Optimizer
    # =========================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )
        # =========================
    # Trainer
    # =========================

    trainer = PINNTrainer(
        model=model,
        optimizer=optimizer,
        x_collocation=x_collocation,
        x_terminal=x_terminal,
        x_boundary=x_boundary,
        sigma=sigma,
        r=r,
        K=K,
        dividend=dividend,
        pde_weight=100.0,
        terminal_weight=1.0,
        boundary_weight=1.0,
        device=device
    )
    print("Trainer initialized.")
        # =========================
    # Training
    # =========================

    print("\nStarting training...")

    history = trainer.train(
        epochs=100
    )

    print("\nTraining completed.")

    print(
        f"Initial total loss: "
        f"{history['total_loss'][0]:.6e}"
    )

    print(
        f"Final total loss: "
        f"{history['total_loss'][-1]:.6e}"
    )    print("Optimizer: Adam")
    print(f"Learning rate: {learning_rate}")
    print(model)
if __name__ == "__main__":
    main()
