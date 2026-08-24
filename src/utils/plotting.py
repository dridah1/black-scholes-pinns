import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(history):
    """
    Plot PINN training losses.
    """

    plt.figure(figsize=(8, 5))

    plt.semilogy(
        history["total_loss"],
        label="Total Loss"
    )

    plt.semilogy(
        history["pde_loss"],
        label="PDE Loss"
    )

    plt.semilogy(
        history["terminal_loss"],
        label="Terminal Loss"
    )

    plt.semilogy(
        history["boundary_loss"],
        label="Boundary Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("PINN Training History")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_solution_comparison(
    S,
    pinn_solution,
    analytical_solution
):
    """
    Compare PINN and analytical Black-Scholes solutions.
    """

    S = np.asarray(S)

    plt.figure(figsize=(8, 5))

    plt.plot(
        S,
        analytical_solution,
        label="Analytical"
    )

    plt.plot(
        S,
        pinn_solution,
        "--",
        label="PINN"
    )

    plt.xlabel("Underlying Asset Price S")
    plt.ylabel("Option Price")
    plt.title("PINN vs Analytical Black-Scholes Solution")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_absolute_error(
    S,
    pinn_solution,
    analytical_solution
):
    """
    Plot absolute error between PINN and analytical solution.
    """

    S = np.asarray(S)

    error = np.abs(
        np.asarray(pinn_solution)
        - np.asarray(analytical_solution)
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        S,
        error
    )

    plt.xlabel("Underlying Asset Price S")
    plt.ylabel("Absolute Error")
    plt.title("PINN Absolute Error")

    plt.grid(True)

    plt.tight_layout()
    plt.show()
