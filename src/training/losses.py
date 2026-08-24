import torch

from src.physics.black_scholes import black_scholes_residual


def mean_squared_error(prediction, target):
    """
    Compute mean squared error.
    """

    return torch.mean((prediction - target) ** 2)


def pde_loss(model, x, sigma, r, dividend=0.0):
    """
    Compute the physics loss associated with
    the Black-Scholes PDE.

    The PINN should satisfy:

        V_t
        + 0.5 * sigma^2 * S^2 * V_SS
        + (r - q) * S * V_S
        - r * V = 0

    Parameters
    ----------
    model : torch.nn.Module
        PINN model.

    x : torch.Tensor
        Interior collocation points.

    sigma : float
        Volatility.

    r : float
        Risk-free interest rate.

    dividend : float
        Dividend yield.

    Returns
    -------
    torch.Tensor
        Mean squared PDE residual.
    """

    residual = black_scholes_residual(
        model=model,
        x=x,
        sigma=sigma,
        r=r,
        dividend=dividend
    )

    return torch.mean(residual ** 2)
