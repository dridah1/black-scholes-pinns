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
def terminal_condition_loss(model, x_terminal, K):
    """
    Compute the terminal-condition loss for a
    European call option.

    The terminal condition is:

        V(S, T) = max(S - K, 0)

    Parameters
    ----------
    model : torch.nn.Module
        PINN model.

    x_terminal : torch.Tensor
        Points on the terminal boundary t = T.
        Expected ordering: [t, S].

    K : float
        Strike price.

    Returns
    -------
    torch.Tensor
        Mean squared terminal-condition error.
    """

    t = x_terminal[:, 0:1]
    S = x_terminal[:, 1:2]

    inputs = torch.cat((t, S), dim=1)

    prediction = model(inputs)

    payoff = torch.maximum(
        S - K,
        torch.zeros_like(S)
    )

    return mean_squared_error(prediction, payoff)
def lower_boundary_loss(model, x_boundary):
    """
    Compute the lower boundary loss at S = 0.

    For a European call option:

        V(0, t) = 0

    Parameters
    ----------
    model : torch.nn.Module
        PINN model.

    x_boundary : torch.Tensor
        Points on the lower boundary S = 0.
        Expected ordering: [t, S].

    Returns
    -------
    torch.Tensor
        Mean squared boundary-condition error.
    """

    prediction = model(x_boundary)

    target = torch.zeros_like(prediction)

    return mean_squared_error(prediction, target)
def total_loss(
    model,
    x_collocation,
    x_terminal,
    x_boundary,
    sigma,
    r,
    K,
    dividend=0.0,
    pde_weight=100.0,
    terminal_weight=1.0,
    boundary_weight=1.0
):
    """
    Compute the total PINN loss.

    Total loss:

        L = lambda_f L_f
          + lambda_T L_T
          + lambda_B L_B

    Parameters
    ----------
    model : torch.nn.Module
        PINN model.

    x_collocation : torch.Tensor
        Interior PDE collocation points.

    x_terminal : torch.Tensor
        Terminal-condition points.

    x_boundary : torch.Tensor
        Lower-boundary points.

    sigma : float
        Volatility.

    r : float
        Risk-free interest rate.

    K : float
        Strike price.

    dividend : float
        Dividend yield.

    pde_weight : float
        Weight of PDE loss.

    terminal_weight : float
        Weight of terminal-condition loss.

    boundary_weight : float
        Weight of boundary-condition loss.

    Returns
    -------
    tuple
        Total loss and individual loss components.
    """

    loss_f = pde_loss(
        model,
        x_collocation,
        sigma,
        r,
        dividend
    )

    loss_terminal = terminal_condition_loss(
        model,
        x_terminal,
        K
    )

    loss_boundary = lower_boundary_loss(
        model,
        x_boundary
    )

    total = (
        pde_weight * loss_f
        + terminal_weight * loss_terminal
        + boundary_weight * loss_boundary
    )

    return total, loss_f, loss_terminal, loss_boundary
