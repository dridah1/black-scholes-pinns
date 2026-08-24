import torch
import numpy as np
from scipy.stats import norm


def black_scholes_residual(model, x, sigma, r, dividend=0.0):
    """
    Compute the Black-Scholes PDE residual.

    The Black-Scholes PDE is:

        V_t + 0.5 * sigma^2 * S^2 * V_SS
        + (r - dividend) * S * V_S - r * V = 0

    Parameters
    ----------
    model : torch.nn.Module
        PINN approximating V(S, t).

    x : torch.Tensor
        Input tensor containing [S, t].

    sigma : float
        Volatility.

    r : float
        Risk-free interest rate.

    Returns
    -------
    torch.Tensor
        PDE residual.
    """

    x = x.clone().detach().requires_grad_(True)

    V = model(x)

    gradients = torch.autograd.grad(
        V,
        x,
        grad_outputs=torch.ones_like(V),
        create_graph=True
    )[0]

    V_S = gradients[:, 0:1]
    V_t = gradients[:, 1:2]

    second_gradients = torch.autograd.grad(
        V_S,
        x,
        grad_outputs=torch.ones_like(V_S),
        create_graph=True
    )[0]

    V_SS = second_gradients[:, 0:1]

    S = x[:, 0:1]

    residual = (
        V_t
        + 0.5 * sigma**2 * S**2 * V_SS
        + r * S * V_S
        - r * V
    )

    return residual
def european_call_payoff(S, K):
    """
    Terminal payoff of a European call option.

    V(S, T) = max(S - K, 0)

    Parameters
    ----------
    S : torch.Tensor
        Underlying asset prices.

    K : float
        Strike price.

    Returns
    -------
    torch.Tensor
        European call payoff.
    """
def analytical_call_price(S, t, K, r, sigma, T, dividend=0.0):
    """
    Analytical Black-Scholes price of a European call option.

    Parameters
    ----------
    S : array-like
        Underlying asset price.

    t : array-like
        Current time.

    K : float
        Strike price.

    r : float
        Risk-free interest rate.

    sigma : float
        Volatility.

    T : float
        Maturity.

    dividend : float
        Continuous dividend yield.

    Returns
    -------
    numpy.ndarray
        Analytical European call price.
    """

    S = np.asarray(S, dtype=float)
    t = np.asarray(t, dtype=float)

    tau = T - t

    # Avoid division by zero exactly at maturity.
    tau_safe = np.maximum(tau, 1e-12)

    d1 = (
        np.log(S / K)
        + (r - dividend + 0.5 * sigma**2) * tau_safe
    ) / (
        sigma * np.sqrt(tau_safe)
    )

    d2 = d1 - sigma * np.sqrt(tau_safe)

    price = (
        S * np.exp(-dividend * tau_safe) * norm.cdf(d1)
        - K * np.exp(-r * tau_safe) * norm.cdf(d2)
    )

    # At maturity, enforce the exact payoff.
    maturity_mask = tau <= 0

    if np.any(maturity_mask):
        price = np.asarray(price)
        price[maturity_mask] = np.maximum(
            S[maturity_mask] - K,
            0.0
        )

    return price

    return torch.maximum(
        S - K,
        torch.zeros_like(S)
    )
