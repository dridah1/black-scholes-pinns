import torch


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

    return torch.maximum(
        S - K,
        torch.zeros_like(S)
    )
