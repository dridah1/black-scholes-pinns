import torch


def mean_squared_error(prediction, target):
    """
    Compute mean squared error.

    Parameters
    ----------
    prediction : torch.Tensor
        Model predictions.

    target : torch.Tensor
        Target values.

    Returns
    -------
    torch.Tensor
        Mean squared error.
    """

    return torch.mean((prediction - target) ** 2)
