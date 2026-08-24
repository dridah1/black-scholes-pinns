import numpy as np


def generate_collocation_points(
    num_points,
    S_min,
    S_max,
    t_min,
    t_max
):
    """
    Generate collocation points inside the Black-Scholes domain.

    Parameters
    ----------
    num_points : int
        Number of interior points.

    S_min, S_max : float
        Minimum and maximum asset prices.

    t_min, t_max : float
        Minimum and maximum time values.

    Returns
    -------
    numpy.ndarray
        Collocation points with columns [t, S].
    """

    t = np.random.uniform(t_min, t_max, num_points)
    S = np.random.uniform(S_min, S_max, num_points)

    return np.column_stack((t, S))


def generate_terminal_points(
    num_points,
    S_min,
    S_max,
    T
):
    """
    Generate points on the terminal boundary t = T.

    Returns points with columns [t, S].
    """

    S = np.random.uniform(S_min, S_max, num_points)
    t = np.full(num_points, T)

    return np.column_stack((t, S))


def generate_lower_boundary_points(
    num_points,
    t_min,
    t_max
):
    """
    Generate points on the lower asset-price boundary S = 0.

    Returns points with columns [t, S].
    """

    t = np.random.uniform(t_min, t_max, num_points)
    S = np.zeros(num_points)

    return np.column_stack((t, S))
