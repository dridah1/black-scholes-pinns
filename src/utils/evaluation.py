import numpy as np


def mean_absolute_error(predicted, exact):
    """
    Compute Mean Absolute Error (MAE).
    """

    predicted = np.asarray(predicted)
    exact = np.asarray(exact)

    return np.mean(np.abs(predicted - exact))


def mean_squared_error(predicted, exact):
    """
    Compute Mean Squared Error (MSE).
    """

    predicted = np.asarray(predicted)
    exact = np.asarray(exact)

    return np.mean((predicted - exact) ** 2)


def root_mean_squared_error(predicted, exact):
    """
    Compute Root Mean Squared Error (RMSE).
    """

    return np.sqrt(
        mean_squared_error(predicted, exact)
    )


def relative_l2_error(predicted, exact):
    """
    Compute relative L2 error.

        ||prediction - exact||_2
        -------------------------
              ||exact||_2
    """

    predicted = np.asarray(predicted)
    exact = np.asarray(exact)

    numerator = np.linalg.norm(
        predicted - exact
    )

    denominator = np.linalg.norm(exact)

    return numerator / denominator
