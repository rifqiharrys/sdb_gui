from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray
from sklearn import metrics


class EvaluationMetrics(NamedTuple):
    """
    A named tuple to store evaluation metrics
    """

    rmse: float
    mae: float
    r2: float


def out_depth_filter(
        array: NDArray,
        top_limit: float = 0.0,
        bottom_limit: float = -12.0
) -> NDArray:
    """
    Filter depth prediction output in an array for the allowed depth
    in positive up direction by changing it to NaN.

    Parameters
    ----------
    array : NDArray
        1D array of depth data.
    top_limit : float, optional
        Top depth limit in positive up. Default value is 0.0.
    bottom_limit : float, optional
        Bottom depth limit in positive up. Default value is -12.0.

    Returns
    -------
    NDArray
        Filtered array with values outside the limits set to NaN.
    """

    # Exchange value of top_limit and bottom_limit if top < bottom
    if top_limit < bottom_limit:
        top_limit, bottom_limit = bottom_limit, top_limit

    filtered_array = np.where(
        (array > top_limit) | (array < bottom_limit),
        np.nan,
        array
    )

    return filtered_array


def reshape_prediction(
        array: NDArray,
        raster: xr.DataArray
) -> NDArray:
    """
    Reshape depth prediction in 1D array to a 2D array shape
    that is similar to its source raster.

    Parameters
    ----------
    array : NDArray
        Depth prediction data in the shape of 1D array.
    raster : xr.DataArray
        Raster data that is read using rioxarray.

    Returns
    -------
    NDArray
        Reshaped array.
    """

    reshaped = array.reshape(raster.values[0].shape)

    return reshaped


def evaluate(
        true_val: ArrayLike,
        pred_val: ArrayLike
) -> EvaluationMetrics:
    """
    Evaluate predicted values from true values by calculating
    RMSE, MAE, and R Squared values.

    Parameters
    ----------
    true_val : ArrayLike
        True values.
    pred_val : ArrayLike
        Predicted values.

    Returns
    -------
    EvaluationMetrics
        Named tuple containing RMSE, MAE, and R Squared.
    """

    rmse = float(metrics.root_mean_squared_error(true_val, pred_val))
    mae = float(metrics.mean_absolute_error(true_val, pred_val))
    r2 = float(metrics.r2_score(true_val, pred_val))

    return EvaluationMetrics(rmse=rmse, mae=mae, r2=r2)


def scatter_plotter(
        true_val: ArrayLike,
        pred_val: ArrayLike,
        plot_color: str = 'royalblue',
        line_color: str = 'r',
        title: str = 'Scatter Plot'
) -> tuple[Figure, Axes]:
    """
    Create a scatter plot of in situ depth against predicted depth
    and plot a pred_val=true_val line.

    Parameters
    ----------
    true_val : ArrayLike
        X coordinates. True values.
    pred_val : ArrayLike
        Y coordinates. Predicted values.
    plot_color : str
        Point color. Default is 'royalblue'.
    line_color : str
        Line color. Default is 'r'.
    title : str
        Graph title. Default is 'Scatter Plot'.

    Returns
    -------
    Tuple[Figure, Axes]
        A tuple containing (matplotlib figure, matplotlib axes).
    """

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(true_val, pred_val, marker='.', color=plot_color, facecolors='none')
    min_val, max_val = round(np.nanmin(true_val)), round(np.nanmax(true_val))
    ax.plot([min_val, max_val], [min_val, max_val], color=line_color)
    ax.set_xlabel('True Depth')
    ax.set_ylabel('Predicted Depth')
    ax.set_title(title)

    return fig, ax