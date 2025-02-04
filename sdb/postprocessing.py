from sklearn import metrics
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


def out_depth_filter(
        array: np.ndarray,
        top_limit: float = 0.0,
        bottom_limit: float = -12.0
):
    """
    Filter depth prediction output in 1D Array based on allowed depth
    in positive up direction by change it to NaN.

    Parameter:
    ----------
    array: np.ndarray
        1D array of depth data.
    top_limit: float
        Top depth limit in positive up. Defaula value is 0.
    bottom_limit: float
        Bottom depth limit in positive up. Defaula value is 12.0.

    Return
    ------
    Array
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
        array: np.ndarray,
        raster: xr.DataArray
):
    """
    Reshape depth prediction in 1D array to a 2D array shape
    that is similar to its source raster.

    Parameter:
    ----------
    array: np.ndarray
        Depth prediction data in the shape of 1D array.
    raster: xr.DataArray
        Raster data that read using rioxarray.

    Return
    ------
    Reshaped array
    """

    reshaped = array.reshape(raster.values[0].shape)

    return reshaped


def evaluate(true_val, pred_val):
    """
    Evaluate predicted values from true values by calculating
    RMSE, MAE, and R Squared values.

    Parameter:
    ----------

    true_val
        True values.
    pred_val
        Predicted values.

    Return
    ------
    Tuple of RMSE, MAE, and R Squared.
    """

    rmse = metrics.root_mean_squared_error(true_val, pred_val)
    mae = metrics.mean_absolute_error(true_val, pred_val)
    r2 = metrics.r2_score(true_val, pred_val)

    return rmse, mae, r2


def scatter_plotter(
        x,
        y,
        plot_color: str = 'royalblue',
        line_color: str = 'r',
        title: str = 'Scatter Plot'
):
    """
    Create a scatter plot of in situ depth against predicted depth
    and plot a y=x line

    Parameter:
    ----------
    x: 
        X coordinates
    y: 
        Y coordinates
    plot_color: str = 'royalblue'
        Point color
    line_color: str = 'r'
        Line color
    title: str = 'Scatter Plot'
        Graph title

    Return
    ------
    A tuple of figure and axes
    """

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(x, y, marker='.', color=plot_color, facecolors='none')
    min_val, max_val = round(np.nanmin(x)), round(np.nanmax(x))
    ax.plot([min_val, max_val], [min_val, max_val], color=line_color)
    ax.set_xlabel('True Depth')
    ax.set_ylabel('Predicted Depth')
    ax.set_title(title)

    return fig, ax