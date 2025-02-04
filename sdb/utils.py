from scipy import ndimage
import numpy as np
import pandas as pd
import xarray as xr


def point_sampling(raster: xr.DataArray, x, y, include_xy: bool = True):
    """
    Extract raster values from a dataarray based on xy coordinates.
    XY coordinates have to be in the same CRS as raster.

    Parameter:
    ----------
    raster: xr.DataArray
        DataArray from rioxarray.
    x
        X coordinates.
    y
        Y coordinates.

    Return
    ------
    DataFrame
    """

    x_reindex = x.reset_index(drop=True)
    y_reindex = y.reset_index(drop=True)

    x_in = xr.DataArray(x_reindex, dims=['location'])
    y_in = xr.DataArray(y_reindex, dims=['location'])

    point_samples = raster.sel(x=x_in, y=y_in, method='nearest').values.T

    point_samples_df = pd.DataFrame(
        point_samples,
        columns=[f'band_{i}' for i in raster.band.values]
    ).reset_index(drop=True)

    if include_xy:
        point_samples_df['x'], point_samples_df['y'] = x_reindex, y_reindex

    return point_samples_df


def median_filter(
        array: np.ndarray,
        filter_size: int = 3
) -> np.ndarray:
    """
    Calculate median filter of a 2D array.

    Parameter:
    ----------
    array: np.ndarray
        2D array data.
    raster: xr.DataArray
        Raster data that read using rioxarray.

    Return
    ------
    Filtered array.
    """

    if filter_size < 3 or filter_size % 2 == 0:
        raise ValueError(
            f'Invalid filter_size value: {filter_size}\n'
            f'Allowed value: >= 3 or odd numbers'
        )

    filtered = ndimage.median_filter(array, size=filter_size)

    return filtered


def array_to_dataarray(
        array: np.ndarray,
        data_array: xr.DataArray,
        band_name: str | int =1,
        attrs: bool = False
):
    """
    Create a new Dataarray from a 2D Numpy array based on 
    rioxarray image specification but only contain 1 band.

    Parameter:
    ----------
    array: np.ndrray
        image data that read using rioxarray
    data_array: xr.DataArray
        dataarray from rioxarray
    band_name: str | int = 1
        A name for the band
    attrs: bool = False
        Copy attributes

    Return
    ------
    A dataarray with the same dimension and coordinates as input dataarray
    """

    new_da = xr.DataArray(
        array[np.newaxis, :, :],
        dims=data_array.dims,
        coords={
            'band': [band_name],
            'y': data_array.coords['y'],
            'x': data_array.coords['x']
        },
        attrs=data_array.attrs if attrs else None
    )

    new_da.rio.write_crs(data_array.rio.crs, inplace=True)

    return new_da