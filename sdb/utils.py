from contextlib import contextmanager

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray
from rich.progress import Progress, SpinnerColumn, TextColumn
from scipy import ndimage


@contextmanager
def progress_spinner(description: str, style: str):
    """Context manager for progress spinner."""
    with Progress(
        SpinnerColumn(),
        TextColumn(
            '[progress.description]{task.description}',
            style=style
        ),
        transient=True,
    ) as progress:
        progress.add_task(description=description, total=None)
        yield


def reproject_vector(
        raster: xr.DataArray,
        vector: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Reproject vector data if it has different CRS with raster data.

    Parameters
    ----------
    raster : xr.DataArray
        Raster data.
    vector : gpd.GeoDataFrame
        Vector data location containing point depth samples.

    Returns
    -------
    gpd.GeoDataFrame
        Reprojected vector data.
    """

    if raster.rio.crs is None:
        raise ValueError('Raster CRS is not defined.')
    if vector.crs is None:
        raise ValueError('Vector CRS is not defined.')

    # Retrieve CRS information from image and sample and change it to uppercase
    raster_crs = str(raster.rio.crs).upper()
    vector_crs = str(vector.crs).upper()

    # Check if CRS is the same and reproject sample if not
    if raster_crs != vector_crs:
        with progress_spinner(
            description='Reprojecting vector data...',
            style='yellow'
        ):
            new_vector = vector.to_crs(crs=raster_crs)
    else:
        new_vector = vector.copy()

    return new_vector


def clip_vector(
        raster: xr.DataArray, 
        vector: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Clip vector that is located outside raster boundary.

    Parameters
    ----------
    raster : xr.DataArray
        Raster data.
    vector : gpd.GeoDataFrame
        Vector data location containing point depth samples.

    Returns
    -------
    gpd.GeoDataFrame
        Clipped vector data.
    """

    # Check if vector has the same CRS as raster
    new_vector = reproject_vector(
        raster=raster,
        vector=vector
    )

    with progress_spinner(
        description='Excluding out-of-bounds points...',
        style='yellow'
    ):
        # Insert xarray image boundary coordinates to variables
        left, bottom, right, top = raster.rio.bounds()
        # Exclude out of boundary points
        new_vector = new_vector.cx[left:right, bottom:top]

    return new_vector


def point_sampling(
        raster: xr.DataArray,
        x: pd.Series | None = None,
        y: pd.Series | None = None,
        vector: gpd.GeoDataFrame | None = None,
        include_xy: bool = True,
        include_attributes: bool = False
) -> pd.DataFrame:
    """
    Extract raster values from a dataarray based on xy coordinates.
    XY coordinates have to be in the same CRS as raster.

    Parameters
    ----------
    raster : xr.DataArray
        DataArray from rioxarray.
    x : pd.Series | None
        X coordinates.
        Default is None, and must be provided together with y.
    y : pd.Series | None
        Y coordinates.
        Default is None, and must be provided together with x.
    vector : gpd.GeoDataFrame | None, optional
        GeoDataFrame containing points for sampling.
        Default is None, and must be provided if x and y are not provided.
    include_xy : bool, optional
        Whether to include the x and y coordinates in the output DataFrame.
        Default is True.
    include_attributes : bool, optional
        Whether to include attributes 
        from the input vector in the output DataFrame.
        Default is False. Only applicable if vector is provided.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the extracted raster values
        and optionally the x and y coordinates
        and attributes from the input vector.
    """

    if x is not None and y is not None:
        x_reindex = x.reset_index(drop=True)
        y_reindex = y.reset_index(drop=True)
    else:
        if vector is None:
            raise ValueError('Either x and y or vector must be provided')

        if not all(vector.geometry.type == 'Point'):
            raise ValueError('Input vector must contain point geometries only.')

        new_vector = clip_vector(
            raster=raster,
            vector=vector
        ).reset_index(drop=True)

        x_reindex = new_vector.geometry.x.reset_index(drop=True)
        y_reindex = new_vector.geometry.y.reset_index(drop=True)

        if include_attributes:
            attributes = new_vector.drop(columns='geometry').reset_index(drop=True)

    x_in = xr.DataArray(x_reindex, dims=['location'])
    y_in = xr.DataArray(y_reindex, dims=['location'])

    with progress_spinner(
        description='Extracting raster values from point locations...',
        style='yellow'
    ):
        point_samples = raster.sel(x=x_in, y=y_in, method='nearest').values.T

    point_samples_df = pd.DataFrame(
        point_samples,
        columns=[f'band_{i}' for i in raster.band.values]
    ).reset_index(drop=True)

    if include_xy:
        point_samples_df['x'], point_samples_df['y'] = x_reindex, y_reindex

    if include_attributes and vector is not None:
        point_samples_df = point_samples_df.join(attributes)

    return point_samples_df


def median_filter(
        array: NDArray,
        filter_size: int = 3
) -> NDArray:
    """
    Calculate median filter of a 3D array from rioxarray.
    The filter applied to each band separately.
    Band dimension is expected to be the first dimension of the array.

    Parameters
    ----------
    array : NDArray
        3D array data.
    filter_size : int, optional
        Size of the median filter window. Must be >= 3 and odd. Default is 3.
    Returns
    -------
    NDArray
        Filtered array.
    """

    if filter_size < 3 or filter_size % 2 == 0:
        raise ValueError('Allowed value: >= 3 and odd numbers')

    fs = (1, filter_size, filter_size)  # filter size for each dimension

    with progress_spinner(
        description='Applying median filter to raster data...',
        style='yellow'
    ):
        filtered = ndimage.median_filter(array, size=fs)

    return filtered


def array_to_dataarray(
        array: NDArray,
        data_array: xr.DataArray,
        band_name:  str | int = 1,
        attrs: bool = False
) -> xr.DataArray:
    """
    Create a new DataArray from a 2D Numpy array based on 
    rioxarray image specification but only contain 1 band.

    Parameters
    ----------
    array : NDArray
        Image data that read using rioxarray.
    data_array : xr.DataArray
        DataArray from rioxarray.
    band_name : str | int, optional
        A name for the band, by default 1.
    attrs : bool, optional
        Copy attributes, by default False.

    Returns
    -------
    xr.DataArray
        A DataArray with the same dimension and coordinates as input DataArray.
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

    if data_array.rio.crs is not None:
        new_da.rio.write_crs(data_array.rio.crs, inplace=True)

    return new_da