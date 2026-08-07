from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import rioxarray as rxr
import xarray as xr
from pyproj.crs.crs import CRS


def read_geotiff(
        raster_loc: Path | str,
        **params: Any,
) -> xr.DataArray:
    """
    Read Geotiff raster data using `rioxarray.open_rasterio`.

    Parameters
    ----------
    raster_loc : Path | str
        Raster data location.
    **params : Any
        Additional parameters passed to `rioxarray.open_rasterio()`

    Returns
    -------
    xr.DataArray
    """

    return rxr.open_rasterio(raster_loc, masked=True, **params) # type: ignore


def read_shapefile(
        vector_loc: Path | str,
        **params: Any,
) -> gpd.GeoDataFrame:
    """
    Read vector data containing depth samples using `geopandas.read_file`.

    Parameters
    ----------
    vector_loc : Path | str
        Vector data location containing point depth samples.
    **params : Any
        Additional parameters passed to `geopandas.read_file()`

    Returns
    -------
    GeoDataFrame

    Raises
    ------
    ValueError
        If the file doesn't contain valid geometry data
    """
    gdf = gpd.read_file(vector_loc, **params)

    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError('Input file does not contain valid geometry data')

    return gdf


def write_geotiff(
        raster: xr.DataArray,
        raster_loc: Path | str,
        to_tif: bool = False,
        printout: bool = True,
        **params: Any,
) -> None:
    """
    Write dataarray to raster format using `rioxarray.DataArray.rio.to_raster`.
    See rioxarray documentation for supported formats.
    Geotiff is recommended for raster data.

    Parameters
    ----------
    raster : xr.DataArray
        Raster data in dataarray.
    raster_loc : Path | str
        Raster save data location.
    to_tif : bool, optional
        Whether to save the file with .tif extension.
        The raster will be written as Geotiff file if True,
        otherwise it will be saved with the provided extension.
        Default is False.
    printout : bool, optional
        Whether to print a message after saving the raster data, by default True.
    **params : Any
        Additional parameters passed to `rioxarray.DataArray.rio.to_raster()`

    Returns
    -------
    None
    """

    if to_tif:
        raster_loc = Path(raster_loc).with_suffix('.tif')

    raster.rio.to_raster(raster_loc, **params)

    if printout:
        print(f'Raster data saved to {raster_loc}')


def write_shapefile(
        table: pd.DataFrame,
        vector_loc: Path | str,
        x_col_name: str,
        y_col_name: str,
        crs: CRS | str | dict[str, Any] | None,
        z_col_name: str | None = None,
        printout: bool = True,
        **params: Any,
) -> None:
    """
    Write dataframe to vector format using `geopandas.GeoDataFrame.to_file`.
    See Geopandas documentation for supported formats.


    Parameters
    ----------
    table : pd.DataFrame
        A dataframe containing XY coordinates.
    vector_loc : Path | str
        Vector save data location.
    x_col_name : str
        X coordinates column name.
    y_col_name : str
        Y coordinates column name.
    crs : CRS | str | dict[str, Any]
        Coordinate Reference System as CRS object, string, or dictionary.
    z_col_name : str, optional
        Z coordinates column name, by default None.
    printout : bool, optional
        Whether to print a message after saving the vector data, by default True.
    **params : Any
        Additional parameters passed to `geopandas.GeoDataFrame.to_file()`

    Returns
    -------
    None
    """

    if crs is None:
        raise ValueError('CRS must be provided to save vector data')

    x = table[x_col_name]
    y = table[y_col_name]

    if z_col_name is None:
        geometry = gpd.points_from_xy(x, y)
    else:
        z = table[z_col_name]
        geometry = gpd.points_from_xy(x, y, z)

    gdf = gpd.GeoDataFrame(
        table,
        geometry=geometry,
        crs=crs
    )

    gdf.to_file(vector_loc, **params)

    if printout:
        print(f'Vector data saved to {vector_loc}')