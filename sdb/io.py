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
    Read Geotiff raster data using Xarray (using rioxarray extension).

    Parameters
    ----------
    raster_loc : Path | str
        Raster data location.
    **params : Any
        Additional parameters passed to rioxarray.open_rasterio()

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
    Read shapefile vector data containing depth samples using Geopandas.

    Parameters
    ----------
    vector_loc : Path | str
        Vector data location containing point depth samples.
    **params : Any
        Additional parameters passed to geopandas.read_file()

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
        raise ValueError('Input file does not contain valid geometry data')

    return gdf


def write_geotiff(
        raster: xr.DataArray,
        raster_loc: Path | str,
        to_tif: bool = False,
        **params: Any,
) -> None:
    """
    Write dataarray to Geotiff.

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
    **params : Any
        Additional parameters passed to rioxarray.DataArray.rio.to_raster()

    Returns
    -------
    None
    """

    if to_tif:
        raster_loc = Path(raster_loc).with_suffix('.tif')

    raster.rio.to_raster(raster_loc, **params)


def write_shapefile(
        table: pd.DataFrame,
        vector_loc: Path | str,
        x_col_name: str,
        y_col_name: str,
        crs: CRS | str | dict[str, Any],
        z_col_name: str | None = None,
        **params: Any,
) -> None:
    """
    Write dataframe to ESRI Shapefile.

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
    **params : Any
        Additional parameters passed to geopandas.GeoDataFrame.to_file()

    Returns
    -------
    None
    """

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