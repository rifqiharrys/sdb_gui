import geopandas as gpd
import pandas as pd
import rioxarray as rxr
import xarray as xr


def read_geotiff(raster_loc: str) -> xr.DataArray:
    """
    Read Geotiff raster data using Xarray (using rioxarray extension).

    Parameters
    ----------
    raster_loc : str
        Raster data location.

    Returns
    -------
    xr.DataArray
    """

    return rxr.open_rasterio(raster_loc, masked=True)


def read_shapefile(vector_loc: str) -> gpd.GeoDataFrame:
    """
    Read shapefile vector data containing depth samples using Geopandas.

    Parameters
    ----------
    vector_loc : str
        Vector data location containing point depth samples.

    Returns
    -------
    GeoDataFrame
    """

    return gpd.read_file(vector_loc)


def write_geotiff(
        raster: xr.DataArray,
        raster_loc: str
):
    """
    Write dataarray to Geotiff.

    Parameters
    ----------
    raster : xr.DataArray
        Raster data in dataarray.
    raster_loc : str
        Raster save data location.

    Returns
    -------
    None
    """

    raster.rio.to_raster(raster_loc)


def write_shapefile(
        table: pd.DataFrame,
        vector_loc: str,
        x_col_name: str,
        y_col_name: str,
        crs,
        z_col_name: str | None = None
):
    """
    Write dataframe to ESRI Shapefile.
    
    Parameters
    ----------
    table : pd.DataFrame
        A dataframe containing XY coordinates.
    vector_loc : str
        Vector save data location.
    x_col_name : str
        X coordinates column name.
    y_col_name : str
        Y coordinates column name.
    crs : Any
        Coordinate Reference System.
    z_col_name : str, optional
        Z coordinates column name, by default None.
    
    Returns
    -------
    None
    """

    x = table[x_col_name]
    y = table[y_col_name]
    z = table[z_col_name]

    if z is None:
        gdf = gpd.GeoDataFrame(
            table,
            geometry=gpd.points_from_xy(x, y),
            crs=crs
        )
    else:
        gdf = gpd.GeoDataFrame(
            table,
            geometry=gpd.points_from_xy(x, y, z),
            crs=crs
        )

    gdf.to_file(vector_loc)