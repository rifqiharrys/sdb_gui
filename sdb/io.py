import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray as rxr


def read_geotiff(raster_loc: str) -> xr.DataArray:
    """
    Read Geotiff raster data using Xarray (using rioxarray extension).

    Parameter:
    ----------
    raster_loc: str
        Raster data location.

    Return
    ------
    xarray.DataArray
    """

    return rxr.open_rasterio(raster_loc, masked=True)


def read_shapefile(vector_loc: str) -> gpd.GeoDataFrame:
    """
    Read shapefile vector data containing depth samples using Geopandas.

    Parameter:
    ----------
    vector_loc: str
        Vector data location containing point depth samples.

    Return
    ------
    GeoDataFrame
    """

    return gpd.read_file(vector_loc)


def write_geotiff(
        raster: xr.DataArray,
        raster_loc: str
):
    """
    Write dataarray to Geotiff.

    Parameter:
    ----------
    raster: xr.DataArray
        Raster data in dataarray.
    raster_loc: str
        Raster save data location.
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

    Parameter:
    ----------
    table: pd.DataFrame
        A dataframe containing XY coordinates.
    vector_loc: str
        Vector save data location.
    x_col_name: str
        X coordinates column name.
    y_col_name: str
        Y coordinates column name.
    crs
        Coordinate Reference System.
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