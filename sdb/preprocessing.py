import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.model_selection import train_test_split

from .utils import point_sampling


def unravel(raster: xr.DataArray):
    """
    Unravel every band from rioxarray raster input to become a 1D array
    and stack it over every band in the form of columns.
    This function also changes values that potentially have issues
    in the upcoming process such as inf, -inf, and NaN to -999.0.

    Parameters
    ----------
    raster : xr.DataArray
        DataArray from rioxarray.

    Returns
    -------
    pd.DataFrame
        DataFrame with unraveled and stacked bands.
    """

    # Check raster size
    nbands = len(raster.band)
    ndata = raster.values[0].size

    # Create empty array based on raster size
    bands_array = np.empty((nbands, ndata))

    # Ravel arrays from each raster bands
    for i in range(nbands):
        bands_array[i, :] = np.ravel(raster.values[i])

    # Transpose the array
    bands_array = bands_array.T

    # Change inf and -inf values to nan (if any)
    bands_array[bands_array == np.inf] = np.nan
    bands_array[bands_array == -np.inf] = np.nan

    # Replace nan values with -999.0
    bands_array[np.isnan(bands_array)] = -999.0

    # Create dataframe from bands array
    bands_df = pd.DataFrame(bands_array, columns=[f'band_{i}' for i in raster.band.values])

    return bands_df


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

    # Retrieve CRS information from image and sample and change it to uppercase
    raster_crs = str(raster.rio.crs).upper()
    vector_crs = str(vector.crs).upper()

    # Check if CRS is the same and reproject sample if not
    if raster_crs != vector_crs:
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

    # Insert xarray image boundary coordinates to variables
    left, bottom, right, top = raster.rio.bounds()
    # Exclude out of boundary points
    new_vector = new_vector.cx[left:right, bottom:top]

    return new_vector


def in_depth_filter(
        vector: gpd.GeoDataFrame,
        header: str,
        depth_direction: str = 'up',
        disable_depth_filter: bool = False,
        top_limit: float = 0.0,
        bottom_limit: float = -12.0
) -> gpd.GeoDataFrame:
    """
    Change depth data in vector data to positive up and then filter it
    based on allowed depth in positive up direction.

    Parameters
    ----------
    vector : gpd.GeoDataFrame
        Vector data of depth points in GeoDataFrame type.
    header : str
        Header name of depth data.
    depth_direction : {'up', 'down'}
        Depth data direction either positive up ('up') or positive down ('down').
        Default value is 'up'.
    top_limit : float
        Top depth limit in positive up. Default value is 0.
    bottom_limit : float
        Bottom depth limit in positive up. Default value is 12.0.

    Returns
    -------
    pd.DataFrame
        Filtered depth data.
    """

    # Exchange value of top_limit and bottom_limit if top < bottom
    if top_limit < bottom_limit:
        top_limit, bottom_limit = bottom_limit, top_limit

    depth_direction_dict = {
        'up': False,
        'down': True
    }

    allowed_depth_direction = set(depth_direction_dict.keys())
    if depth_direction not in allowed_depth_direction:
        raise ValueError(
            f'Invalid depth direction: {depth_direction}.\n'
            f'Allowed: {allowed_depth_direction}'
        )

    # Change depth data direction to positive up
    if depth_direction_dict[depth_direction]:
        vector[header] *=-1

    if not disable_depth_filter:
        new_vector = vector[
            (vector[header] <= top_limit) & (vector[header] >= bottom_limit)
        ].reset_index(drop=True)

    return new_vector


def features_label(
        raster: xr.DataArray,
        vector: gpd.GeoDataFrame,
        header: str,
):
    """
    Extract raster values which are considered as features based on
    depth (label) points' xy position and combine it into one dataframe
    containing raster values from every bands in the raster, xy coordinates,
    and z or depth values.
    XY coordinates are included.

    Parameters
    ----------
    raster : xr.DataArray
        DataArray from rioxarray.
    vector : gpd.GeoDataFrame
        Vector data of depth points in GeoDataFrame type.
    header : str
        Header name of depth data.

    Returns
    -------
    pd.DataFrame
        A dataframe containing features and label.
    """

    x = vector.geometry.x
    y = vector.geometry.y
    z = vector[header]

    # Sampling image based on sample location
    df = point_sampling(raster, x, y)

    # Append depth data to the dataframe
    df['z'] = z

    # Delete rows with inf, -inf, and nan values
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    return df


def split_random(
        raster: xr.DataArray,
        vector: gpd.GeoDataFrame,
        header: str,
        train_size: float = 0.75,
        random_state: int = 0
):
    """
    Split train and test data randomly based on percentage.
    This process begins by point sampling every depth point, then separating
    features and label, and lastly splitting train and test data from features
    and label using train_test_split function from scikit-learn.
    XY coordinates are included.

    Parameters
    ----------
    raster : xr.DataArray
        DataArray from rioxarray.
    vector : gpd.GeoDataFrame
        Vector data of depth points in GeoDataFrame type.
    header : str
        Header name of depth data.
    train_size : float, optional
        Train data size, by default 0.75.
    random_state : int, optional
        Random state, by default 0.

    Returns
    -------
    Tuple
        A tuple containing train and test data.
    """

    df = features_label(raster, vector, header)
    features = df.drop(columns=['z'])
    z = df['z']

    features_train, features_test, z_train, z_test = train_test_split(
        features,
        z,
        train_size=train_size,
        random_state=random_state
    )

    return features_train, features_test, z_train, z_test


def split_attribute(
        raster: xr.DataArray,
        vector: gpd.GeoDataFrame,
        depth_header: str,
        split_header: str,
        group_name: str
):
    """
    Split train and test data based on assigned attribute.
    This process begins by separating train and test data based on attribute
    group, then point sampling every depth point from train and test data, and
    lastly separating features and label from train and test data.
    XY coordinates are included.

    Parameters
    ----------
    raster : xr.DataArray
        DataArray from rioxarray.
    vector : gpd.GeoDataFrame
        Vector data of depth points in GeoDataFrame type.
    depth_header : str
        Header name of depth data.
    split_header : str
        Header name of data that separates train and test data.
    group_name : str
        Group name that identifies the data as train data.

    Returns
    -------
    Tuple
        A tuple containing train and test data.
    """

    train = vector[vector[split_header] == group_name].reset_index(drop=True)
    test = vector[vector[split_header] != group_name].reset_index(drop=True)

    df_train = features_label(raster, train, depth_header)
    features_train, z_train = df_train.drop(columns=['z']), df_train['z']

    df_test = features_label(raster, test, depth_header)
    features_test, z_test = df_test.drop(columns=['z']), df_test['z']

    return features_train, features_test, z_train, z_test