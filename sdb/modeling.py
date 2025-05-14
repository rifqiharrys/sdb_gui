import pandas as pd
from joblib import parallel_backend
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor


def k_nearest_neighbors(
        unraveled_band: pd.DataFrame,
        features_train: pd.DataFrame,
        label_train: pd.Series,
        backend: str = 'threading',
        n_jobs: int = -2,
        **params
):
    """
    Predicting depth using K-Nearest Neighbors.

    Parameters
    ----------
    unraveled_band : pd.DataFrame
        Unraveled raster data.
    features_train : pd.DataFrame
        Features from train data.
    label_train : pd.Series
        Label from train data.
    backend : str, optional
        Backend to use for parallel processing. Default is 'threading'.
    n_jobs : int, optional
        The number of jobs to run in parallel. Default is -2.
    **params : dict
        Parameters to pass to KNeighborsRegressor such as
        n_neighbors, weights, algorithm, leaf_size, etc.
        See sklearn documentation for more details.

    Returns
    -------
    np.ndarray
        An array of predicted depth from trained model using unraveled raster data.
    """

    allowed_backend = {'loky', 'threading', 'multiprocessing'}
    if backend not in allowed_backend:
        raise ValueError(
            f'Invalid backend: {backend}.\n'
            f'Allowed: {allowed_backend}'
        )

    regressor = KNeighborsRegressor(**params)

    with parallel_backend(backend=backend, n_jobs=n_jobs):
        regressor.fit(features_train, label_train)
        z_predict = regressor.predict(unraveled_band)

    return z_predict


def linear_regression(
        unraveled_band: pd.DataFrame,
        features_train: pd.DataFrame,
        label_train: pd.Series,
        backend: str = 'threading',
        n_jobs: int = -2,
        **params
):
    """
    Predicting depth using Linear Regression.

    Parameters
    ----------
    unraveled_band : pd.DataFrame
        Unraveled raster data.
    features_train : pd.DataFrame
        Features from train data.
    label_train : pd.Series
        Label from train data.
    backend : str, optional
        Backend to use for parallel processing. Default is 'threading'.
    n_jobs : int, optional
        The number of jobs to run in parallel. Default is -2.
    **params : dict
        Parameters to pass to LinearRegression such as fit_intercept,
        copy_X, etc.
        See sklearn documentation for more details.

    Returns
    -------
    np.ndarray
        An array of predicted depth from trained model using unraveled raster data.
    """

    allowed_backend = {'loky', 'threading', 'multiprocessing'}
    if backend not in allowed_backend:
        raise ValueError(
            f'Invalid backend: {backend}.\n'
            f'Allowed: {allowed_backend}'
        )

    regressor = LinearRegression(**params)

    with parallel_backend(backend=backend, n_jobs=n_jobs):
        regressor.fit(features_train, label_train)
        z_predict = regressor.predict(unraveled_band)

    return z_predict


def random_forest(
        unraveled_band: pd.DataFrame,
        features_train: pd.DataFrame,
        label_train: pd.Series,
        backend: str = 'threading',
        n_jobs: int = -2,
        **params
):
    """
    Predicting depth using Random Forest.

    Parameters
    ----------
    unraveled_band : pd.DataFrame
        Unraveled raster data.
    features_train : pd.DataFrame
        Features from train data.
    label_train : pd.Series
        Label from train data.
    backend : str, optional
        Backend to use for parallel processing. Default is 'threading'.
    n_jobs : int, optional
        The number of jobs to run in parallel. Default is -2.
    **params : dict
        Parameters to pass to RandomForestRegressorsuch as n_estimators,
        criterion, bootstrap, etc.
        See sklearn documentation for more details.

    Returns
    -------
    np.ndarray
        An array of predicted depth from trained model using unraveled raster data.
    """

    allowed_backend = {'loky', 'threading', 'multiprocessing'}
    if backend not in allowed_backend:
        raise ValueError(
            f'Invalid backend: {backend}.\n'
            f'Allowed: {allowed_backend}'
        )

    regressor = RandomForestRegressor(**params)

    with parallel_backend(backend=backend, n_jobs=n_jobs):
        regressor.fit(features_train, label_train)
        z_predict = regressor.predict(unraveled_band)

    return z_predict