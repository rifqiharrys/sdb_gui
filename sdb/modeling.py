import pandas as pd
from joblib import parallel_backend
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor


def k_nearest_neighbors(
        unraveled_band: pd.DataFrame,
        features_train: pd.DataFrame,
        label_train: pd.Series,
        n_neighbors: int = 3,
        backend: str = 'threading',
        n_jobs: int = -2
):
    """
    Predicting depth using K-Nearest Neighbors.

    Parameter:
    ----------
    unraveled_band: pd.DataFrame
        Unraveled raster data.
    features_train: pd.DataFrame
        Features from train data.
    label_train: pd.Series
        Label from train data.
    n_neighbors: int = 3
        Number of neighbors in KNN.
    backend: str = 'threading'
    n_jobs: int = -2

    Result
    ------
    An array of predicted depth from trained model using unraveled raster data.
    """

    allowed_backend = {'loky', 'threading', 'multiprocessing'}
    if backend not in allowed_backend:
        raise ValueError(
            f'Invalid backend: {backend}.\n'
            f'Allowed: {allowed_backend}'
        )

    regressor = KNeighborsRegressor(n_neighbors=n_neighbors)

    with parallel_backend(backend=backend, n_jobs=n_jobs):
        regressor.fit(features_train, label_train)
        z_predict = regressor.predict(unraveled_band)

    return z_predict


def linear_regression(
        unraveled_band: pd.DataFrame,
        features_train: pd.DataFrame,
        label_train: pd.Series,
        fit_intercept: bool = True,
        copy_X: bool = True,
        backend: str = 'threading',
        n_jobs: int = -2
):
    """
    Predicting depth using Linear Regression.

    Parameter:
    ----------
    unraveled_band: pd.DataFrame
        Unraveled raster data.
    features_train: pd.DataFrame
        Features from train data.
    label_train: pd.Series 
        Label from train data.
    backend: str = 'threading'
    n_jobs: int = -2

    Result
    ------  
    An array of predicted depth from trained model using unraveled raster data.
    """

    allowed_backend = {'loky', 'threading', 'multiprocessing'}
    if backend not in allowed_backend:
        raise ValueError(
            f'Invalid backend: {backend}.\n'
            f'Allowed: {allowed_backend}'
        )

    regressor = LinearRegression(
        fit_intercept=fit_intercept,
        copy_X=copy_X
    )

    with parallel_backend(backend=backend, n_jobs=n_jobs):
        regressor.fit(features_train, label_train)
        z_predict = regressor.predict(unraveled_band)

    return z_predict


def random_forest(
        unraveled_band: pd.DataFrame,
        features_train: pd.DataFrame,
        label_train: pd.Series,
        ntree: int = 300,
        criterion: str = 'squared_error',
        bootstrap: bool = True,
        backend: str = 'threading',
        n_jobs: int = -2
):
    """
    Predicting depth using Random Forest.

    Parameter:
    ----------
    unraveled_band: pd.DataFrame
        Unraveled raster data.
    features_train: pd.DataFrame
        Features from train data.
    label_train: pd.Series
        Label from train data.
    ntree: int = 300
        Number of tree in Randomised Forest.
    criterion: str = 'squared_error'
    backend: str = 'threading'
    n_jobs: int = -2

    Result
    ------
    An array of predicted depth from trained model using unraveled raster data.
    """

    allowed_criterion = {
        'squared_error', 'absolute_error', 'friedman_mse', 'poisson'
    }
    if criterion not in allowed_criterion:
        raise ValueError(
            f'Invalid criterion: {criterion}.\n'
            f'Allowed: {allowed_criterion}'
        )

    allowed_backend = {'loky', 'threading', 'multiprocessing'}
    if backend not in allowed_backend:
        raise ValueError(
            f'Invalid backend: {backend}.\n'
            f'Allowed: {allowed_backend}'
        )

    regressor = RandomForestRegressor(
        n_estimators=ntree,
        criterion=criterion,
        bootstrap=bootstrap
    )

    with parallel_backend(backend=backend, n_jobs=n_jobs):
        regressor.fit(features_train, label_train)
        z_predict = regressor.predict(unraveled_band)

    return z_predict