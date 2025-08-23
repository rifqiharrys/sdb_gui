from typing import Any, Dict, Set, Tuple

import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor


def prediction(
        model: str,
        unraveled_band: pd.DataFrame,
        features_train: pd.DataFrame,
        label_train: pd.Series,
        features_test: pd.DataFrame | None = None,
        backend: str = 'threading',
        n_jobs: int = -2,
        **params: Any
) -> Tuple[np.ndarray, np.ndarray | None]:
    """
    Predicting depth using different models.

    Parameters
    ----------
    model : str
        The model to use for prediction. Options are 'knn', 'linear', or 'rf'.
        See model_alias_dict for more details.
    unraveled_band : pd.DataFrame
        Unraveled raster data.
    features_train : pd.DataFrame
        Features from train data.
    label_train : pd.Series
        Label from train data.
    features_test : pd.DataFrame | None, optional
        Features from test data.
    backend : str, optional
        Backend to use for parallel processing. Default is 'threading'.
    n_jobs : int, optional
        The number of jobs to run in parallel. Default is -2.
    **params : Dict[str, Union[str, int, float, bool]]
        Parameters to pass to the respective model.
        See sklearn documentation for more details.
        For KNeighborsRegressor: n_neighbors, weights, algorithm, leaf_size, etc.
        For LinearRegression: fit_intercept, copy_X, etc.
        For RandomForestRegressor: n_estimators, criterion, bootstrap, etc.

    Returns
    -------
    np.ndarray
        An array of predicted depth from trained model using unraveled raster data.
    """

    allowed_backend: Set[str] = {'loky', 'threading', 'multiprocessing'}
    if backend not in allowed_backend:
        raise ValueError(
            f'Invalid backend: {backend}.\n'
            f'Allowed: {allowed_backend}'
        )

    model_alias_dict: Dict[str, Set[str]] = {
        'knn': {
            'knn', 'k_nearest_neighbors', 'K-Nearest Neighbors'
        },
        'linear': {
            'mlr', 'linear', 'linear_regression', 'Multiple Linear Regression'
        },
        'rf': {
            'rf', 'random_forest', 'Random Forest'
        }
    }

    if model in model_alias_dict['knn']:
        regressor = KNeighborsRegressor(**params)
    elif model in model_alias_dict['linear']:
        regressor = LinearRegression(**params)
    elif model in model_alias_dict['rf']:
        regressor = RandomForestRegressor(**params)
    else:
        raise ValueError(
            f'Invalid model: {model}.\n'
            f'Allowed: {set.union(*model_alias_dict.values())}'
        )

    with parallel_backend(backend=backend, n_jobs=n_jobs):
        regressor.fit(features_train, label_train)
        z_predict = regressor.predict(unraveled_band)

        if features_test is not None:
            z_validate = regressor.predict(features_test)
        else:
            z_validate = None

    return z_predict, z_validate