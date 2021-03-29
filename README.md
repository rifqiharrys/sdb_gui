# Satellite Derived Bathymetry (SDB) GUI
Mainly, there are two methods to create a bathymetric prediction using satellite imagery. Two of which are analytical method and empirical method. The former predict depth using water body properties and calculate depth using some formula and those properties as variable input. The latter predict depth using depth training samples and fit the sample into some model and predict the depth using the model based on the depth sample training.

## Getting Started

This is a GUI to make a bathimetric prediction using satellite imagery and some depth samples corresponding to the imagery. If you wish to use the GUI, please download the latest [release](https://github.com/rifqiharrys/sdb_gui/releases). Or if you want to run the source code (`sdb_gui.py`) instead, please install `python 3.6.x` and the following libraries first:

1. [Numpy](https://numpy.org/)
2. [Scipy](https://www.scipy.org/)
3. [Pandas](https://pandas.pydata.org/)
4. [Rasterio](https://rasterio.readthedocs.io/)
5. [Geopandas](https://geopandas.readthedocs.io/)
6. [Scikit Learn](https://scikit-learn.org)
7. [PyQt5](https://www.riverbankcomputing.com/static/Docs/PyQt5/)

Prepare your own data before using this SDB GUI. In general, there are two types of data needed in order to use SDB GUI, which are georeferenced and corrected imagery and depth samples. The image required should be multi-band imagery (e.g. RGB, RGBN, or others) and the GUI will automatically take all the image bands into processing. There are differences in how the depth samples should be prepared before loading when you're running SDB GUI version 2.x.x and below or running the latest source code and SDB GUI after version 2.x.x release.

If you're running SDB GUI version 2.x.x and below, the depth samples must be in tabular data consisting depth samples and corresponding raster values in the form of text file (e.g. CSV, TXT, or DAT file). If you don't have that kind of dwpth samples, you could extract it from your depth sample and the imagery using QGIS Plugin "Point Sampling Tool". If you're running the latest source code or SDB GUI after version 2.x.x release, you have to load your depth samples in the form of ESRI Shapefile format, then SDB GUI will sample the depth for you.

## How To Use

Open SDB GUI and load both data, and then select the header of your depth samples. Choose one of the methods and decide how much of the sample you're going to use as training data. If you push `Make Prediction` button right away, the software will use default hyperparameters. If you want to tweak the hyperparameters, push `Method Options` button and it will show you some changeable hyperparameters depends on which method is selected. Push `Processing Options` button to change some options on how to process like parallel backend, processing cores (n jobs), and random state. Note that SDB GUI will automatically change the depth sample values to negative by multiplying it to -1 if the data have more positive values. If you want the sample input unchanged, go to `Processing Option` and uncheck `Auto Negative Sign` and don't forget to adjust the depth limit window to the sample data.

After the prediction complete, you can save it into georeferenced raster file or XYZ ASCII file containing coordinates of each center of pixel. The prediction will show you depth values even on land. So, you have to mask the prediction result in the end and extracting prediction result of only water body.

## Workflow
Image below is the workflow of predicting bathymetric depth using SDB GUI if you're running the latest [release](https://github.com/rifqiharrys/sdb_gui/releases) and the latest source code or release version 3.x.x.

![workflow](workflow_sdb_gui.png "Workflow")

Inside SDB GUI Processing, the software first check if data inputs, which are raster data and depth samples have the same Coordinate Reference System (CRS). If they don't match with each other, the depth samples' CRS will reprojected into raster input reference system. And then, SDB GUI extracting each depth point samples coordinates and their respective raster value from raster input.

The next process is depth limit filtering. The depth limitation process is based on depth points as seamless land and water height points, so the software will automatically multiply all the depth sample points by `-1` if most of the depth sample values are positives. However, this could be turned off from `Processing Options` so the software will process the data as it is, but remember to adjust the depth limit to the original values.

When the depth samples is filtered, then it is separated into features and result so the machine learning library Scikit Learn know which are input variables and its corresponding results. And to test the resulting data, both features and result are splitted into train data and test data. The train data then used to train the selected model to fit the known results.

## Methods
In order to make depth prediction, there are four methods available. All of which are Machine Learning methods that is available on [Scikit Learn](https://scikit-learn.org) web page. The methods are [K-Nearest Neighbors](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html#sklearn.neighbors.KNeighborsRegressor "KNN Regressor"), [Multiple Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression "MLR Regression"), [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor "RF Regressor") and [Support Vector Machines](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR "SVM Regressor"). All of which are using [Scikit Learn](https://scikit-learn.org) module.
### K-Nearest Neighbors
This method implements learning based on k nearest neighbors of each query point. The adjustable hyperparameters for this method are number of neighbors, weights, algorithm, and leaf size. The default values are 3, distance, auto, and 300.

### Multiple Linear Regression
In Scikit Learn modules, this method called only with the name Linear Regression. The 'Multiple' implies that the Linear Regression is used on multiple features as input.

### Random Forest
The adjustable hyperparameters for Random Forest method are the number of trees, the function to measure the quality of a split (criterion), bootstrap, and random state. The default values respectively are 300 and mse (Mean Square Error). The other value for the criterion is mae (Mean Absolute Error).

### Support Vector Machines
The adjustable hyperparameters for SVM method are kernel type, kernel coefficient (gamma), regularization parameter (C), and degree (which working for polynomial kernel only). The default hyperparameter values are poly for kernel type, 0.1 for gamma, 1.0 for C, and 3 for degree.

## Features
SDB GUI has some features that helps in making prediction and saving output data. These features are Depth Limitation and Median Filter. User could disable one or both these features when they are not needed.

### Depth Limitation
Visible light that comes from the sun and goes through sea surface will weaken as it goes into the water body. The maximum depth the visible light could penetrate into water body varies depend on its water properties. Depth Limitation will filter depth on input sample and prediction output by creating accepted depth window from zero depth until selected depth limit (default value is -30).

### Median Filter
Median Filter is an image filter that will clear outliers (salt-and-pepper noise) that seems out of place from the depth prediction process. The default value of Median Filter size is 3. The filter size value should only in odd numbers because the matrix size of odd numbers will always have one array as the center.

## Releases
See [RELEASES](https://github.com/rifqiharrys/sdb_gui/releases)

## License
See [LICENSE](https://github.com/rifqiharrys/sdb_gui/blob/main/LICENSE)