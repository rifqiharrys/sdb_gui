# Satellite Derived Bathymetry (SDB) GUI

[![Last Commit](https://img.shields.io/github/last-commit/rifqiharrys/sdb_gui?color=red)](https://github.com/rifqiharrys/sdb_gui/commit)
[![Github Release](https://img.shields.io/github/v/release/rifqiharrys/sdb_gui?label=latest%20release&color=green)](https://github.com/rifqiharrys/sdb_gui/releases/latest)
[![DOI](https://zenodo.org/badge/309878273.svg)](https://zenodo.org/doi/10.5281/zenodo.8220196)

## Preface

Mainly, there are two methods to create a bathymetric prediction using satellite imagery. Two of which are analytical method and empirical method. The former predict depth using water body properties and calculate depth using some formula and those properties as variable input. The latter predict depth using depth training samples and fit the sample into some model and predict the depth using the model based on the depth sample training.

This SDB project is using python and its packages listed below:

|Packages|Description|
|---------|---------|
|[numpy](https://numpy.org/)|The fundamental package for scientific computing with Python. It provides functions for tasks such as array operations, matrix operations, and random number generation. It is the most widely used Python package for numerical computing. |
|[scipy](https://www.scipy.org/)|The SciPy library is one of the core packages that make up the SciPy stack. It provides functions for scientific and engineering applications. The SciPy library depends on NumPy, which provides support for large, multi-dimensional arrays and matrices, including a large collection of high-level mathematical functions to operate on these arrays. The SciPy library is built on top of the NumPy extension of the Python programming language. It adds functionality in several areas, including numerical integration, special functions, statistics, and optimization. |
|[pandas](https://pandas.pydata.org/)|Used for data manipulation and analysis. It offers data structures and functions to efficiently handle structured data, including tabular data such as spreadsheets and SQL tables. |
|[xarray](https://xarray.dev/)|The project integrates the array-orientated features of NumPy with the labeling features of Pandas. It provides a powerful and flexible way of working with labeled, multidimensional arrays. |
|[rioxarray](https://corteva.github.io/rioxarray)|Rioxarray is a Python package that enables the use of rasterio for xarray's raster-based operations. It provides an optional dependency for xarray, allowing it to read and write raster formats supported by rasterio. |
|[geopandas](https://geopandas.readthedocs.io/)|Geopandas extends the datatypes used by pandas to allow spatial operations on geometric types. It provides tools to read, write, and process geospatial data, making it easier to work with geographic datasets in Python.|
|[scikit-learn](https://scikit-learn.org)|A machine learning library for Python that features various classification, regression and clustering algorithms including support vector machines, random forests, gradient boosting, k-means and DBSCAN, and is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy. |
|[matplotlib](https://matplotlib.org/)|A plotting library for creating static, animated, and interactive visualizations in Python. |
|[pyqt5](https://www.riverbankcomputing.com/static/Docs/PyQt5/)|Used to create the GUI of this software. It is a set of Python bindings for Nokia's Qt application framework and runs on all platforms supported by Qt including Windows, OS X, Linux, iOS and Android. |

## 1. Setup and Preparation

### a. Download executable file or source code

You can download the latest [release](https://github.com/rifqiharrys/sdb_gui/releases) or the latest source code from [sdb_gui](https://github.com/rifqiharrys/sdb_gui). If you are using the executable version, you can skip the python and packages installation steps and head to...

### b. Python and packages installation

If you're downloading the source code, you need to have the packages from the table listed above installed. There are many ways to install them, but I prefer using [Miniconda](https://docs.anaconda.com/miniconda/) because of its little installation size. You could refer to [Miniconda installation instructions](https://docs.anaconda.com/miniconda/install/) on how to install miniconda. After conda was installed, open anaconda prompt and create new environment using conda create below.

```bash
# Replace <ENV_NAME> with a name for your environment
conda create --name <ENV_NAME>
```

To ensure that the packages installed are the latest version, install them from conda forge. To ensure the packages are installed from conda forge, add conda forge as priority channel.

```bash
conda config --add channels conda-forge
```

Then activate your new environment and install python 3.12 and the packages by typing prompts below.

```bash
conda activate <ENV_NAME>
conda install python=3.12 numpy scipy pandas xarray rioxarray geopandas scikit-learn matplotlib pyqt -y
```

### c. Data preparation

There are two types of data needed to use SDB GUI. They are:

1. Georeferenced and corrected imagery in GeoTIFF format
2. Depth samples in ESRI Shapefile format

The imagery required should be a multi-band imagery (e.g. RGB, RGBN, or others) in one stacked file. The depth samples doesn't have special requirements except if you want to process by attribute selection. In that case, the depth samples should have an additional field to differentiate the attributes which you want to select as training data, while the rest will marked as test data. Also, you need to understand the vertical reference and depth units of your depth samples because the results of prediction will be in the same units.

## 2. How To Use SDB GUI

### a. Open SDB GUI and load data

Open `sdb_gui_x.x.x_one_file.exe` if you're using the executable version or run `sdb_gui.py` if you're using the source code.

If you're using the executable version, you can open by double click on the file (`sdb_gui_x.x.x_one_file.exe`) or if you're using terminal, you can type `sdb_gui_x.x.x_one_file.exe` in the terminal in the same directory. If you're using the source code, run `sdb_gui.py` using python in your conda environment. Wait until SDB GUI opens.

Load your data into SDB GUI. When your data is successfully loaded, the GUI will show the file name beside the load buttons.

### b. Insert parameters and setting options

After loading depth sample data, you will notice a table loaded with said data and show the first 100 rows (or all depend your chosen setting while loading data). Above the table are two selection about the loaded sample data. The first one is the header of the depth data, while the second one is the direction of the depth data.

Correctly selecting depth header and depth direction is important because it will allow SDB GUI to process the data correctly. The first selection is the column name or header name of the column of the depth data. The selection will show all of the header/column names of the sample data. The second selection will show two options, which are **Positive Up** and **Positive Down**. If your depth data decreases in values as it goes deeper, choose **Positive Up** because the positive values are going up, otherwise choose **Positive Down**.

The next parameters are depth limitation window for sample data input. There are two values for depth limit window, upper limit (default value is 0) and bottom limit (default value is -30). Both values are in the **Positive Up** direction manner. You could disable depth limitation by checking the Disable Depth Limitation checkbox.

Next, select your desired regression method. There are three options to select, which are K-Nearest Neighbors, Multiple Linear Regression, and Random Forest. For every regression method, you could change its hyperparameters by clicking the **Method Options** button. The explanation of every hyperparameter is in [scikit-learn user guide](https://scikit-learn.org/stable/user_guide.html).

Right below regression method selection is train data selection. It is how you want to select train data among loaded depth data. You could select either **Random Selection** or **Attribute Selection**. With **Random Selection**, train data will be selected randomly by your desired percentage (default 75%) and random state. Press **Processing Options** button to change this parameters. With **Attribute Selection**, you can select which attribute you want to use as train data by marking it with certain strings of your selection. If you want to use this option, **YOU HAVE TO** select which header name is the attribute of the marker and select the group of which it belongs. These options are available in **Processing Options**.

### c. Generate depth prediction

Generate depth prediction by pressing **Generate Prediction** button. While processing occurs, some information will be displayed under Result Information section. After the process completed, there will be a pop up alert showing the process is done. Any information regarding the processing will be displayed under Result Information section too.

### d. Save depth prediction into file

After depth prediction was generated, you can save it into a Geotiff or XYZ file. You can also generate a report

## Workflow

Image below is the workflow of predicting bathymetric depth using SDB GUI if you're running the latest [release](https://github.com/rifqiharrys/sdb_gui/releases) and the latest source code or release version 3.x.x.

![workflow](workflow_sdb_gui.png "Workflow")

Inside SDB GUI Processing, the software first check if data inputs, which are raster data and depth samples have the same Coordinate Reference System (CRS). If they don't match with each other, the depth samples' CRS will reprojected into raster input reference system. And then, SDB GUI extracting each depth point samples coordinates and their respective raster value from raster input.

The next process is depth limit filtering. The depth limitation process is based on depth points as seamless land and water height points, so the software will automatically multiply all the depth sample points by `-1` if most of the depth sample values are positives. However, this could be turned off from `Processing Options` so the software will process the data as it is, but remember to adjust the depth limit to the original values.

When the depth samples is filtered, then it is separated into features and result so the machine learning library Scikit Learn know which are input variables and its corresponding results. And to test the resulting data, both features and result are splitted into train data and test data. The train data then used to train the selected model to fit the known results.

## Notebook

To have a better understanding about SDB processing workflow using SDB GUI v3, you could read a [Jupyter Notebook](https://github.com/rifqiharrys/sdb_gui/blob/main/notebooks/sdb.ipynb) in this repository. The notebook contain a simple SDB processing workflow without GUI using Random Forest Regression.

## Methods

There are four methods available make depth prediction using SDB GUI. All of which are Machine Learning methods that is available on [Scikit Learn](https://scikit-learn.org). The methods are [K-Nearest Neighbors](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html#sklearn.neighbors.KNeighborsRegressor "KNN Regressor"), [Multiple Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression "MLR Regression"), [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor "RF Regressor") and [Support Vector Machines](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR "SVM Regressor"). All of which are using [Scikit Learn](https://scikit-learn.org) module. Overall, Random Forest method has the tiniest RMSE value when using high number of sample. Meanwhile, Multiple Linear Regression is the fastest method, but usually resulting in the largest RMSE value.

### K-Nearest Neighbors

This method implements learning based on k nearest neighbors of each query point. The adjustable hyperparameters for this method are number of neighbors, weights, algorithm, and leaf size. The default values are 3, distance, auto, and 300.

### Multiple Linear Regression

In Scikit Learn modules, this method called only with the name Linear Regression. The 'Multiple' implies that the Linear Regression is used on multiple features as input.

### Random Forest

The adjustable hyperparameters for Random Forest method are the number of trees, the function to measure the quality of a split (criterion), bootstrap, and random state. The default values respectively are 300 and mse (Mean Square Error). The other value for the criterion is mae (Mean Absolute Error).

### Support Vector Machines

The adjustable hyperparameters for SVM method are kernel type, kernel coefficient (gamma), regularization parameter (C), and degree (which working for polynomial kernel only). The default hyperparameter values are rbf for kernel type, 0.1 for gamma, 1.0 for C, and 3 for degree.

## Features

SDB GUI has some features that helps making prediction and saving output data. These features are Depth Limitation, Median Filter, and Used Depth Samples output. User could disable these features when they are not needed.

### Depth Limitation

Visible light that comes from the sun and goes through sea surface will weaken as it goes into the water body. The maximum depth the visible light could penetrate into water body varies depend on its water properties. Depth Limitation will filter depth on input sample and prediction output by creating accepted depth window from zero depth until selected depth limit (default value is -30).

### Median Filter

Median Filter is an image filter that will clear outliers (salt-and-pepper noise) that seems out of place from the depth prediction process. The default value of Median Filter size is 3. The filter size value should only in odd numbers because the matrix size of odd numbers will always have one array as the center.

### Used Depth Samples

Create depth samples outputs that was used in data training and testing. The outputs are splitted train and test depth samples in Comma Separated Value or ESRI Shapefile. Those two outputs are containing sampled raster values, xy coordinates and depth values.

### Scatter Plot

Create scatter plot of the predicted result against its real value. The scatter plot saved in PNG file format.

## Releases

See [RELEASES](https://github.com/rifqiharrys/sdb_gui/releases)

## License

See [LICENSE](https://github.com/rifqiharrys/sdb_gui/blob/main/LICENSE)

## Citation

See [DOI](https://zenodo.org/doi/10.5281/zenodo.8220196)
