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

## Workflow

Image below is the workflow of predicting bathymetric depth using SDB GUI if you're running the latest [release](https://github.com/rifqiharrys/sdb_gui/releases) and the latest source code or release version 3 and up.

![workflow](workflow_sdb_gui.png "Workflow")

## Table of Contents

- [Satellite Derived Bathymetry (SDB) GUI](#satellite-derived-bathymetry-sdb-gui)
  - [Preface](#preface)
  - [Workflow](#workflow)
  - [Table of Contents](#table-of-contents)
  - [1. Setup and Preparation](#1-setup-and-preparation)
    - [a. Download executable file or source code](#a-download-executable-file-or-source-code)
    - [b. Python and packages installation](#b-python-and-packages-installation)
    - [c. Data preparation](#c-data-preparation)
  - [2. How To Use SDB GUI](#2-how-to-use-sdb-gui)
    - [a. Open SDB GUI and load data](#a-open-sdb-gui-and-load-data)
    - [b. Insert parameters and setting options](#b-insert-parameters-and-setting-options)
    - [c. Generate depth prediction](#c-generate-depth-prediction)
    - [d. Save depth prediction into file](#d-save-depth-prediction-into-file)
  - [3. Notebook](#3-notebook)
  - [Releases](#releases)
  - [License](#license)
  - [Citation](#citation)

## 1. Setup and Preparation

### a. Download executable file or source code

You can download the latest [release](https://github.com/rifqiharrys/sdb_gui/releases) or the latest source code from [sdb_gui](https://github.com/rifqiharrys/sdb_gui). If you are using the executable version, you can skip the python and packages installation steps and head to [Data Preparation](#c-data-preparation).

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

After loading depth sample data, you will notice a table loaded with said data and show the first 100 rows (or all depend on your chosen setting while loading data). Above the table are two selection about the loaded sample data. The first one is the header of the depth data, while the second one is the direction of the depth data.

Correctly selecting depth header and depth direction is important because it will allow SDB GUI to process the data correctly. The first selection is the column name or header name of the column of the depth data. The selection will show all of the header/column names of the sample data. The second selection will show two options, which are **Positive Up** and **Positive Down**. If your depth data decreases in values as it goes deeper, choose **Positive Up** because the positive values are going up, otherwise choose **Positive Down**.

The next parameters are depth limitation window for sample data input. There are two values for depth limit window, upper limit (default value is 0) and bottom limit (default value is -15). Both values are in the **Positive Up** direction manner. You could disable depth limitation by checking the Disable Depth Limitation checkbox.

Next, select your desired regression method. There are three options to select, which are K-Nearest Neighbors, Multiple Linear Regression, and Random Forest. For every regression method, you could change its hyperparameters by clicking the **Method Options** button. The explanation of every hyperparameter is in [scikit-learn user guide](https://scikit-learn.org/stable/user_guide.html).

Right below regression method selection is train data selection. It is how you want to select train data among loaded depth data. You could select either **Random Selection** or **Attribute Selection**. With **Random Selection**, train data will be selected randomly by your desired percentage (default 75%) and random state. Press **Processing Options** button to change this parameters. With **Attribute Selection**, you can select which attribute you want to use as train data by marking it with certain strings of your selection. If you want to use this option, **YOU HAVE TO** select which header name is the attribute of the marker and select the group of which it belongs. These options are available in **Processing Options**.

### c. Generate depth prediction

Generate depth prediction by pressing **Generate Prediction** button. While processing occurs, some information will be displayed under Result Information section. After the process completed, there will be a pop up alert showing the process is done. Any information regarding the processing will be displayed under Result Information section too.

### d. Save depth prediction into file

After depth prediction was generated, you can save it into a Geotiff or XYZ file. In the save file window, there are other options to use median filter to remove noise (default is on), save report, save train and test data, and create scatter plot using test data.

## 3. Notebook

To have a better understanding about the new SDB processing workflow in SDB GUI project, you could read a [Jupyter Notebook](./notebooks/) in this repository. There are two notebooks in this repository, which are `sdb-how-to-xarray-workflow.ipynb` and `sdb-module-how-to.ipynb`. Both notebooks contain a simple SDB processing workflow without GUI using Random Forest Regression.  The workflow of the first notebook might be a bit different from the recent SDB GUI, the basic idea is the same and this notebook is a prototype to the recent update (v4) of SDB GUI. The second notebook contain SDB processing using SDB module.

## Releases

See [RELEASES](https://github.com/rifqiharrys/sdb_gui/releases)

## License

See [LICENSE](./LICENSE)

## Citation

See [DOI](https://zenodo.org/doi/10.5281/zenodo.8220196)
