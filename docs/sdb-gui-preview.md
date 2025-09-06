# SDB GUI Preview

This is a document that shows the SDB GUI interface and a little bit about what they do. The first picture below shows the main window. Inside the main window are 11 buttons, 3 dropdown menus, 1 table field, 2 numeric input fields, 1 check box, and 1 text field.

![main-window](./fig/main_window.png)

## Table of Contents

- [SDB GUI Preview](#sdb-gui-preview)
  - [Table of Contents](#table-of-contents)
  - [A. Buttons](#a-buttons)
    - [1. Load data](#1-load-data)
      - [a. Load image](#a-load-image)
      - [b. Load sample](#b-load-sample)
    - [2. Option buttons](#2-option-buttons)
      - [a. Method options](#a-method-options)
      - [b. Processing options](#b-processing-options)
      - [c. Reset settings](#c-reset-settings)
    - [3. Processing buttons](#3-processing-buttons)
      - [a. Generate prediction](#a-generate-prediction)
      - [b. Stop process](#b-stop-process)
    - [4. About buttons](#4-about-buttons)
      - [a. Releases](#a-releases)
      - [b. Licenses](#b-licenses)
      - [c. Readme](#c-readme)
  - [B. Dropdown menus](#b-dropdown-menus)
    - [1. Depth header](#1-depth-header)
    - [2. Depth direction](#2-depth-direction)
    - [3. Regression method](#3-regression-method)
  - [C. Table field](#c-table-field)
  - [D. Numeric inputs](#d-numeric-inputs)
    - [1. Upper limit](#1-upper-limit)
    - [2. Lower limit](#2-lower-limit)
  - [E. Check box](#e-check-box)
  - [F. Text field](#f-text-field)

## A. Buttons

### 1. Load data

#### a. Load image

Pushing the "Load Image" button will open a window that allows users to load a GeotTIFF image file. After the user push the "Load" button, the image filename will show up beside the "Load Image" button as the image is loaded.

![load-image](./fig/load_image_window.png)

#### b. Load sample

Pushing the "Load Sample" button will open a window that allows user to load an ESRI Shapefile data. By default, after the user push the "Load" button, the first 100 data will show up on the table field under depth header dropdown menu and the filename will show up beside the "Load Sample" button as the shapefile is loaded.

![load-sample](./fig/load_sample_window.png)

### 2. Option buttons

These buttons are related to options functionalities in general or related to each ML method.

#### a. Method options

Pictures below show every method option windows while selecting each regression method and then pushing the "Method Options" button.

![knn-option](./fig/knn_option_window.png) ![mlr-option](./fig/mlr_option_window.png) ![rf-option](./fig/rf_option_window.png)

#### b. Processing options

Pushing the "Processing Options" button will open a window that allows user to change the general settings on the data processing such as parallel backend, number of processing cores, howt to evaluate the result, and how to select train data from all the samples.

![processing-option](./fig/processing_option_window.png)

#### c. Reset settings

Pushing the "Reset Settings" button will open a confirmation window to reset all options to default and saved last directory.

![reset-setting](./fig/reset_setting_window.png)

### 3. Processing buttons

#### a. Generate prediction

Pushing this button will start the depth prediction process using the selected ML method and options.

#### b. Stop process

Pushing this button will stop the ongoing depth prediction process.

### 4. About buttons

#### a. Releases

Pushing this button will direct the user to the SDB GUI releases.

#### b. Licenses

Pushing this button will open a window that shows the SDB GUI license and another related libraries that are used in the SDB GUI.

![licenses](./fig/license_window.png)

#### c. Readme

Pushing this button will direct the user to the README page in GitHub.

## B. Dropdown menus

### 1. Depth header

### 2. Depth direction

The option from the list are Positive Up and Positive Down. The user must select the option that matches the depth data from the loaded sample data. Positive Up means that the data that has the positive sign is pointed to up direction. Positive down means that the data that has the positive sign is pointed to down direction.

### 3. Regression method

There are the ML methods available, which are K-Nearest Neighbors (KNN), Multiple Linear Regression (MLR), and Random Forest (RF).

## C. Table field

This field shows a table of the loaded depth sample data.

## D. Numeric inputs

### 1. Upper limit

### 2. Lower limit

## E. Check box

## F. Text field
