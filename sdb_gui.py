'''
MIT License

Copyright (c) 2020-present Rifqi Muhammad Harrys

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

'''
###############################################################################
################################# Main Imports ################################

from sklearn import metrics
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from joblib import parallel_backend
from scipy import ndimage
import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio as rio
from pathlib import Path
import sys, os
import datetime
import webbrowser
from PyQt5.QtCore import (Qt, QThread, pyqtSignal)
from PyQt5.QtWidgets import(QApplication, QWidget, QTextBrowser, QProgressBar, QFileDialog, QDialog,
                            QGridLayout, QPushButton, QVBoxLayout, QComboBox, QLabel, QCheckBox,
                            QDoubleSpinBox, QSpinBox, QTableWidgetItem, QTableWidget, QScrollArea,
                            QErrorMessage)
from PyQt5.QtGui import QIcon

###############################################################################
#################### For Auto PY to EXE or PyInstaller Use ####################

import sklearn.utils._weight_vector
import fiona._shim
import fiona.schema
import rasterio._features
import rasterio._shim
import rasterio.control
import rasterio.crs
import rasterio.sample
import rasterio.vrt

###############################################################################
###############################################################################

def resource_path(relative_path):
    '''Get the absolute path to the resource, works for dev and for PyInstaller'''
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)



class SDBWidget(QWidget):
    '''
    PyQt5 widget of SDB GUI
    '''

    widget_signal = pyqtSignal(dict)

    def __init__(self):
        '''
        Initialize widget and default values
        '''

        super(SDBWidget, self).__init__()

        ####### Default Values #######

        global method_dict
        method_dict = {
            'K-Nearest Neighbors': self.knnOptionWindow,
            'Multiple Linear Regression': self.mlrOptionWindow,
            'Random Forest': self.rfOptionWIndow, 
            'Support Vector Machines': self.svmOptionWindow
        }

        global proc_op_dict
        proc_op_dict = {
            'backend': 'threading',
            'n_jobs': -2,
            'random_state': 0,
            'auto_negative': 'checked'
        }

        global knn_op_dict
        knn_op_dict = {
            'n_neighbors': 5,
            'weights': 'distance',
            'algorithm': 'auto',
            'leaf_size': 30
        }

        global mlr_op_dict
        mlr_op_dict = {
            'fit_intercept': True,
            'normalize': False,
            'copy_x': True
        }

        global rf_op_dict
        rf_op_dict = {
            'n_estimators': 300,
            'criterion': 'mse',
            'bootstrap': True,
            'random_state': 0
        }

        global svm_op_dict
        svm_op_dict = {
            'kernel': 'poly',
            'gamma': .1,
            'c': 1000.0,
            'degree': 3
        }

        ####### Default Values #######

        self.initUI()


    def initUI(self):
        '''
        Initialize User Interface for SDB GUI Widget
        '''

        self.setGeometry(300, 100, 480, 640)
        self.setWindowTitle('Satellite Derived Bathymetry (v3.1.0)')
        self.setWindowIcon(QIcon(resource_path('icons/satellite.png')))

        loadImageButton = QPushButton('Load Image')
        loadImageButton.clicked.connect(self.loadImageWindow)
        self.loadImageLabel = QLabel()
        self.loadImageLabel.setText('No Image Loaded')
        self.loadImageLabel.setAlignment(Qt.AlignCenter)

        loadSampleButton = QPushButton('Load Sample')
        loadSampleButton.clicked.connect(self.loadSampleWindow)
        self.loadSampleLabel = QLabel()
        self.loadSampleLabel.setText('No Sample Loaded')
        self.loadSampleLabel.setAlignment(Qt.AlignCenter)

        depthHeaderLabel = QLabel('Depth Header:')
        self.depthHeaderCB = QComboBox()

        self.table = QTableWidget()
        scroll = QScrollArea()
        scroll.setWidget(self.table)

        limitLabel = QLabel('Depth Limit Window:')

        limitALabel = QLabel('Upper Limit:')
        self.limitADSB = QDoubleSpinBox()
        self.limitADSB.setRange(-100, 100)
        self.limitADSB.setDecimals(1)
        self.limitADSB.setValue(0)
        self.limitADSB.setSuffix(' m')
        self.limitADSB.setAlignment(Qt.AlignRight)

        limitBLabel = QLabel('Bottom Limit:')
        self.limitBDSB = QDoubleSpinBox()
        self.limitBDSB.setRange(-100, 100)
        self.limitBDSB.setDecimals(1)
        self.limitBDSB.setValue(-30)
        self.limitBDSB.setSuffix(' m')
        self.limitBDSB.setAlignment(Qt.AlignRight)

        self.limitCheckBox = QCheckBox('Disable Depth Limitation')
        self.limitCheckBox.setChecked(False)
        self.limitState = QLabel('unchecked')
        self.limitCheckBox.toggled.connect(
            lambda: self.checkBoxState(
                check_box=self.limitCheckBox,
                status=self.limitState
            )
        )

        method_list = list(method_dict)

        methodLabel = QLabel('Regression Method:')
        self.methodCB = QComboBox()
        self.methodCB.addItems(method_list)
        self.methodCB.activated.connect(
            lambda: self.methodSelection(
                option=method_dict[self.methodCB.currentText()]
            )
        )

        trainPercentLabel = QLabel('Train Data (Percent):')
        self.trainPercentDSB = QDoubleSpinBox()
        self.trainPercentDSB.setRange(10.0, 90.0)
        self.trainPercentDSB.setDecimals(2)
        self.trainPercentDSB.setValue(75.0)
        self.trainPercentDSB.setSuffix(' %')
        self.trainPercentDSB.setAlignment(Qt.AlignRight)

        self.optionsButton = QPushButton('Method Options')
        self.optionsButton.clicked.connect(self.knnOptionWindow)

        makePredictionButton = QPushButton('Make Prediction')
        makePredictionButton.clicked.connect(self.predict)
        saveFileButton = QPushButton('Save Into File')
        saveFileButton.clicked.connect(self.saveOptionWindow)

        processingOptionsButton = QPushButton('Processing Options')
        processingOptionsButton.clicked.connect(self.processingOptionWindow)

        resultInfo = QLabel('Result Information')
        self.resultText = QTextBrowser()
        self.resultText.setAlignment(Qt.AlignRight)

        self.progressBar = QProgressBar()
        self.progressBar.setFormat('%p%')
        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(6)

        releaseButton =  QPushButton('Releases')
        releaseButton.clicked.connect(lambda: webbrowser.open(
            'https://github.com/rifqiharrys/sdb_gui/releases'
        ))

        licensesButton = QPushButton('Licenses')
        licensesButton.clicked.connect(self.licensesDialog)

        readmeButton = QPushButton('Readme')
        readmeButton.clicked.connect(lambda: webbrowser.open(
            'https://github.com/rifqiharrys/sdb_gui/blob/main/README.md'
        ))

        grid = QGridLayout()
        vbox = QVBoxLayout()

        grid.addWidget(loadImageButton, 1, 1, 1, 2)
        grid.addWidget(self.loadImageLabel, 1, 3, 1, 2)

        grid.addWidget(loadSampleButton, 2, 1, 1, 2)
        grid.addWidget(self.loadSampleLabel, 2, 3, 1, 2)

        grid.addWidget(depthHeaderLabel, 3, 1, 1, 1)
        grid.addWidget(self.depthHeaderCB, 3, 2, 1, 3)

        grid.addWidget(self.table, 5, 1, 5, 4)

        grid.addWidget(limitLabel, 10, 1, 1, 2)
        grid.addWidget(self.limitCheckBox, 11, 1, 1, 2)

        grid.addWidget(limitALabel, 10, 3, 1, 1)
        grid.addWidget(self.limitADSB, 10, 4, 1, 1)
        grid.addWidget(limitBLabel, 11, 3, 1, 1)
        grid.addWidget(self.limitBDSB, 11, 4, 1, 1)

        grid.addWidget(methodLabel, 12, 1, 1, 1)
        grid.addWidget(self.methodCB, 12, 2, 1, 1)

        grid.addWidget(self.optionsButton, 12, 3, 1, 2)

        grid.addWidget(trainPercentLabel, 13, 1, 1, 1)
        grid.addWidget(self.trainPercentDSB, 13, 2, 1, 1)

        grid.addWidget(processingOptionsButton, 13, 3, 1, 2)

        grid.addWidget(makePredictionButton, 14, 1, 1, 2)
        grid.addWidget(saveFileButton, 14, 3, 1, 2)

        grid.addWidget(resultInfo, 15, 1, 1, 2)
        grid.addWidget(self.resultText, 16, 1, 1, 4)

        vbox.addStretch(1)
        grid.addLayout(vbox, 21, 1)

        grid.addWidget(self.progressBar, 22, 1, 1, 4)

        grid.addWidget(releaseButton, 23, 1, 1, 1)
        grid.addWidget(licensesButton, 23, 2, 1, 2)
        grid.addWidget(readmeButton, 23, 4, 1, 1)
        self.setLayout(grid)


    def str2bool(self, v):
        '''
        Transform string True or False to boolean type
        '''

        return v in ('True')


    def checkBoxState(self, check_box, status):
        '''
        Saving a checkbox status into a QLabel object whether
        it's checked or unchecked
        '''

        if check_box.isChecked() == True:
            status.setText('checked')
        else:
            status.setText('unchecked')


    def fileDialog(self, command, window_text, file_type, text_browser):
        '''
        Showing file dialog, whether opening file or saving.
        '''

        home_dir = str(Path.home())
        fileFilter = 'All Files (*.*) ;; ' + file_type
        selectedFilter = file_type
        fname = command(self, window_text, home_dir, fileFilter, selectedFilter)

        text_browser.setText(fname[0])


    def methodSelection(self, option):
        '''
        Method selection connection from option button
        to each methods' option window
        '''

        self.optionsButton.clicked.disconnect()
        self.optionsButton.clicked.connect(option)


    def loadImageWindow(self):
        '''
        Image loading User Interface
        '''

        self.loadImageDialog = QDialog()
        self.loadImageDialog.setWindowTitle('Load Image')
        self.loadImageDialog.setWindowIcon(QIcon(resource_path('icons/load-pngrepo-com.png')))

        openFilesButton = QPushButton('Open File')
        openFilesButton.clicked.connect(
            lambda: self.fileDialog(
                command=QFileDialog.getOpenFileName,
                window_text='Open Image File',
                file_type='GeoTIFF (*.tif)',
                text_browser=self.imglocList
            )
        )

        locLabel = QLabel('Location:')
        self.imglocList = QTextBrowser()

        cancelButton = QPushButton('Cancel')
        cancelButton.clicked.connect(self.loadImageDialog.close)
        loadButton = QPushButton('Load')
        loadButton.clicked.connect(self.loadImageAction)
        loadButton.clicked.connect(self.loadImageDialog.close)

        grid = QGridLayout()
        grid.addWidget(openFilesButton, 1, 1, 1, 4)

        grid.addWidget(locLabel, 4, 1, 1, 1)

        grid.addWidget(self.imglocList, 5, 1, 10, 4)

        grid.addWidget(loadButton, 15, 3, 1, 1)
        grid.addWidget(cancelButton, 15, 4, 1, 1)

        self.loadImageDialog.setLayout(grid)

        self.loadImageDialog.exec_()


    def loadImageAction(self):
        '''
        Loading selected image and retrieve some metadata such as file size,
        band quantity, array size, pixel size, etc. Then, recreate image 3D
        array into a simple column and row array.
        '''

        try:
            global img_size
            img_size = os.path.getsize(self.imglocList.toPlainText())

            global image_raw
            image_raw = rio.open(self.imglocList.toPlainText())

            nbands = len(image_raw.indexes)
            ndata = image_raw.read(1).size
            bands_dummy = np.empty((nbands, ndata))

            for i in range(1, nbands + 1):
                bands_dummy[i - 1, :] = np.ravel(image_raw.read(i))

            global bands_array
            bands_array = bands_dummy.T

            self.loadImageLabel.setText(
                os.path.split(self.imglocList.toPlainText())[1]
            )
            print(image_raw.crs)
        except:
            self.loadImageDialog.close()
            self.warningWithClear(
                'No data loaded. Please load your data!'
            )
            self.loadImageWindow()


    def loadSampleWindow(self):
        '''
        Sample loading User Interface
        '''

        self.loadSampleDialog = QDialog()
        self.loadSampleDialog.setWindowTitle('Load Sample')
        self.loadSampleDialog.setWindowIcon(QIcon(resource_path('icons/load-pngrepo-com.png')))

        openFilesButton = QPushButton('Open File')
        openFilesButton.clicked.connect(
            lambda: self.fileDialog(
                command=QFileDialog.getOpenFileName,
                window_text='Open Depth Sample File',
                file_type='ESRI Shapefile (*.shp)',
                text_browser=self.samplelocList
            )
        )

        locLabel = QLabel('Location:')
        self.samplelocList = QTextBrowser()

        self.showCheckBox = QCheckBox('Show All Data to Table')
        self.showCheckBox.setChecked(False)
        self.showState = QLabel('unchecked')
        self.showCheckBox.toggled.connect(
            lambda: self.checkBoxState(
                check_box=self.showCheckBox,
                status=self.showState
            )
        )

        cancelButton = QPushButton('Cancel')
        cancelButton.clicked.connect(self.loadSampleDialog.close)
        loadButton = QPushButton('Load')
        loadButton.clicked.connect(self.loadSampleAction)
        loadButton.clicked.connect(self.loadSampleDialog.close)

        grid = QGridLayout()
        grid.addWidget(openFilesButton, 1, 1, 1, 4)

        grid.addWidget(locLabel, 4, 1, 1, 1)

        grid.addWidget(self.samplelocList, 5, 1, 10, 4)

        grid.addWidget(self.showCheckBox, 15, 1, 1, 2)
        grid.addWidget(loadButton, 15, 3, 1, 1)
        grid.addWidget(cancelButton, 15, 4, 1, 1)

        self.loadSampleDialog.setLayout(grid)

        self.loadSampleDialog.exec_()


    def loadSampleAction(self):
        '''
        Loading selected sample and retrieve file size.
        Then, some or all data on selected sample to the widget.
        '''

        try:
            global sample_size
            sample_size = os.path.getsize(self.samplelocList.toPlainText())

            global sample_raw
            sample_raw = gpd.read_file(self.samplelocList.toPlainText())

            self.loadSampleLabel.setText(os.path.split(
                self.samplelocList.toPlainText())[1]
            )

            if (sample_raw.geom_type != 'Point').any():
                del sample_raw
                self.loadSampleLabel.setText('Sample Retracted')
                self.depthHeaderCB.clear()
                self.table.clearContents()

                self.loadSampleDialog.close()
                self.warningWithoutClear(
                    'Your data is not Point type. Please load another data!'
                )
                self.loadSampleWindow()
            else:
                raw = sample_raw.copy()

                if self.showState.text() == 'checked':
                    data = raw
                else:
                    data = raw.head(100)

                self.depthHeaderCB.clear()
                self.depthHeaderCB.addItems(data.columns)

                self.table.setColumnCount(len(data.columns))
                self.table.setRowCount(len(data.index))

                for h in range(len(data.columns)):
                    self.table.setHorizontalHeaderItem(h, QTableWidgetItem(data.columns[h]))

                for i in range(len(data.index)):
                    for j in range(len(data.columns)):
                        self.table.setItem(i, j, QTableWidgetItem(str(data.iloc[i, j])))

                self.table.resizeColumnsToContents()
                self.table.resizeRowsToContents()

                print(sample_raw.crs)
        except:
            self.loadSampleDialog.close()
            self.warningWithClear(
                'No data loaded. Please load your data!'
            )
            self.loadSampleWindow()


    def knnOptionWindow(self):
        '''
        K-Nearest Neighbor option User Interface
        '''

        optionDialog = QDialog()
        optionDialog.setWindowTitle('Options (K Neighbors)')
        optionDialog.setWindowIcon(QIcon(resource_path('icons/setting-tool-pngrepo-com.png')))

        nneighborLabel = QLabel('Number of Neighbors:')
        self.nneighborSB = QSpinBox()
        self.nneighborSB.setRange(1, 1000)
        self.nneighborSB.setValue(5)
        self.nneighborSB.setAlignment(Qt.AlignRight)

        weightsLabel = QLabel('Weights:')
        self.weightsCB = QComboBox()
        self.weightsCB.addItems(['uniform', 'distance'])
        self.weightsCB.setCurrentIndex(1)

        algorithmLabel = QLabel('Algorithm:')
        self.algorithmCB = QComboBox()
        self.algorithmCB.addItems(['auto', 'ball_tree', 'kd_tree', 'brute'])

        leafSizeLabel = QLabel('Leaf Size:')
        self.leafSizeSB = QSpinBox()
        self.leafSizeSB.setRange(1, 1000)
        self.leafSizeSB.setValue(30)
        self.leafSizeSB.setAlignment(Qt.AlignRight)

        cancelButton = QPushButton('Cancel')
        cancelButton.clicked.connect(optionDialog.close)
        loadButton = QPushButton('Load')
        loadButton.clicked.connect(self.loadKNNOptionAction)
        loadButton.clicked.connect(optionDialog.close)

        grid = QGridLayout()

        grid.addWidget(nneighborLabel, 1, 1, 1, 2)
        grid.addWidget(self.nneighborSB, 1, 3, 1, 2)

        grid.addWidget(weightsLabel, 2, 1, 1, 2)
        grid.addWidget(self.weightsCB, 2, 3, 1, 2)

        grid.addWidget(algorithmLabel, 3, 1, 1, 2)
        grid.addWidget(self.algorithmCB, 3, 3, 1, 2)

        grid.addWidget(leafSizeLabel, 4, 1, 1, 2)
        grid.addWidget(self.leafSizeSB, 4, 3, 1, 2)

        grid.addWidget(loadButton, 5, 3, 1, 1)
        grid.addWidget(cancelButton, 5, 4, 1, 1)

        optionDialog.setLayout(grid)

        optionDialog.exec_()


    def loadKNNOptionAction(self):
        '''
        Loading defined KNN option input
        '''

        knn_op_dict['n_neighbors'] = self.nneighborSB.value()
        knn_op_dict['weights'] = self.weightsCB.currentText()
        knn_op_dict['algorithm'] = self.algorithmCB.currentText()
        knn_op_dict['leaf_size'] = self.leafSizeSB.value()


    def mlrOptionWindow(self):
        '''
        Multi Linear Regression option User Interface
        '''

        optionDialog = QDialog()
        optionDialog.setWindowTitle('Options (MLR)')
        optionDialog.setWindowIcon(QIcon(resource_path('icons/setting-tool-pngrepo-com.png')))

        fitInterceptLabel = QLabel('Fit Intercept:')
        self.fitInterceptCB = QComboBox()
        self.fitInterceptCB.addItems(['True', 'False'])

        normalizeLabel = QLabel('Normalize:')
        self.normalizeCB = QComboBox()
        self.normalizeCB.addItems(['True', 'False'])
        self.normalizeCB.setCurrentIndex(1)

        copyXLabel = QLabel('Copy X:')
        self.copyXCB = QComboBox()
        self.copyXCB.addItems(['True', 'False'])

        cancelButton = QPushButton('Cancel')
        cancelButton.clicked.connect(optionDialog.close)
        loadButton = QPushButton('Load')
        loadButton.clicked.connect(self.loadMLROptionAction)
        loadButton.clicked.connect(optionDialog.close)

        grid = QGridLayout()

        grid.addWidget(fitInterceptLabel, 1, 1, 1, 2)
        grid.addWidget(self.fitInterceptCB, 1, 3, 1, 2)

        grid.addWidget(normalizeLabel, 2, 1, 1, 2)
        grid.addWidget(self.normalizeCB, 2, 3, 1, 2)

        grid.addWidget(copyXLabel, 3, 1, 1, 2)
        grid.addWidget(self.copyXCB, 3, 3, 1, 2)

        grid.addWidget(loadButton, 4, 3, 1, 1)
        grid.addWidget(cancelButton, 4, 4, 1, 1)

        optionDialog.setLayout(grid)

        optionDialog.exec_()


    def loadMLROptionAction(self):
        '''
        Loading defined MLR option input
        '''

        mlr_op_dict['fit_intercept'] = self.str2bool(self.fitInterceptCB.currentText())
        mlr_op_dict['normalize'] = self.str2bool(self.normalizeCB.currentText())
        mlr_op_dict['copy_x'] = self.str2bool(self.copyXCB.currentText())


    def rfOptionWIndow(self):
        '''
        Random Forest option User Interface
        '''

        optionDialog = QDialog()
        optionDialog.setWindowTitle('Options (Random Forest)')
        optionDialog.setWindowIcon(QIcon(resource_path('icons/setting-tool-pngrepo-com.png')))

        ntreeLabel = QLabel('Number of Trees:')
        self.ntreeSB = QSpinBox()
        self.ntreeSB.setRange(1, 10000)
        self.ntreeSB.setValue(300)
        self.ntreeSB.setAlignment(Qt.AlignRight)

        criterionLabel = QLabel('Criterion:')
        self.criterionCB = QComboBox()
        self.criterionCB.addItems(['mse', 'mae'])

        bootstrapLabel = QLabel('Bootstrap:')
        self.bootstrapCB = QComboBox()
        self.bootstrapCB.addItems(['True', 'False'])

        randomStateLabel = QLabel('Random State:')
        self.randomStateRFSB = QSpinBox()
        self.randomStateRFSB.setRange(0, 1000)
        self.randomStateRFSB.setValue(0)
        self.randomStateRFSB.setAlignment(Qt.AlignRight)

        cancelButton = QPushButton('Cancel')
        cancelButton.clicked.connect(optionDialog.close)
        loadButton = QPushButton('Load')
        loadButton.clicked.connect(self.loadRFOptionAction)
        loadButton.clicked.connect(optionDialog.close)

        grid = QGridLayout()

        grid.addWidget(ntreeLabel, 1, 1, 1, 2)
        grid.addWidget(self.ntreeSB, 1, 3, 1, 2)

        grid.addWidget(criterionLabel, 2, 1, 1, 2)
        grid.addWidget(self.criterionCB, 2, 3, 1, 2)

        grid.addWidget(bootstrapLabel, 3, 1, 1, 2)
        grid.addWidget(self.bootstrapCB, 3, 3, 1, 2)

        grid.addWidget(randomStateLabel, 4, 1, 1, 2)
        grid.addWidget(self.randomStateRFSB, 4, 3, 1, 2)

        grid.addWidget(loadButton, 5, 3, 1, 1)
        grid.addWidget(cancelButton, 5, 4, 1, 1)

        optionDialog.setLayout(grid)

        optionDialog.exec_()


    def loadRFOptionAction(self):
        '''
        Loading defined RF option input
        '''

        rf_op_dict['n_estimators'] = self.ntreeSB.value()
        rf_op_dict['criterion'] = self.criterionCB.currentText()
        rf_op_dict['bootstrap'] = self.str2bool(self.bootstrapCB.currentText())
        rf_op_dict['random_state'] = self.randomStateRFSB.value()


    def svmOptionWindow(self):
        '''
        Support Vector Machine option User Interface
        '''

        optionDialog = QDialog()
        optionDialog.setWindowTitle('Options (SVM)')
        optionDialog.setWindowIcon(QIcon(resource_path('icons/setting-tool-pngrepo-com.png')))

        kernelLabel = QLabel('Kernel:')
        self.kernelCB = QComboBox()
        self.kernelCB.addItems(['linear', 'poly', 'rbf', 'sigmoid'])
        self.kernelCB.setCurrentIndex(1)

        gammaLabel = QLabel('Gamma:')
        self.gammaDSB = QDoubleSpinBox()
        self.gammaDSB.setRange(0, 10)
        self.gammaDSB.setDecimals(3)
        self.gammaDSB.setValue(.1)
        self.gammaDSB.setAlignment(Qt.AlignRight)

        cLabel = QLabel('C:')
        self.cDSB = QDoubleSpinBox()
        self.cDSB.setRange(.001, 10000)
        self.cDSB.setDecimals(3)
        self.cDSB.setValue(1000.0)
        self.cDSB.setAlignment(Qt.AlignRight)

        degreeLabel = QLabel('degree (poly):')
        self.degreeSB = QSpinBox()
        self.degreeSB.setRange(2, 20)
        self.degreeSB.setValue(3)
        self.degreeSB.setAlignment(Qt.AlignRight)

        cancelButton = QPushButton('Cancel')
        cancelButton.clicked.connect(optionDialog.close)
        loadButton = QPushButton('Load')
        loadButton.clicked.connect(self.loadSVMOptionAction)
        loadButton.clicked.connect(optionDialog.close)

        grid = QGridLayout()

        grid.addWidget(kernelLabel, 1, 1, 1, 2)
        grid.addWidget(self.kernelCB, 1, 3, 1, 2)

        grid.addWidget(gammaLabel, 2, 1, 1, 2)
        grid.addWidget(self.gammaDSB, 2, 3, 1, 2)

        grid.addWidget(cLabel, 3, 1, 1, 2)
        grid.addWidget(self.cDSB, 3, 3, 1, 2)

        grid.addWidget(degreeLabel, 4, 1, 1, 2)
        grid.addWidget(self.degreeSB, 4, 3, 1, 2)

        grid.addWidget(loadButton, 5, 3, 1, 1)
        grid.addWidget(cancelButton, 5, 4, 1, 1)

        optionDialog.setLayout(grid)

        optionDialog.exec_()


    def loadSVMOptionAction(self):
        '''
        Loading defined SVM option input
        '''

        svm_op_dict['kernel'] = self.kernelCB.currentText()
        svm_op_dict['gamma'] = self.gammaDSB.value()
        svm_op_dict['c'] = self.cDSB.value()
        svm_op_dict['degree'] = self.degreeSB.value()


    def processingOptionWindow(self):
        '''
        Processing option User Interface
        '''

        self.processingOptionDialog = QDialog()
        self.processingOptionDialog.setWindowTitle('Processing Options')
        self.processingOptionDialog.setWindowIcon(QIcon(resource_path('icons/setting-tool-pngrepo-com.png')))

        backendLabel = QLabel('Parallel Backend:')
        self.backendCB = QComboBox()
        self.backendCB.addItems(['loky', 'threading', 'multiprocessing'])
        self.backendCB.setCurrentIndex(1)

        njobsLabel = QLabel('Processing Cores:')
        self.njobsSB = QSpinBox()
        self.njobsSB.setRange(-100, 100)
        self.njobsSB.setValue(-2)
        self.njobsSB.setAlignment(Qt.AlignRight)

        randomStateLabel = QLabel('Random State:')
        self.randomStateProcSB = QSpinBox()
        self.randomStateProcSB.setRange(0, 1000)
        self.randomStateProcSB.setValue(0)
        self.randomStateProcSB.setAlignment(Qt.AlignRight)

        self.autoNegativeCB = QCheckBox('Auto Negative Sign')
        self.autoNegativeCB.setChecked(True)
        self.autoNegativeState = QLabel('checked')
        self.autoNegativeCB.toggled.connect(
            lambda: self.checkBoxState(
                check_box=self.autoNegativeCB,
                status=self.autoNegativeState
            )
        )

        cancelButton = QPushButton('Cancel')
        cancelButton.clicked.connect(self.processingOptionDialog.close)
        loadButton = QPushButton('Load')
        loadButton.clicked.connect(self.loadProcessingOptionAction)
        loadButton.clicked.connect(self.processingOptionDialog.close)

        grid = QGridLayout()

        grid.addWidget(backendLabel, 1, 1, 1, 2)
        grid.addWidget(self.backendCB, 1, 3, 1, 2)

        grid.addWidget(njobsLabel, 2, 1, 1, 2)
        grid.addWidget(self.njobsSB, 2, 3, 1, 2)

        grid.addWidget(randomStateLabel, 3, 1, 1, 2)
        grid.addWidget(self.randomStateProcSB, 3, 3, 1, 2)

        grid.addWidget(self.autoNegativeCB, 4, 1, 1, 4)

        grid.addWidget(loadButton, 5, 3, 1, 1)
        grid.addWidget(cancelButton, 5, 4, 1, 1)

        self.processingOptionDialog.setLayout(grid)

        self.processingOptionDialog.exec_()


    def loadProcessingOptionAction(self):
        '''
        Loading defined processing option input
        '''

        if self.njobsSB.value() == 0:
            self.processingOptionDialog.close()
            self.warningWithoutClear(
                'Do not insert zero on Processing Cores!'
            )
            self.processingOptionWindow()
        else:
            proc_op_dict['backend'] = self.backendCB.currentText()
            proc_op_dict['n_jobs'] = self.njobsSB.value()
            proc_op_dict['random_state'] = self.randomStateProcSB.value()
            proc_op_dict['auto_negative'] = self.autoNegativeState.text()


    def predict(self):
        '''
        Sending parameters and inputs from widget to Process Class
        '''
        print('widget predict')

        self.resultText.clear()
        self.progressBar.setValue(0)

        if self.limitADSB.value() < self.limitBDSB.value():
            a = self.limitADSB.value()
            b = self.limitBDSB.value()

            self.limitADSB.setValue(b)
            self.limitBDSB.setValue(a)

        global time_list
        time_list = []
        init_input = {
            'depth_label': self.depthHeaderCB.currentText(),
            'train_size': self.trainPercentDSB.value() / 100,
            'limit_state': self.limitState.text(),
            'limit_a': self.limitADSB.value(),
            'limit_b': self.limitBDSB.value(),
            'method': self.methodCB.currentText()
        }

        try:
            if sample_raw[self.depthHeaderCB.currentText()].dtype == 'float':
                self.sdbProcess = Process()
                self.widget_signal.connect(self.sdbProcess.inputs)
                self.widget_signal.emit(init_input)
                self.sdbProcess.start()
                self.sdbProcess.time_signal.connect(self.timeCounting)
                self.sdbProcess.thread_signal.connect(self.results)
                self.sdbProcess.warning_with_clear.connect(self.warningWithClear)
            else:
                self.warningWithClear(
                    'Please select headers correctly!'
                )
        except NameError:
            self.warningWithClear(
                'No depth sample loaded. Please load your depth sample!'
            )


    def timeCounting(self, time_text):
        '''
        Receive time value on every step and its corresponding processing
        text to show in result text browser and increase progress bar.
        '''

        time_list.append(time_text[0])
        self.resultText.append(time_text[1])
        self.progressBar.setValue(self.progressBar.value() + 1)

        if self.progressBar.value() == self.progressBar.maximum():
            self.completeDialog()


    def results(self, result_dict):
        '''
        Recieve processing results and filter the predicted value to depth
        limit window (if enabled).
        Counting runtimes using saved time values and printing result info.
        '''

        global z_predict
        z_predict = result_dict['z_predict']
        rmse = result_dict['rmse']
        mae = result_dict['mae']
        r2 = result_dict['r2']

        if self.limitState.text() == 'unchecked':
            print('checking prediction')
            z_predict[z_predict < self.limitBDSB.value()] = np.nan
            z_predict[z_predict > self.limitADSB.value()] = np.nan

            print_limit = (
                'Depth Limit:\t\tfrom ' + str(self.limitADSB.value()) + ' m ' +
                'to ' + str(self.limitBDSB.value()) + ' m'
            )
        else:
            print_limit = (
                'Depth Limit:\t\tDisabled'
            )

        if proc_op_dict['auto_negative'] == 'checked':
            auto_negative = 'Enabled'
        elif proc_op_dict['auto_negative'] == 'unchecked':
            auto_negative = 'Disabled'

        time_array = np.array(time_list)
        time_diff = time_array[1:] - time_array[:-1]
        runtime = np.append(time_diff, time_list[-1] - time_list[0])

        coord1 = np.array(image_raw.transform * (0, 0))
        coord2 = np.array(image_raw.transform * (1, 1))
        pixel_size = abs(coord2 - coord1)

        global print_result_info
        print_result_info = (
            'Image Input:\t\t' + self.imglocList.toPlainText() + ' (' +
            str(round(img_size / 2**20, 2)) + ' MB)\n' +
            'Sample Data:\t\t' + self.samplelocList.toPlainText() + ' (' +
            str(round(sample_size / 2**20, 2)) + ' MB)\n\n' +
            print_limit + '\n' +
            'Train Data:\t\t' + str(self.trainPercentDSB.value()) + ' %\n' +
            'Test Data:\t\t' + str(100 - self.trainPercentDSB.value()) + ' %\n\n' +
            'Method:\t\t' + self.methodCB.currentText() + '\n' +
            print_parameters_info + '\n\n'
            'RMSE:\t\t' + str(rmse) + '\n' +
            'MAE:\t\t' + str(mae) + '\n' +
            'R\u00B2:\t\t' + str(r2) + '\n\n' +
            'Parallel Backend:\t' + str(proc_op_dict['backend']) + '\n' +
            'Processing Cores:\t' + str(proc_op_dict['n_jobs']) + '\n' +
            'Random State:\t\t' + str(proc_op_dict['random_state']) + '\n'
            'Auto Negative Sign:\t' + auto_negative + '\n\n' +
            'Reproject Runtime:\t' + str(runtime[0]) + '\n' +
            'Sampling Runtime:\t' + str(runtime[1]) + '\n' +
            'Fitting Runtime:\t\t' + str(runtime[2]) + '\n' +
            'Prediction Runtime:\t' + str(runtime[3]) + '\n' +
            'Validating Runtime:\t' + str(runtime[4]) + '\n' +
            'Overall Runtime:\t' + str(runtime[5]) + '\n\n' +
            'CRS:\t\t' + str(image_raw.crs) + '\n'
            'Dimensions:\t\t' + str(image_raw.width) + ' x ' +
            str(image_raw.height) + ' pixels\n' +
            'Pixel Size:\t\t' + str(pixel_size[0]) + ' , ' +
            str(pixel_size[1]) + '\n\n'
        )

        self.resultText.setText(print_result_info)


    def warningWithClear(self, warning_text):
        '''
        Show warning dialog and customized warning text
        and then clear result info and progress bar after closing
        '''

        warning = QErrorMessage()
        warning.setWindowTitle('WARNING')
        warning.setWindowIcon(QIcon(resource_path('icons/warning-pngrepo-com.png')))
        warning.showMessage(warning_text)

        warning.exec_()
        self.resultText.clear()
        self.progressBar.setValue(0)


    def warningWithoutClear(self, warning_text):
        '''
        Show warning dialog and customized warning text
        without clearing result info and progress bar after closing
        '''

        warning = QErrorMessage()
        warning.setWindowTitle('WARNING')
        warning.setWindowIcon(QIcon(resource_path('icons/warning-pngrepo-com.png')))
        warning.showMessage(warning_text)

        warning.exec_()


    def completeDialog(self):
        '''
        Showing complete pop up dialog
        '''

        complete = QDialog()
        complete.setWindowTitle('Complete')
        complete.setWindowIcon(QIcon(resource_path('icons/complete-pngrepo-com.png')))
        complete.resize(180,30)

        textLabel = QLabel('Tasks has been completed')
        textLabel.setAlignment(Qt.AlignCenter)

        okButton = QPushButton('OK')
        okButton.clicked.connect(complete.close)

        grid = QGridLayout()

        grid.addWidget(textLabel, 1, 1, 1, 4)
        grid.addWidget(okButton, 2, 2, 1, 2)

        complete.setLayout(grid)

        complete.exec_()


    def saveOptionWindow(self):
        '''
        Saving option window
        '''

        self.saveOptionDialog = QDialog()
        self.saveOptionDialog.setWindowTitle('Save Options')
        self.saveOptionDialog.setWindowIcon(QIcon(resource_path('icons/load-pngrepo-com.png')))

        global format_dict
        format_dict = {
            'GeoTIFF (*.tif)': 'GTiff',
            'Erdas Imagine image (*.img)': 'HFA',
            'ASCII Gridded XYZ (*.xyz)': 'XYZ'
        }

        format_list = list(format_dict)
        format_list.sort()

        dataTypeLabel = QLabel('Data Type:')
        self.dataTypeCB = QComboBox()
        self.dataTypeCB.addItems(format_list)
        self.dataTypeCB.setCurrentText('GeoTIFF (*.tif)')

        saveFileButton = QPushButton('Save File Location')
        saveFileButton.clicked.connect(
            lambda:self.fileDialog(
                command=QFileDialog.getSaveFileName,
                window_text='Save File',
                file_type=self.dataTypeCB.currentText(),
                text_browser=self.savelocList
            )
        )

        medianFilterLabel = QLabel('Median Filter Size:')
        self.medianFilterSB = QSpinBox()
        self.medianFilterSB.setRange(3, 33)
        self.medianFilterSB.setValue(3)
        self.medianFilterSB.setSingleStep(2)
        self.medianFilterSB.setAlignment(Qt.AlignRight)

        self.medianFilterCheckBox = QCheckBox('Disable Median Filter')
        self.medianFilterCheckBox.setChecked(False)
        self.medianFilterState = QLabel('unchecked')
        self.medianFilterCheckBox.toggled.connect(
            lambda: self.checkBoxState(
                check_box=self.medianFilterCheckBox,
                status=self.medianFilterState
            )
        )

        locLabel = QLabel('Location:')
        self.savelocList = QTextBrowser()

        self.reportCheckBox = QCheckBox('Save Report')
        self.reportCheckBox.setChecked(True)
        self.reportState = QLabel('checked')
        self.reportCheckBox.toggled.connect(
            lambda: self.checkBoxState(
                check_box=self.reportCheckBox,
                status=self.reportState
            )
        )

        cancelButton = QPushButton('Cancel')
        cancelButton.clicked.connect(self.saveOptionDialog.close)
        saveButton = QPushButton('Save')
        saveButton.clicked.connect(self.saveAction)
        saveButton.clicked.connect(self.saveOptionDialog.close)

        grid = QGridLayout()
        grid.addWidget(dataTypeLabel, 1, 1, 1, 2)
        grid.addWidget(self.dataTypeCB, 1, 3, 1, 2)

        grid.addWidget(medianFilterLabel, 2, 1, 1, 1)
        grid.addWidget(self.medianFilterSB, 2, 2, 1, 1)
        grid.addWidget(self.medianFilterCheckBox, 2, 3, 1, 2)

        grid.addWidget(saveFileButton, 3, 1, 1, 4)

        grid.addWidget(locLabel, 4, 1, 1, 4)
        grid.addWidget(self.savelocList, 5, 1, 1, 4)

        grid.addWidget(self.reportCheckBox, 6, 1, 1, 2)
        grid.addWidget(saveButton, 6, 3, 1, 1)
        grid.addWidget(cancelButton, 6, 4, 1, 1)

        self.saveOptionDialog.setLayout(grid)

        self.saveOptionDialog.exec_()


    def saveAction(self):
        '''
        Saving predicted depth and report (optional) into file.
        Applying median filter (or not) to the predicted depth array
        before saving.
        '''

        try:
            z_img_ar = z_predict.reshape(image_raw.height, image_raw.width)

            if self.medianFilterState.text() == 'unchecked':
                print_filter_info = (
                    'Median Filter Size:\t' + str(self.medianFilterSB.value())
                )
                z_img_ar = ndimage.median_filter(z_img_ar, size=self.medianFilterSB.value())
            else:
                print_filter_info = (
                    'Median Filter Size:\tDisabled'
                )

            new_img = rio.open(
                self.savelocList.toPlainText(),
                'w',
                driver=format_dict[self.dataTypeCB.currentText()],
                height=image_raw.height,
                width=image_raw.width,
                count=1,
                dtype=z_img_ar.dtype,
                crs=image_raw.crs,
                transform=image_raw.transform
            )

            new_img.write(z_img_ar, 1)
            new_img.close()

            new_img_size = os.path.getsize(self.savelocList.toPlainText())
            print_output_info = (
                print_filter_info + '\n\n'
                'Output:\t\t' + self.savelocList.toPlainText() + ' (' +
                str(round(new_img_size / 2**10 / 2**10, 2)) + ' MB)'
            )

            self.resultText.append(print_output_info)

            if self.reportState.text() == 'checked':
                report_save_loc = (
                    os.path.splitext(self.savelocList.toPlainText())[0] +
                    '_report.txt'
                )
                report = open(report_save_loc, 'w')

                report.write(
                    print_result_info +
                    print_output_info
                )
        except:
            self.saveOptionDialog.close()
            self.warningWithoutClear(
                'Please insert save location!'
            )
            self.saveOptionWindow()


    def licensesDialog(self):
        '''
        Showing the license of SDB GUI and another library licenses
        '''

        licenses = QDialog()
        licenses.setWindowTitle('Licenses')
        licenses.setWindowIcon(QIcon(resource_path('icons/information-pngrepo-com.png')))
        licenses.resize(600, 380)

        license_dict = {
            'SDB GUI': 'LICENSE',
            'NumPy': 'licenses/numpy_license',
            'Scipy': 'licenses/scipy_license',
            'Pandas': 'licenses/pandas_license',
            'Rasterio': 'licenses/rasterio_license',
            'GeoPandas': 'licenses/geopandas_license',
            'Scikit Learn': 'licenses/scikit-learn_license'
        }
        license_list = list(license_dict)

        licenseCB = QComboBox()
        licenseCB.addItems(license_list)
        licenseCB.activated.connect(
            lambda: self.licenseSelection(
                location=license_dict[licenseCB.currentText()]
            )
        )

        license_file = open(resource_path('LICENSE'), 'r')
        self.licenseText = QTextBrowser()
        self.licenseText.setText(license_file.read())

        okButton = QPushButton('OK')
        okButton.clicked.connect(licenses.close)

        grid = QGridLayout()

        grid.addWidget(licenseCB, 1, 1, 1, 4)
        grid.addWidget(self.licenseText, 2, 1, 1, 4)
        grid.addWidget(okButton, 3, 4, 1, 1)

        licenses.setLayout(grid)

        licenses.exec_()


    def licenseSelection(self, location):
        '''
        Selecting license file location
        '''

        license_file = open(resource_path(location), 'r')
        self.licenseText.setText(license_file.read())



class Process(QThread):
    '''
    Data processing class of SDB GUI.
    Sending inputs from SDBWidget to process in the background
    so the GUI won't freeze while processing data.
    '''

    thread_signal = pyqtSignal(dict)
    time_signal = pyqtSignal(list)
    warning_with_clear = pyqtSignal(str)
    warning_without_clear = pyqtSignal(str)

    def __init__(self):

        QThread.__init__(self)


    def inputs(self, input_dict):
        '''
        Pooling intputs from widget
        '''

        self.depth_label = input_dict['depth_label']
        self.train_size = input_dict['train_size']
        self.limit_state = input_dict['limit_state']
        self.limit_a_value = input_dict['limit_a']
        self.limit_b_value = input_dict['limit_b']
        self.method = input_dict['method']


    def sampling(self):
        '''
        Preparing input values from widget to use on 
        training models and predicting depth by reprojecting 
        depth sample CRS, sampling raster value and depth value, 
        and then limiting or not limiting depth value.
        '''
        print('Process sampling')

        time_start = datetime.datetime.now()

        image_crs = str(image_raw.crs).upper()
        sample_crs = str(sample_raw.crs).upper()

        if image_crs != sample_crs:
            start_list = [time_start, 'Reprojecting...\n']
            self.time_signal.emit(start_list)

            sample_reproj = sample_raw.to_crs(image_crs)
        else:
            start_list = [time_start, 'Skip Reproject...\n']
            self.time_signal.emit(start_list)

            sample_reproj = sample_raw.copy()

        time_reproj = datetime.datetime.now()
        reproj_list = [time_reproj, 'Point Sampling...\n']
        self.time_signal.emit(reproj_list)

        shp_geo = sample_reproj['geometry']

        col_names = []

        with parallel_backend(proc_op_dict['backend'], n_jobs=proc_op_dict['n_jobs']):

            row, col = np.array(image_raw.index(shp_geo.x, shp_geo.y))
            sample_bands = image_raw.read()[:, row, col].T

            for i in image_raw.indexes:
                col_names.append('band' + str(i))

        samples_edit = pd.DataFrame(sample_bands, columns=col_names)
        samples_edit['z'] = sample_reproj[self.depth_label]

        if proc_op_dict['auto_negative'] == 'checked' and np.median(samples_edit['z']) > 0:
            samples_edit['z'] = samples_edit['z'] * -1

        if self.limit_state == 'unchecked':
            samples_edit = samples_edit[samples_edit['z'] >= self.limit_b_value]
            samples_edit = samples_edit[samples_edit['z'] <= self.limit_a_value]

        features = samples_edit.iloc[:, 0:-1]
        z = samples_edit['z']

        features_train, features_test, z_train, z_test = train_test_split(
            features,
            z,
            train_size=self.train_size,
            random_state=proc_op_dict['random_state']
            )

        samples_split = [features_train, features_test, z_train, z_test]

        return samples_split


    def knnPredict(self):
        '''
        Preparing KNN prediction and saving selected parameters
        for report
        '''
        print('knnPredict')

        parameters = self.sampling()

        regressor = KNeighborsRegressor(
            n_neighbors=knn_op_dict['n_neighbors'],
            weights=knn_op_dict['weights'],
            algorithm=knn_op_dict['algorithm'],
            leaf_size=knn_op_dict['leaf_size']
        )

        parameters.append(regressor)

        global print_parameters_info
        print_parameters_info = (
            'N Neighbors:\t\t' + str(knn_op_dict['n_neighbors']) + '\n' +
            'Weights:\t\t' + str(knn_op_dict['weights']) + '\n' +
            'Algorithm:\t\t' + str(knn_op_dict['algorithm']) + '\n' +
            'Leaf Size:\t\t' + str(knn_op_dict['leaf_size'])
        )

        return parameters


    def mlrPredict(self):
        '''
        Preparing MLR prediction and saving selected parameters
        for report
        '''
        print('mlrPredict')

        parameters = self.sampling()

        regressor = LinearRegression(
            fit_intercept=mlr_op_dict['fit_intercept'],
            normalize=mlr_op_dict['normalize'],
            copy_X=mlr_op_dict['copy_x']
        )

        parameters.append(regressor)

        global print_parameters_info
        print_parameters_info = (
            'Fit Intercept:\t\t' + str(mlr_op_dict['fit_intercept']) + '\n' +
            'Normalize:\t\t' + str(mlr_op_dict['normalize']) + '\n' +
            'Copy X:\t\t' + str(mlr_op_dict['copy_x'])
        )

        return parameters


    def rfPredict(self):
        '''
        Preparing RF prediction and saving selected parameters
        for report
        '''
        print('rfPredict')

        parameters = self.sampling()

        regressor = RandomForestRegressor(
            n_estimators=rf_op_dict['n_estimators'],
            criterion=rf_op_dict['criterion'],
            bootstrap=rf_op_dict['bootstrap'],
            random_state=rf_op_dict['random_state'])

        parameters.append(regressor)

        global print_parameters_info
        print_parameters_info = (
            'N Trees:\t\t' + str(rf_op_dict['n_estimators']) + '\n' +
            'Criterion:\t\t' + str(rf_op_dict['criterion']) + '\n' +
            'Bootstrap:\t\t' + str(rf_op_dict['bootstrap']) + '\n' +
            'Random State:\t\t' + str(rf_op_dict['random_state'])
        )

        return parameters


    def svmPredict(self):
        '''
        Preparing SVM prediction and saving selected parameters
        for report
        '''
        print('svmPredict')

        parameters = self.sampling()

        regressor = SVR(
            kernel=svm_op_dict['kernel'],
            gamma=svm_op_dict['gamma'],
            C=svm_op_dict['c'],
            degree=svm_op_dict['degree'],
            cache_size=8000)

        parameters.append(regressor)

        global print_parameters_info
        print_parameters_info = (
            'Kernel:\t\t' + str(svm_op_dict['kernel']) +'\n' +
            'Gamma:\t\t' + str(svm_op_dict['gamma']) + '\n' +
            'C:\t\t' + str(svm_op_dict['c'])
        )

        if svm_op_dict['kernel'] == 'poly':
            print_parameters_info = (
                print_parameters_info + '\n' +
                'Degree:\t\t' + str(svm_op_dict['degree'])
            )

        return parameters


    def run(self):
        '''
        Taking pre processed input and chosen method, then 
        fitting training data to chosen model and make prediction
        based on trained model.
        '''
        print('Process run')

        method_list = list(method_dict)

        try:
            if self.method == method_list[0]:
                parameters = self.knnPredict()
            elif self.method == method_list[1]:
                parameters = self.mlrPredict()
            elif self.method == method_list[2]:
                parameters = self.rfPredict()
            elif self.method == method_list[3]:
                parameters = self.svmPredict()

            features_train = parameters[0]
            features_test = parameters[1]
            z_train = parameters[2]
            z_test = parameters[3]
            regressor = parameters[4]

            time_sampling = datetime.datetime.now()
            sampling_list = [time_sampling, 'Fitting...\n']
            self.time_signal.emit(sampling_list)

            with parallel_backend(proc_op_dict['backend'], n_jobs=proc_op_dict['n_jobs']):

                regressor.fit(features_train, z_train)
                time_fit = datetime.datetime.now()
                fit_list = [time_fit, 'Predicting...\n']
                self.time_signal.emit(fit_list)

                z_predict = regressor.predict(bands_array)
                time_predict = datetime.datetime.now()
                predict_list = [time_predict,'Validating...\n']
                self.time_signal.emit(predict_list)

                z_validate = regressor.predict(features_test)
                rmse = np.sqrt(metrics.mean_squared_error(z_test, z_validate))
                mae = metrics.mean_absolute_error(z_test, z_validate)
                r2 = metrics.r2_score(z_test, z_validate)
                time_test = datetime.datetime.now()
                test_list = [time_test, 'Done.']
                self.time_signal.emit(test_list)

            result = {
                'z_predict': z_predict,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }

            self.thread_signal.emit(result)
        except NameError:
            self.warning_with_clear.emit(
                'No image data loaded. Please load your image data!'
            )
        except IndexError:
            self.warning_with_clear.emit(
                'Depth sample is out of image boundary'
            )



def main():

    global sdb
    sdb = SDBWidget()
    sdb.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main()
    sys.exit(app.exec_())
