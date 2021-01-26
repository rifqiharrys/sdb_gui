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
#################### For Auto PY to EXE or PyInstaller Use ####################

import sklearn.neighbors
import sklearn.utils._cython_blas
import sklearn.tree
import sklearn.tree._utils
import rasterio._features
import rasterio._shim
import rasterio.control
import rasterio.crs
import rasterio.sample
import rasterio.vrt

###############################################################################
###############################################################################

from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from joblib import parallel_backend
from scipy import ndimage
import pandas as pd
import numpy as np
import rasterio as rio
from pathlib import Path
import glob
import sys, os
import datetime
import webbrowser
from PyQt5.QtCore import (Qt, QThread, pyqtSignal)
from PyQt5.QtWidgets import(QApplication, QWidget, QTextBrowser, QProgressBar, QFileDialog, QDialog,
                            QGridLayout, QPushButton, QVBoxLayout, QComboBox, QLabel, QCheckBox,
                            QDoubleSpinBox, QSpinBox, QRadioButton, QTableWidgetItem, QTableWidget,
                            QScrollArea, QHeaderView, QErrorMessage)
from PyQt5.QtGui import QIcon

def resource_path(relative_path):
    """ Get the absolute path to the resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


os.environ['PROJ_LIB'] = resource_path('share/proj')
os.environ['GDAL_DATA'] = resource_path('share')



class SDBWidget(QWidget):

    widget_signal = pyqtSignal(list)

    def __init__(self):

        super(SDBWidget, self).__init__()

####### Default Values #######
        global njobs
        njobs = -2

        global method_list
        method_list = [
            'Multiple Linear Regression',
            'Random Forest', 
            'Support Vector Machines'
        ]

        global mlr_op_list
        mlr_op_list = [
            True, # fit_intercept
            False, # normalize
            True # copy_X
        ]

        global rf_op_list
        rf_op_list = [
            300, # n_estimators
            'mse' # criterion
        ]

        global svm_op_list
        svm_op_list = [
            'poly', # kernel
            .1, # gamma
            1000.0, # C
            3 # degree
        ]
####### Default Values #######

        self.initUI()


    def initUI(self):

        self.setGeometry(300, 100, 480, 640)
        self.setWindowTitle('Satellite Derived Bathymetry')
        self.setWindowIcon(QIcon(resource_path('icons/satellite.png')))

        loadImageButton = QPushButton('Load Image')
        loadImageButton.clicked.connect(self.loadImageWindow)
        self.loadImageLabel = QLabel()
        self.loadImageLabel.setAlignment(Qt.AlignCenter)

        loadSampleButton = QPushButton('Load Sample')
        loadSampleButton.clicked.connect(self.loadSampleWindow)
        self.loadSampleLabel = QLabel()
        self.loadSampleLabel.setAlignment(Qt.AlignCenter)

        depthHeaderLabel = QLabel('Depth Header:')
        self.depthHeaderCB = QComboBox()

        bandStartLabel = QLabel('First Band Column:')
        self.bandStartCB = QComboBox()
        bandEndLabel = QLabel('Last Band Column:')
        self.bandEndCB = QComboBox()

        self.table = QTableWidget()
        scroll = QScrollArea()
        scroll.setWidget(self.table)

        limitLabel = QLabel('Depth Limit Value:')
        self.limitSB = QSpinBox()
        self.limitSB.setRange(-100, 0)
        self.limitSB.setValue(-30)
        self.limitSB.setSuffix(' m')
        self.limitSB.setAlignment(Qt.AlignRight)

        self.limitCheckBox = QCheckBox('Disable Depth Limitation')
        self.limitCheckBox.setChecked(False)
        self.limitCheckBox.toggled.connect(self.limitCheckBoxState)
        self.limitState = QLabel('unchecked')

        methodLabel = QLabel('Regression Method:')
        self.methodCB = QComboBox()

        self.methodCB.addItems(method_list)
        self.methodCB.activated.connect(self.methodSelection)

        trainPercentLabel = QLabel('Train Data (Percent):')
        self.trainPercentDSB = QDoubleSpinBox()
        self.trainPercentDSB.setRange(10.0, 90.0)
        self.trainPercentDSB.setDecimals(2)
        self.trainPercentDSB.setValue(75.0)
        self.trainPercentDSB.setSuffix(' %')
        self.trainPercentDSB.setAlignment(Qt.AlignRight)

        self.optionsButton = QPushButton('Options')
        self.optionsButton.clicked.connect(self.mlrOptionDialog)
        self.optionsButton.clicked.connect(self.methodSelection)

        makePredictionButton = QPushButton('Make Prediction')
        makePredictionButton.clicked.connect(self.predict)
        saveFileButton = QPushButton('Save Into File')
        saveFileButton.clicked.connect(self.saveOptionWindow)

        resultInfo = QLabel('Result Information')
        self.resultText = QTextBrowser()
        self.resultText.setAlignment(Qt.AlignRight)

        self.progressBar = QProgressBar()
        self.progressBar.setFormat('%p%')
        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(5)

        releaseButton =  QPushButton('Releases')
        releaseButton.clicked.connect(lambda: webbrowser.open(
            'https://github.com/rifqiharrys/sdb_gui/releases'
        ))

        aboutButton = QPushButton('About')
        aboutButton.clicked.connect(self.aboutDialog)

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

        grid.addWidget(bandStartLabel, 4, 1, 1, 1)
        grid.addWidget(self.bandStartCB, 4, 2, 1, 1)
        grid.addWidget(bandEndLabel, 4, 3, 1, 1)
        grid.addWidget(self.bandEndCB, 4, 4, 1, 1)

        grid.addWidget(self.table, 5, 1, 5, 4)

        grid.addWidget(limitLabel, 10, 1, 1, 1)
        grid.addWidget(self.limitSB, 10, 2, 1, 1)
        grid.addWidget(self.limitCheckBox, 10, 3, 1, 2)

        grid.addWidget(methodLabel, 11, 1, 1, 1)
        grid.addWidget(self.methodCB, 11, 2, 1, 3)

        grid.addWidget(trainPercentLabel, 12, 1, 1, 1)
        grid.addWidget(self.trainPercentDSB, 12, 2, 1, 1)

        grid.addWidget(self.optionsButton, 12, 3, 1, 2)

        grid.addWidget(makePredictionButton, 13, 1, 1, 2)
        grid.addWidget(saveFileButton, 13, 3, 1, 2)

        grid.addWidget(resultInfo, 14, 1, 1, 2)
        grid.addWidget(self.resultText, 15, 1, 1, 4)

        vbox.addStretch(1)
        grid.addLayout(vbox, 21, 1)

        grid.addWidget(self.progressBar, 22, 1, 1, 4)

        grid.addWidget(releaseButton, 23, 1, 1, 1)
        grid.addWidget(aboutButton, 23, 2, 1, 2)
        grid.addWidget(readmeButton, 23, 4, 1, 1)
        self.setLayout(grid)


    def str2bool(self, v):
        '''Transform string to boolean'''

        return v in ('True')


    def limitCheckBoxState(self):

        if self.limitCheckBox.isChecked() == True:
            self.limitState.setText('checked')
        else:
            self.limitState.setText('unchecked')


    def methodSelection(self):

        if self.methodCB.currentText() == method_list[0]:
            self.optionsButton.clicked.disconnect()
            self.optionsButton.clicked.connect(self.mlrOptionDialog)
        elif self.methodCB.currentText() == method_list[1]:
            self.optionsButton.clicked.disconnect()
            self.optionsButton.clicked.connect(self.rfOptionDialog)
        elif self.methodCB.currentText() == method_list[2]:
            self.optionsButton.clicked.disconnect()
            self.optionsButton.clicked.connect(self.svmOptionDialog)


    def loadImageWindow(self):

        self.loadImageDialog = QDialog()
        self.loadImageDialog.setWindowTitle('Load Image')
        self.loadImageDialog.setWindowIcon(QIcon(resource_path('icons/load-pngrepo-com.png')))

        openFilesButton = QPushButton('Open File')
        openFilesButton.clicked.connect(self.imageFileDialog)

        locLabel = QLabel('Location:')
        self.locList = QTextBrowser()

        cancelButton = QPushButton('Cancel')
        cancelButton.clicked.connect(self.loadImageDialog.close)
        loadButton = QPushButton('Load')
        loadButton.clicked.connect(self.loadImageAction)
        loadButton.clicked.connect(self.loadImageDialog.close)

        grid = QGridLayout()
        grid.addWidget(openFilesButton, 1, 1, 1, 4)

        grid.addWidget(locLabel, 4, 1, 1, 1)

        grid.addWidget(self.locList, 5, 1, 10, 4)

        grid.addWidget(loadButton, 15, 3, 1, 1)
        grid.addWidget(cancelButton, 15, 4, 1, 1)

        self.loadImageDialog.setLayout(grid)

        self.loadImageDialog.exec_()


    def imageFileDialog(self):

        home_dir = str(Path.home())
        fileFilter = 'All Files (*.*) ;; GeoTIFF (*.tif)'
        selectedFilter = 'GeoTIFF (*.tif)'
        fname = QFileDialog.getOpenFileName(self, 'Open File(s)', home_dir, fileFilter, selectedFilter)

        global img_loc
        img_loc = fname[0]

        self.locList.setText(img_loc)


    def loadImageAction(self):

        try:
            global img_size
            img_size = os.path.getsize(img_loc)

            global image_raw
            image_raw = rio.open(img_loc)

            nbands = len(image_raw.indexes)
            ndata = image_raw.read(1).size
            bands_dummy = np.zeros((nbands, ndata))
            for i in range(1, nbands + 1):
                bands_dummy[i - 1, :] = np.ravel(image_raw.read(i))

            global bands_array
            bands_array = bands_dummy.T

            coord1 = np.array(image_raw.transform * (0, 0))
            coord2 = np.array(image_raw.transform * (1, 1))

            global pixel_size
            pixel_size = coord2 - coord1

            self.loadImageLabel.setText('Image Data Loaded')
        except:
            self.loadImageDialog.close()
            self.noDataWarning()
            self.loadImageWindow()


    def loadSampleWindow(self):

        self.loadSampleDialog = QDialog()
        self.loadSampleDialog.setWindowTitle('Load Sample')
        self.loadSampleDialog.setWindowIcon(QIcon(resource_path('icons/load-pngrepo-com.png')))

        openFilesButton = QPushButton('Open File(s)')
        openFilesButton.clicked.connect(self.sampleFilesDialog)

        sepLabel = QLabel('Separator:')
        self.sepCB = QComboBox()
        self.sepCB.addItems(['Comma', 'Tab', 'Space', 'Semicolon'])

        headerLineLabel = QLabel('Header Starting Line:')
        self.headerLineSB = QSpinBox()
        self.headerLineSB.setMinimum(1)
        self.headerLineSB.setAlignment(Qt.AlignRight)

        dataLineLabel = QLabel('Data Starting Line:')
        self.dataLineSB = QSpinBox()
        self.dataLineSB.setMinimum(1)
        self.dataLineSB.setAlignment(Qt.AlignRight)

        locLabel = QLabel('Location:')
        self.locList = QTextBrowser()

        self.showCheckBox = QCheckBox('Show All Data to Table')
        self.showCheckBox.setChecked(False)
        self.showCheckBox.toggled.connect(self.showCheckBoxState)
        self.showState = QLabel()

        cancelButton = QPushButton('Cancel')
        cancelButton.clicked.connect(self.loadSampleDialog.close)
        loadButton = QPushButton('Load')
        loadButton.clicked.connect(self.loadSampleAction)
        loadButton.clicked.connect(self.loadSampleDialog.close)

        grid = QGridLayout()
        grid.addWidget(openFilesButton, 1, 1, 1, 4)

        grid.addWidget(sepLabel, 2, 1, 1, 1)
        grid.addWidget(self.sepCB, 2, 2, 1, 3)

        grid.addWidget(headerLineLabel, 3, 1, 1, 1)
        grid.addWidget(self.headerLineSB, 3, 2, 1, 1)
        grid.addWidget(dataLineLabel, 3, 3, 1, 1)
        grid.addWidget(self.dataLineSB, 3, 4, 1, 1)

        grid.addWidget(locLabel, 4, 1, 1, 1)

        grid.addWidget(self.locList, 5, 1, 10, 4)

        grid.addWidget(self.showCheckBox, 15, 1, 1, 2)
        grid.addWidget(loadButton, 15, 3, 1, 1)
        grid.addWidget(cancelButton, 15, 4, 1, 1)

        self.loadSampleDialog.setLayout(grid)

        self.loadSampleDialog.exec_()


    def sampleFilesDialog(self):

        home_dir = str(Path.home())
        fileFilter = 'All Files (*.*) ;; Text Files (*.txt) ;; Comma Separated Value (*.csv) ;; DAT Files (*.dat)'
        selectedFilter = 'Comma Separated Value (*.csv)'
        fname = QFileDialog.getOpenFileNames(self, 'Open File(s)', home_dir, fileFilter, selectedFilter)

        global filesList
        filesList = fname[0]

        global fileListPrint
        fileListPrint = ''

        for file in filesList:
            fileListPrint += file + '\n'

        self.locList.setText(fileListPrint)


    def showCheckBoxState(self):

        if self.showCheckBox.isChecked() == True:
            self.showState.setText('checked')
        else:
            self.showState.setText('unchecked')


    def loadSampleAction(self):

        try:
            head = self.headerLineSB.value() - 1
            start_data = self.dataLineSB.value() - 1
            sepDict = {'Tab': '\t', 'Comma': ',',
                       'Space': ' ', 'Semicolon': ';'}
            sepSelect = sepDict[self.sepCB.currentText()]

            dummy = []

            global sample_size
            sample_size = np.ones(len(filesList))

            for file in filesList:
                raw_single = pd.read_csv(file, sep=sepSelect, header=head)
                raw_single = raw_single.iloc[start_data:, 0:]

                dummy.append(raw_single)

                sample_size[filesList.index(file)] = os.path.getsize(file)

            sample_size = sample_size.sum()

            global samples_raw
            samples_raw = pd.concat(dummy, ignore_index=True, sort=False)

            raw = samples_raw.copy()

            if self.showState.text() == 'checked':
                data = raw
            else:
                data = raw.head(100)

            self.depthHeaderCB.clear()
            self.depthHeaderCB.addItems(data.columns)
            self.bandStartCB.clear()
            self.bandStartCB.addItems(data.columns)
            self.bandStartCB.setCurrentIndex(1)
            self.bandEndCB.clear()
            self.bandEndCB.addItems(data.columns)
            self.bandEndCB.setCurrentIndex(data.columns.size - 1)

            self.table.setColumnCount(len(data.columns))
            self.table.setRowCount(len(data.index))

            for h in range(len(data.columns)):
                self.table.setHorizontalHeaderItem(h, QTableWidgetItem(data.columns[h]))

            for i in range(len(data.index)):
                for j in range(len(data.columns)):
                    self.table.setItem(
                        i, j, QTableWidgetItem(str(data.iloc[i, j])))

            self.table.resizeRowsToContents()
            self.table.resizeColumnsToContents()

            self.loadSampleLabel.setText('Sample Data Loaded')
        except:
            self.loadSampleDialog.close()
            self.noDataWarning()
            self.loadSampleWindow()


    def mlrOptionDialog(self):

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

        global mlr_op_list
        mlr_op_list = [
            self.str2bool(self.fitInterceptCB.currentText()),
            self.str2bool(self.normalizeCB.currentText()),
            self.str2bool(self.copyXCB.currentText())
        ]


    def rfOptionDialog(self):

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

        grid.addWidget(loadButton, 3, 3, 1, 1)
        grid.addWidget(cancelButton, 3, 4, 1, 1)

        optionDialog.setLayout(grid)

        optionDialog.exec_()


    def loadRFOptionAction(self):

        global rf_op_list
        rf_op_list = [
            self.ntreeSB.value(),
            self.criterionCB.currentText()
        ]


    def svmOptionDialog(self):

        optionDialog = QDialog()
        optionDialog.setWindowTitle('Options (SVM)')
        optionDialog.setWindowIcon(QIcon(resource_path('icons/setting-tool-pngrepo-com.png')))

        kernelLabel = QLabel('Kernel:')
        self.kernelCB = QComboBox()
        self.kernelCB.addItems(['linear', 'poly', 'rbf', 'sigmoid'])
        self.kernelCB.setCurrentIndex(2)

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

        global svm_op_list
        svm_op_list = [
            self.kernelCB.currentText(),
            self.gammaDSB.value(),
            self.cDSB.value(),
            self.degreeSB.value()
        ]


    def predict(self):
        print('widget predict')

        self.resultText.clear()
        self.progressBar.setValue(0)

        global time_list
        time_list = []
        init_input = [
            self.depthHeaderCB.currentText(),
            self.bandStartCB.currentText(),
            self.bandEndCB.currentText(),
            self.trainPercentDSB.value() / 100,
            self.limitState.text(),
            self.limitSB.value(),
            self.methodCB.currentText()
        ]

        self.sdbProcess = Process()
        self.widget_signal.connect(self.sdbProcess.inputs)
        self.widget_signal.emit(init_input)
        self.sdbProcess.start()
        self.sdbProcess.time_signal.connect(self.timeCounting)
        self.sdbProcess.thread_signal.connect(self.results)
        self.sdbProcess.no_data_signal.connect(self.noDataWarning)
        self.sdbProcess.header_warning_signal.connect(self.headerWarning)


    def timeCounting(self, time_text):

        time_list.append(time_text[0])
        self.resultText.append(time_text[1])
        self.progressBar.setValue(self.progressBar.value() + 1)

        if self.progressBar.value() == 5:
            self.completeDialog()
        else:
            pass


    def results(self, result_list):

        global z_predict
        z_predict = result_list[0]
        rmse = result_list[1]
        mae = result_list[2]
        r2 = result_list[3]

        if self.limitState.text() == 'unchecked':
            print('checking prediction')
            z_predict[z_predict < self.limitSB.value()] = np.nan
            z_predict[z_predict > 0] = np.nan

            print_limit = (
                'Depth Limit:' + '\t\t' + str(self.limitSB.value())
            )
        else:
            print_limit = (
                'Depth Limit:' + '\t\t' + 'Disabled'
            )

        runtime = [
            time_list[1] - time_list[0],
            time_list[2] - time_list[1],
            time_list[3] - time_list[2],
            time_list[4] - time_list[3],
            time_list[4] - time_list[0]
        ]

        global print_result_info
        print_result_info = (
            'Image Input:' + '\t\t' + img_loc + ' (' +
            str(round(img_size / 2**10 / 2**10, 2)) + ' MB)' + '\n' +
            'Sample Data:' + '\t\t' + fileListPrint + ' (' +
            str(round(sample_size / 2**10 / 2**10, 2)) + ' MB)' + '\n\n' +
            print_limit + '\n' +
            'Train Data:' + '\t\t' + str(self.trainPercentDSB.value()) + ' %' + '\n' +
            'Test Data:' + '\t\t' + str(100 - self.trainPercentDSB.value()) + ' %' + '\n\n' +
            'Method:' + '\t\t' + self.methodCB.currentText() + '\n' +
            print_parameters_info + '\n\n'
            'RMSE:' + '\t\t' + str(rmse) + '\n' +
            'MAE:' + '\t\t' + str(mae) + '\n' +
            'R\u00B2:' + '\t\t' + str(r2) + '\n\n' +
            'Preparation Runtime:' + '\t' + str(runtime[0]) + '\n' +
            'Fitting Runtime:' + '\t\t' + str(runtime[1]) + '\n' +
            'Prediction Runtime:' + '\t' + str(runtime[2]) + '\n' +
            'Validating Runtime:' + '\t' + str(runtime[3]) + '\n' +
            'Overall Runtime:' + '\t' + str(runtime[4]) + '\n\n' +
            'CRS:' + '\t\t' + str(image_raw.crs) + '\n'
            'Dimensions:' + '\t\t' + str(image_raw.width) + ' x ' +
            str(image_raw.height) + ' pixels' + '\n' +
            'Pixel Size:' + '\t\t' + str(pixel_size[0]) + ' , ' +
            str(pixel_size[1]) + '\n\n'
        )

        self.resultText.setText(print_result_info)


    def noDataWarning(self):

        warning = QErrorMessage()
        warning.setWindowTitle('WARNING')
        warning.setWindowIcon(QIcon(resource_path('icons/warning-pngrepo-com.png')))
        warning.showMessage('No data loaded. Please input your data!')

        warning.exec_()
        self.resultText.clear()


    def headerWarning(self):

        warning = QErrorMessage()
        warning.setWindowTitle('WARNING')
        warning.setWindowIcon(QIcon(resource_path('icons/warning-pngrepo-com.png')))
        warning.showMessage('Please select headers correctly!')

        warning.exec_()
        self.resultText.clear()


    def noSaveLocWarning(self):

        warning = QErrorMessage()
        warning.setWindowTitle('WARNING')
        warning.setWindowIcon(QIcon(resource_path('icons/warning-pngrepo-com.png')))
        warning.showMessage('Please insert save location!')

        warning.exec_()
        self.resultText.clear()


    def completeDialog(self):

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

        self.saveOptionDialog = QDialog()
        self.saveOptionDialog.setWindowTitle('Save Options')
        self.saveOptionDialog.setWindowIcon(QIcon(resource_path('icons/load-pngrepo-com.png')))

        saveFileButton = QPushButton('Save File Location')
        saveFileButton.clicked.connect(self.savePathDialog)

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

        medianFilterLabel = QLabel('Median Filter Size:')
        self.medianFilterSB = QSpinBox()
        self.medianFilterSB.setRange(3, 33)
        self.medianFilterSB.setValue(3)
        self.medianFilterSB.setSingleStep(2)
        self.medianFilterSB.setAlignment(Qt.AlignRight)

        self.medianFilterCheckBox = QCheckBox('Disable Median Filter')
        self.medianFilterCheckBox.setChecked(False)
        self.medianFilterCheckBox.toggled.connect(self.medianFilterCheckBoxState)
        self.medianFilterState = QLabel('unchecked')

        locLabel = QLabel('Location:')
        self.locList = QTextBrowser()

        self.reportCheckBox = QCheckBox('Save Report')
        self.reportCheckBox.setChecked(True)
        self.reportCheckBox.toggled.connect(self.reportCheckBoxState)
        self.reportState = QLabel('checked')

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
        grid.addWidget(self.locList, 5, 1, 1, 4)

        grid.addWidget(self.reportCheckBox, 6, 1, 1, 2)
        grid.addWidget(saveButton, 6, 3, 1, 1)
        grid.addWidget(cancelButton, 6, 4, 1, 1)

        self.saveOptionDialog.setLayout(grid)

        self.saveOptionDialog.exec_()


    def savePathDialog(self):

        home_dir = str(Path.home())
        fileFilter = 'All Files(*.*) ;; ' + self.dataTypeCB.currentText()
        selectedFilter = self.dataTypeCB.currentText()
        fname = QFileDialog.getSaveFileName(self, 'Save File', home_dir, fileFilter, selectedFilter)

        global save_loc
        save_loc = fname[0]

        self.locList.setText(save_loc)


    def medianFilterCheckBoxState(self):

        if self.medianFilterCheckBox.isChecked() == True:
            self.medianFilterState.setText('checked')
        else:
            self.medianFilterState.setText('unchecked')


    def reportCheckBoxState(self):

        if self.reportCheckBox.isChecked() == True:
            self.reportState.setText('checked')
        else:
            self.reportState.setText('unchecked')


    def saveAction(self):

        try:
            z_img_ar = z_predict.reshape(image_raw.height, image_raw.width)

            if self.medianFilterState.text() == 'unchecked':
                print_filter_info = (
                    'Median Filter Size:' + '\t' + str(self.medianFilterSB.value())
                )
                z_img_ar = ndimage.median_filter(z_img_ar, size=self.medianFilterSB.value())
            else:
                print_filter_info = (
                    'Median Filter Size:' + '\t' + 'Disabled'
                )

            new_img = rio.open(
                save_loc,
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

            new_img_size = os.path.getsize(save_loc)
            print_output_info = (
                print_filter_info + '\n\n'
                'Output:' + '\t\t' + save_loc + ' (' +
                str(round(new_img_size / 2**10 / 2**10, 2)) + ' MB)'
            )

            self.resultText.append(print_output_info)

            if self.reportState.text() == 'checked':
                report_save_loc = os.path.splitext(save_loc)[0] + '_report.txt'
                report = open(report_save_loc, 'w')

                report.write(
                    print_result_info +
                    print_output_info
                )
            else:
                pass
        except:
            self.saveOptionDialog.close()
            self.noSaveLocWarning()
            self.saveOptionWindow()


    def aboutDialog(self):

        about = QDialog()
        about.setWindowTitle('About')
        about.setWindowIcon(QIcon(resource_path('icons/information-pngrepo-com.png')))
        about.resize(500, 380)

        okButton = QPushButton('OK')
        okButton.clicked.connect(about.close)

        license_file = open(resource_path('LICENSE'), 'r')
        licenseLabel = QLabel('SDB GUI')
        licenseText = QTextBrowser()
        licenseText.setText(license_file.read())

        grid = QGridLayout()

        grid.addWidget(licenseLabel, 1, 1, 1, 4)
        grid.addWidget(licenseText, 2, 1, 1, 4)
        grid.addWidget(okButton, 3, 4, 1, 1)

        about.setLayout(grid)

        about.exec_()



class Process(QThread):

    thread_signal = pyqtSignal(list)
    time_signal = pyqtSignal(list)
    no_data_signal = pyqtSignal()
    header_warning_signal = pyqtSignal()

    def __init__(self):

        QThread.__init__(self)


    def inputs(self, input_list):

        self.depth_label = input_list[0]
        self.band_start = input_list[1]
        self.band_end = input_list[2]
        self.train_size = input_list[3]
        self.limitState = input_list[4]
        self.limitValue = input_list[5]
        self.method = input_list[6]


    def sampling(self):
        print('Process sampling')

        time_start = datetime.datetime.now()
        start_list = [time_start, 'Preparing...\n']
        self.time_signal.emit(start_list)

        samples_edit = samples_raw.copy()

        positives_count = samples_edit[samples_edit[self.depth_label] > 0][self.depth_label].count()
        samples_count = samples_edit[self.depth_label].count()

        if positives_count > samples_count / 2:
            samples_edit[self.depth_label] = samples_edit[self.depth_label] * -1
        else:
            pass

        if self.limitState == 'unchecked':
            print('depth limit')
            samples_edit = samples_edit[samples_edit[self.depth_label] >= self.limitValue]
            samples_edit = samples_edit[samples_edit[self.depth_label] <= 0]
        else:
            pass

        start_loc = samples_edit.columns.get_loc(self.band_start)
        end_loc = samples_edit.columns.get_loc(self.band_end)

        features = samples_edit.iloc[:, start_loc:end_loc+1]
        z = samples_edit[self.depth_label]

        features_train, features_test, z_train, z_test = train_test_split(features, z, train_size=self.train_size, random_state=0)

        samples_split = [features_train, features_test, z_train, z_test]

        return samples_split


    def mlrPredict(self):
        print('mlrPredict')

        parameters = self.sampling()

        regressor = LinearRegression(
            fit_intercept=mlr_op_list[0],
            normalize=mlr_op_list[1],
            copy_X=mlr_op_list[2]
        )

        parameters.append(regressor)

        global print_parameters_info
        print_parameters_info = (
            'Fit Intercept:' + '\t\t' + str(mlr_op_list[0]) + '\n' +
            'Normalize:' + '\t\t' + str(mlr_op_list[1]) + '\n' +
            'Copy X:' + '\t\t' + str(mlr_op_list[2])
        )

        return parameters


    def rfPredict(self):
        print('rfPredict')

        parameters = self.sampling()

        regressor = RandomForestRegressor(
            n_estimators=rf_op_list[0],
            criterion=rf_op_list[1],
            random_state=0)

        parameters.append(regressor)

        global print_parameters_info
        print_parameters_info = (
            'N Trees:' + '\t\t' + str(rf_op_list[0]) + '\n' +
            'Criterion:' + '\t\t' + str(rf_op_list[1])
        )

        return parameters


    def svmPredict(self):
        print('svmPredict')

        parameters = self.sampling()

        regressor = SVR(
            kernel=svm_op_list[0],
            gamma=svm_op_list[1],
            C=svm_op_list[2],
            cache_size=8000)

        parameters.append(regressor)

        global print_parameters_info
        print_parameters_info = (
            'Kernel:' + '\t\t' + str(svm_op_list[0]) +'\n' +
            'Gamma:' + '\t\t' + str(svm_op_list[1]) + '\n' +
            'C:' + '\t\t' + str(svm_op_list[2])
        )

        if svm_op_list[0] == 'poly':
            print_parameters_info = (
                print_parameters_info + '\n' +
                'Degree:' + '\t\t' + str(svm_op_list[3])
            )
        else:
            print_parameters_info = print_parameters_info

        return parameters


    def run(self):
        print('Process run')

        try:
            if self.method == method_list[0]:
                parameters = self.mlrPredict()
            elif self.method == method_list[1]:
                parameters = self.rfPredict()
            else:
                parameters = self.svmPredict()

            features_train = parameters[0]
            features_test = parameters[1]
            z_train = parameters[2]
            z_test = parameters[3]
            regressor = parameters[4]

            time_sampling = datetime.datetime.now()
            sampling_list = [time_sampling, 'Fitting...\n']
            self.time_signal.emit(sampling_list)

            with parallel_backend('threading', n_jobs=njobs):

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

            result = [
                z_predict,
                rmse,
                mae,
                r2
            ]

            self.thread_signal.emit(result)
        except NameError:
            self.no_data_signal.emit()
        except TypeError:
            self.header_warning_signal.emit()
        except ValueError:
            self.header_warning_signal.emit()



def main():

    global sdb
    sdb = SDBWidget()
    sdb.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main()
    sys.exit(app.exec_())
