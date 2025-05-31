"""
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

"""

import datetime
import logging
import os
import re
import sys
import webbrowser
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
from PyQt5.QtCore import QSettings, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog,
                             QDoubleSpinBox, QErrorMessage, QFileDialog,
                             QGridLayout, QLabel, QMessageBox, QProgressBar,
                             QPushButton, QScrollArea, QSpinBox, QTableWidget,
                             QTableWidgetItem, QTextBrowser, QVBoxLayout,
                             QWidget)

import sdb

## CONSTANTS ##
SDB_GUI_VERSION: str = '4.1.0'
LOG_NAME: str = 'sdb_gui.log'
PROGRESS_STEP: int = 6
DEPTH_DIR_DICT: Dict[str, Tuple[str, bool]] = {
    'Positive Up': ('up', False),
    'Positive Down': ('down', True),
}

def get_log_level() -> int:
    """
    Get logging level from command line argument.
    Default to INFO if no argument provided.
    """

    if len(sys.argv) > 1:
        level = sys.argv[1].upper()
        if hasattr(logging, level):
            return getattr(logging, level)
    return logging.INFO

logging.basicConfig(
    level=get_log_level(),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_NAME, mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info(
    f'logging level set to: {logging.getLevelName(logger.getEffectiveLevel())}'
)


class SDBWidget(QWidget):
    """
    PyQt5 widget of SDB GUI
    """

    widget_signal = pyqtSignal(dict)

    def __init__(self):
        """
        Initialize widget and default values
        """

        super(SDBWidget, self).__init__()

        self.settings = QSettings('SDB', 'SDB GUI')

        global option_pool, proc_op_dict
        option_pool = self.loadSettings()
        proc_op_dict = option_pool['processing']

        self.method_dict = {
            method: self.methodOptionWindow 
            for method in option_pool['method'].keys()
        }

        self.dir_path = self.settings.value('last_directory', os.path.abspath(Path.home()))
        self.initUI()


    def initUI(self):
        """
        Initialize User Interface for SDB GUI Widget
        """

        self.setGeometry(300, 100, 480, 640)
        self.setWindowTitle(f'Satellite Derived Bathymetry v{SDB_GUI_VERSION}')
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

        direction_list = list(DEPTH_DIR_DICT.keys())

        depthDirectionLabel = QLabel('Depth Direction:')
        self.depthDirectionCB =QComboBox()
        self.depthDirectionCB.addItems(direction_list)

        self.table = QTableWidget()
        scroll = QScrollArea()
        scroll.setWidget(self.table)

        limitLabel = QLabel('Depth Limit Window:')

        limitALabel = QLabel('Upper Limit:')
        self.limitADSB = QDoubleSpinBox()
        self.limitADSB.setRange(-100, 100)
        self.limitADSB.setDecimals(1)
        self.limitADSB.setValue(proc_op_dict['depth_limit']['upper'])
        self.limitADSB.setSuffix(' m')
        self.limitADSB.setAlignment(Qt.AlignRight)

        limitBLabel = QLabel('Lower Limit:')
        self.limitBDSB = QDoubleSpinBox()
        self.limitBDSB.setRange(-100, 100)
        self.limitBDSB.setDecimals(1)
        self.limitBDSB.setValue(proc_op_dict['depth_limit']['lower'])
        self.limitBDSB.setSuffix(' m')
        self.limitBDSB.setAlignment(Qt.AlignRight)

        self.limitCheckBox = QCheckBox('Disable Depth Limitation')
        self.limitCheckBox.setChecked(proc_op_dict['depth_limit']['disable'])

        method_list = list(self.method_dict)

        methodLabel = QLabel('Regression Method:')
        self.methodCB = QComboBox()
        self.methodCB.addItems(method_list)

        self.optionsButton = QPushButton('Method Options')
        self.optionsButton.clicked.connect(
            lambda: self.methodOptionWindow()
        )

        selection_list = list(proc_op_dict['selection'].keys())

        trainSelectLabel = QLabel('Train Data Selection:')
        self.trainSelectCB = QComboBox()
        self.trainSelectCB.addItems(selection_list)

        processingOptionsButton = QPushButton('Processing Options')
        processingOptionsButton.clicked.connect(self.processingOptionWindow)
        resetSettingsButton = QPushButton('Reset Settings')
        resetSettingsButton.clicked.connect(self.resetToDefault)

        makePredictionButton = QPushButton('Generate Prediction')
        makePredictionButton.clicked.connect(self.predict)
        stopProcessingButton = QPushButton('Stop Processing')
        stopProcessingButton.clicked.connect(self.stopProcess)

        resultInfo = QLabel('Result Information')
        saveFileButton = QPushButton('Save Into File')
        saveFileButton.clicked.connect(self.saveOptionWindow)
        self.resultText = QTextBrowser()
        self.resultText.setAlignment(Qt.AlignRight)

        self.progressBar = QProgressBar()
        self.progressBar.setFormat('%p%')
        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(PROGRESS_STEP)

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

        mainLayout = QVBoxLayout()

        grid1 = QGridLayout()
        grid1.addWidget(loadImageButton, 1, 1, 1, 1)
        grid1.addWidget(self.loadImageLabel, 1, 2, 1, 3)

        grid1.addWidget(loadSampleButton, 2, 1, 1, 1)
        grid1.addWidget(self.loadSampleLabel, 2, 2, 1, 3)

        grid1.addWidget(depthHeaderLabel, 3, 1, 1, 1)
        grid1.addWidget(self.depthHeaderCB, 3, 2, 1, 1)
        grid1.addWidget(depthDirectionLabel, 3, 3, 1, 1)
        grid1.addWidget(self.depthDirectionCB, 3, 4, 1, 1)

        grid1.addWidget(self.table, 4, 1, 5, 4)

        mainLayout.addLayout(grid1)

        grid2 = QGridLayout()
        grid2.addWidget(limitLabel, 1, 1, 1, 2)
        grid2.addWidget(self.limitCheckBox, 2, 1, 1, 2)

        grid2.addWidget(limitALabel, 1, 3, 1, 1)
        grid2.addWidget(self.limitADSB, 1, 4, 1, 1)
        grid2.addWidget(limitBLabel, 2, 3, 1, 1)
        grid2.addWidget(self.limitBDSB, 2, 4, 1, 1)

        mainLayout.addLayout(grid2)

        grid3 = QGridLayout()
        grid3.addWidget(methodLabel, 1, 1, 1, 1)
        grid3.addWidget(self.methodCB, 1, 2, 1, 1)

        grid3.addWidget(self.optionsButton, 1, 3, 1, 1)
        grid3.addWidget(processingOptionsButton, 1, 4, 1, 1)

        grid3.addWidget(trainSelectLabel, 2, 1, 1, 1)
        grid3.addWidget(self.trainSelectCB, 2, 2, 1, 1)
        grid3.addWidget(resetSettingsButton, 2, 3, 1, 2)

        mainLayout.addLayout(grid3)

        grid4 = QGridLayout()
        grid4.addWidget(makePredictionButton, 1, 1, 1, 2)
        grid4.addWidget(stopProcessingButton, 1, 3, 1, 2)

        grid4.addWidget(resultInfo, 2, 1, 1, 2)
        grid4.addWidget(saveFileButton, 2, 3, 1, 2)
        grid4.addWidget(self.resultText, 3, 1, 1, 4)

        grid4.addWidget(self.progressBar, 7, 1, 1, 4)

        grid4.addWidget(releaseButton, 8, 1, 1, 1)
        grid4.addWidget(licensesButton, 8, 2, 1, 2)
        grid4.addWidget(readmeButton, 8, 4, 1, 1)

        mainLayout.addLayout(grid4)

        self.setLayout(mainLayout)


    def loadSettings(self) -> dict:
        """
        Load settings or return defaults if none exist
        """

        if self.settings.value('options'):
            return self.settings.value('options')
        return default_values()


    def saveSettings(self) -> None:
        """
        Save all current settings
        """

        self.settings.setValue('options', option_pool)
        self.settings.setValue('last_directory', self.dir_path)


    def resetToDefault(self) -> None:
        """
        Reset all settings to default values
        """

        resetWindow = QMessageBox()
        resetWindow.setWindowTitle('Reset Settings')
        resetWindow.setText('Are you sure you want to reset all settings to default values?')
        resetWindow.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        resetWindow.setDefaultButton(QMessageBox.No)
        resetWindow.setWindowIcon(QIcon(resource_path('icons/warning-pngrepo-com.png')))

        reply = resetWindow.exec_()

        if reply == QMessageBox.Yes:
            self.settings.clear()

            global option_pool, proc_op_dict
            option_pool = default_values()
            proc_op_dict = option_pool['processing']

            home_dir = os.path.abspath(Path.home())
            self.dir_path = home_dir
            self.settings.setValue('last_directory', home_dir)
            logger.info('last directory and options reset to default')

            complete = QMessageBox()
            complete.setWindowTitle('Settings Reset')
            complete.setText('All settings have been reset to default values.')
            complete.setWindowIcon(QIcon(resource_path('icons/complete-pngrepo-com.png')))
            complete.exec_()


    def str2bool(self, v: str) -> bool:
        """
        Transform string True or False to boolean type
        """

        return v in ('True')


    def fileDialog(
        self, 
        command: Callable[..., tuple[str, str]],
        window_text: str,
        file_type: str,
        text_browser: QTextBrowser
    ) -> None:
        """
        Showing file dialog, whether opening file or saving.

        Parameters
        ----------
        command : Callable[..., tuple[str, str]]
            QFileDialog method (either getOpenFileName or getSaveFileName)
            that returns a tuple of (selected_path: str, selected_filter: str)
        window_text : str
            Title of the dialog window
        file_type : str
            File type filter (e.g., 'GeoTIFF (*.tif)')
        text_browser : QTextBrowser
            Text browser widget to display selected path

        Returns
        -------
        None
        """

        fileFilter = f'All Files (*.*) ;; {file_type}'
        selectedFilter = file_type
        fname = command(self, window_text, self.dir_path, fileFilter, selectedFilter)

        if fname[0]:
            text_browser.setText(fname[0])
            self.dir_path = os.path.dirname(fname[0])
            self.settings.setValue('last_directory', self.dir_path)


    def loadImageWindow(self):
        """
        Image loading User Interface
        """

        self.loadImageDialog = QDialog()
        self.loadImageDialog.setWindowTitle('Load Image')
        self.loadImageDialog.setWindowIcon(
            QIcon(resource_path('icons/load-pngrepo-com.png'))
        )

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
        """
        Loading selected image and retrieve some metadata such as file size,
        band quantity, array size, pixel size, etc. Then, recreate image 3D
        array into a simple column and row array.
        """

        try:
            if not self.imglocList.toPlainText():
                logger.critical('no image filepath')
                raise ValueError('empty file path')

            logger.debug(f'loading image from: {self.imglocList.toPlainText()}')

            self.img_size = os.path.getsize(self.imglocList.toPlainText())

            global image_raw
            image_raw = sdb.read_geotiff(self.imglocList.toPlainText())

            global bands_df
            bands_df = sdb.unravel(image_raw)

            self.loadImageLabel.setText(
                os.path.split(self.imglocList.toPlainText())[1]
            )

            logger.info(f'load image successfully of size: {self.img_size} B')
            logger.info(f'image CRS: {image_raw.rio.crs}')
        except ValueError as e:
            if 'empty file path' in str(e):
                self.loadImageDialog.close()
                self.warningWithClear(
                    'No data loaded. Please load your data!'
                )
                self.loadImageWindow()


    def loadSampleWindow(self):
        """
        Sample loading User Interface
        """

        self.loadSampleDialog = QDialog()
        self.loadSampleDialog.setWindowTitle('Load Sample')
        self.loadSampleDialog.setWindowIcon(
            QIcon(resource_path('icons/load-pngrepo-com.png'))
        )

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
        """
        Loading selected sample and retrieve file size.
        Then, some or all data on selected sample to the widget.
        """

        try:
            if not self.samplelocList.toPlainText():
                logger.critical('no sample filepath')
                raise ValueError('empty file path')

            logger.debug(
                f'loading sample data from: {self.samplelocList.toPlainText()}'
            )

            global sample_size
            sample_size = os.path.getsize(self.samplelocList.toPlainText())

            global sample_raw
            sample_raw = sdb.read_shapefile(self.samplelocList.toPlainText())

            proc_op_dict['selection']['Attribute Selection']['parameters'].update({
                'header': '',
                'group': ''
            })

            logger.debug('reset attribute selection parameters for new dataset')

            self.loadSampleLabel.setText(os.path.split(
                self.samplelocList.toPlainText())[1]
            )

            if (sample_raw.geom_type != 'Point').any():
                logger.critical('sample is not point type')
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

                if self.showCheckBox.isChecked() == True:
                    data = raw
                else:
                    data = raw.head(100)

                self.depthHeaderCB.clear()
                self.depthHeaderCB.addItems(data.columns)

                self.table.setColumnCount(len(data.columns))
                self.table.setRowCount(len(data.index))

                for h in range(len(data.columns)):
                    self.table.setHorizontalHeaderItem(
                        h, QTableWidgetItem(data.columns[h])
                    )

                for i in range(len(data.index)):
                    for j in range(len(data.columns)):
                        self.table.setItem(
                            i, j, QTableWidgetItem(str(data.iloc[i, j]))
                        )

                self.table.resizeColumnsToContents()
                self.table.resizeRowsToContents()

                logger.info(
                    f'load sample data successfully of size: {sample_size} B'
                )
                logger.info(f'sample CRS: {sample_raw.crs}')
        except ValueError as e:
            if 'empty file path' in str(e):
                self.loadSampleDialog.close()
                self.warningWithClear(
                    'No data loaded. Please load your data!'
                )
                self.loadSampleWindow()


    def methodOptionWindow(self):
        """
        Generic method option window
        """

        method = self.methodCB.currentText()
        method_options = option_pool['method'][method]
        
        optionDialog = QDialog()
        optionDialog.setWindowTitle(f'Options ({acronym(method)})')
        optionDialog.setWindowIcon(
            QIcon(resource_path('icons/setting-tool-pngrepo-com.png'))
        )

        grid = QGridLayout()
        row = 1
        self.option_widgets = {}

        for param, value in method_options['model_parameters'].items():
            label = QLabel(param.replace('_', ' ').title() + ':')
            grid.addWidget(label, row, 1, 1, 2)

            if isinstance(value, bool):
                widget = QComboBox()
                widget.addItems(['True', 'False'])
                widget.setCurrentText(str(value))
            elif isinstance(value, int):
                widget = QSpinBox()
                widget.setRange(1, 10000)
                widget.setValue(value)
                widget.setAlignment(Qt.AlignRight)
            elif isinstance(value, str):
                widget = QComboBox()
                set_name = f"{param}_set"
                if set_name in method_options:
                    widget.addItems(method_options[set_name])
                else:
                    widget.addItems([value])
                widget.setCurrentText(value)

            self.option_widgets[param] = widget
            grid.addWidget(widget, row, 3, 1, 2)
            row += 1

        cancelButton = QPushButton('Cancel')
        cancelButton.clicked.connect(optionDialog.close)
        loadButton = QPushButton('Load')
        loadButton.clicked.connect(self.loadMethodOptionAction)
        loadButton.clicked.connect(optionDialog.close)

        grid.addWidget(loadButton, row, 3, 1, 1)
        grid.addWidget(cancelButton, row, 4, 1, 1)

        optionDialog.setLayout(grid)
        optionDialog.exec_()


    def loadMethodOptionAction(self) -> None:
        """
        Update settings for any method and save them
        """

        method = self.methodCB.currentText()

        for param, widget in self.option_widgets.items():
            if isinstance(widget, QComboBox):
                value = (self.str2bool(widget.currentText()) 
                        if widget.currentText() in ('True', 'False')
                        else widget.currentText())
            else:
                value = widget.value()
                
            option_pool['method'][method]['model_parameters'][param] = value
        logger.info(f'{method} parameters updated')

        self.saveSettings()


    def processingOptionWindow(self):
        """
        Processing option User Interface
        """

        try:
            self.processingOptionDialog = QDialog()
            self.processingOptionDialog.setWindowTitle('Processing Options')
            self.processingOptionDialog.setWindowIcon(
                QIcon(resource_path('icons/setting-tool-pngrepo-com.png'))
            )

            grid = QGridLayout()
            row = 1

            backendLabel = QLabel('Parallel Backend:')
            self.backendCB = QComboBox()
            self.backendCB.addItems(proc_op_dict['backend_set'])
            self.backendCB.setCurrentText(proc_op_dict['backend'])
            grid.addWidget(backendLabel, row, 1, 1, 2)
            grid.addWidget(self.backendCB, row, 3, 1, 2)

            row += 1
            njobsLabel = QLabel('Processing Cores:')
            self.njobsSB = QSpinBox()
            self.njobsSB.setRange(-100, 100)
            self.njobsSB.setValue(proc_op_dict['n_jobs'])
            self.njobsSB.setAlignment(Qt.AlignRight)
            grid.addWidget(njobsLabel, row, 1, 1, 2)
            grid.addWidget(self.njobsSB, row, 3, 1, 2)

            row += 1
            self.currentSelection = proc_op_dict['selection'][
                self.trainSelectCB.currentText()
            ]
            self.selection_widgets = {}

            for param, value in self.currentSelection['parameters'].items():
                label = QLabel(param.replace('_', ' ').title() + ':')
                grid.addWidget(label, row, 1, 1, 2)

                if param == 'train_size':
                    widget = QDoubleSpinBox()
                    widget.setRange(0.1, 0.9)
                    widget.setSingleStep(0.05)
                    widget.setDecimals(2)
                    widget.setValue(value)
                    widget.setSuffix('')
                    widget.setAlignment(Qt.AlignRight)
                elif param == 'random_state':
                    widget = QSpinBox()
                    widget.setRange(0, 1000)
                    widget.setValue(value)
                    widget.setAlignment(Qt.AlignRight)
                elif param in ('header', 'group'):
                    widget = QComboBox()
                    if param == 'header':
                        object_only = sample_raw.select_dtypes(include=['object'])
                        widget.addItems(object_only.columns)
                        widget.activated.connect(self.updateGroupSelection)
                        if value:
                            widget.setCurrentText(value)
                        else:
                            widget.setCurrentText(widget.itemText(0))
                    elif param == 'group':
                        header = self.currentSelection[
                            'parameters'
                        ].get('header', '')
                        if header:
                            groups = list(sample_raw.groupby(header).groups.keys())
                            widget.addItems(groups)
                            if value:
                                widget.setCurrentText(value)
                            elif groups:
                                widget.setCurrentText(groups[0])

                self.selection_widgets[param] = widget
                grid.addWidget(widget, row, 3, 1, 2)
                row += 1

            cancelButton = QPushButton('Cancel')
            cancelButton.clicked.connect(self.processingOptionDialog.close)
            loadButton = QPushButton('Load')
            loadButton.clicked.connect(self.loadProcessingOptionAction)
            loadButton.clicked.connect(self.processingOptionDialog.close)

            grid.addWidget(loadButton, row, 3, 1, 1)
            grid.addWidget(cancelButton, row, 4, 1, 1)

            self.processingOptionDialog.setLayout(grid)

            self.processingOptionDialog.exec_()
        except NameError:
            self.warningWithClear(
                'No depth sample loaded. Please load your depth sample!'
            )


    def loadProcessingOptionAction(self):
        """
        Loading defined processing option input
        """

        if self.njobsSB.value() == 0:
            self.processingOptionDialog.close()
            self.warningWithoutClear(
                'Do not insert zero on Processing Cores!'
            )
            self.processingOptionWindow()
            return

        proc_op_dict['backend'] = self.backendCB.currentText()
        proc_op_dict['n_jobs'] = self.njobsSB.value()

        current_selection_name = self.trainSelectCB.currentText()
        if current_selection_name in proc_op_dict['selection']:
            selection_params = proc_op_dict['selection'][
                current_selection_name
            ]['parameters']
            for param, widget in self.selection_widgets.items():
                if isinstance(widget, QDoubleSpinBox):
                    selection_params[param] = widget.value()
                elif isinstance(widget, QSpinBox):
                    selection_params[param] = widget.value()
                elif isinstance(widget, QComboBox):
                    selection_params[param] = widget.currentText()
            logger.info('processing options updated')
            logger.debug(f'updated parameters: {selection_params}')
        else:
            logger.error(f'selection "{current_selection_name}" not found.')

        self.saveSettings()


    def updateGroupSelection(self):
        """
        Update list in group selection when header selection changes
        """

        try:
            selected_header = self.selection_widgets['header'].currentText()

            if selected_header:
                object_only = sample_raw.select_dtypes(include=['object'])
                group_list = list(object_only.groupby(selected_header).groups.keys())

                group_widget = self.selection_widgets['group']
                group_widget.clear()
                group_widget.addItems(group_list)

                if group_list:
                    group_widget.setCurrentText(group_list[0])
                logger.debug(
                    f'group widget updated "{selected_header}": {group_list}'
                )

        except Exception as e:
            logger.error(f'failed to update groups: {e}')


    def predict(self):
        """
        Sending parameters and inputs from widget to Process Class
        """

        logging.debug('Sending user inputs to process class')
        self.saveSettings()

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
            'depth_direction': self.depthDirectionCB.currentText(), 
            'limit_state': self.limitCheckBox.isChecked(),
            'limit_a': self.limitADSB.value(),
            'limit_b': self.limitBDSB.value(),
            'method': self.methodCB.currentText(),
            'train_select': self.trainSelectCB.currentText(),
            'selection': proc_op_dict['selection'][
                self.trainSelectCB.currentText()
            ]['parameters']
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


    def timeCounting(self, time_text: List[Union[datetime.datetime, str]]) -> None:
        """
        Receive time value on every step and its corresponding processing
        text to show in result text browser and increase progress bar.
        """

        time_list.append(time_text[0])
        self.resultText.append(time_text[1])
        self.progressBar.setValue(self.progressBar.value() + 1)

        if self.progressBar.value() == self.progressBar.maximum():
            self.completeDialog()


    def results(self, result_dict: Dict[str, Any]) -> None:
        """
        Recieve processing results and filter the predicted value to depth
        limit window (if enabled).
        Counting runtimes using saved time values and printing result info.
        """

        global end_results
        end_results = result_dict

        rmse, mae, r2 = end_results['rmse'], end_results['mae'], end_results['r2']

        train_size_percent = (
            end_results['train'].shape[0] / (
                end_results['train'].shape[0] + end_results['test'].shape[0]
            )
            * 100
        )
        train_size_percent = round(train_size_percent, 2)

        used_sample_size = (
            end_results['train'].shape[0] + end_results['test'].shape[0]
        )

        if self.limitCheckBox.isChecked() == False:
            print_limit = (
                f'Depth Limit:\t\tfrom {self.limitADSB.value():.1f} m '
                f'to {self.limitBDSB.value():.1f} m'
            )
        else:
            print_limit = (
                'Depth Limit:\t\tDisabled'
            )

        time_array = np.array(time_list)
        time_diff = time_array[1:] - time_array[:-1]
        runtime = np.append(time_diff, time_list[-1] - time_list[0])

        global print_result_info
        print_result_info = (
            f'Software Version:\t{SDB_GUI_VERSION}\n\n'
            f'Image Input:\t\t{self.imglocList.toPlainText()} '
            f'({round(self.img_size / 2**20, 2)} MiB)\n'
            f'Sample Data:\t\t{self.samplelocList.toPlainText()} '
            f'({round(sample_size / 2**20, 2)} MiB)\n'
            f'Selected Header:\t{self.depthHeaderCB.currentText()}\n'
            f'Depth Direction:\t\t{self.depthDirectionCB.currentText()}\n\n'
            f'{print_limit}\n'
            f'Used Sample:\t\t{used_sample_size} points '
            f'({round(used_sample_size / sample_raw.shape[0] * 100, 2)}% '
            f'of all sample)\n'
            f'Train Data:\t\t{end_results["train"].shape[0]} points '
            f'({round(train_size_percent, 2)} % of used sample)\n'
            f'Test Data:\t\t{end_results["test"].shape[0]} points '
            f'({round((100 - train_size_percent), 2)} % of used sample)\n\n'
            f'Method:\t\t{self.methodCB.currentText()}\n'
            f'{print_parameters_info}\n'
            f'RMSE:\t\t{round(rmse, 3)}\n'
            f'MAE:\t\t{round(mae, 3)}\n'
            f'R\u00B2:\t\t{round(r2, 3)}\n\n'
            f'Train Test Selection:\t{self.trainSelectCB.currentText()}\n'
            f'Parallel Backend:\t{proc_op_dict["backend"]}\n'
            f'Processing Cores:\t{proc_op_dict["n_jobs"]}\n'
            f'Clipping Runtime:\t{runtime[0]}\n'
            f'Filtering Runtime:\t{runtime[1]}\n'
            f'Splitting Runtime:\t{runtime[2]}\n'
            f'Modeling Runtime:\t{runtime[3]}\n'
            f'Evaluation Runtime:\t{runtime[4]}\n'
            f'Overall Runtime:\t{runtime[5]}\n\n'
            f'CRS:\t\t{image_raw.rio.crs}\n'
            f'Dimensions:\t\t{image_raw.rio.width} x '
            f'{image_raw.rio.height} pixels\n'
            f'Pixel Size:\t\t{abs(image_raw.rio.resolution()[0])} , '
            f'{abs(image_raw.rio.resolution()[1])}\n\n'
        )

        self.resultText.setText(print_result_info)


    def stopProcess(self):
        """
        Stop processing and clear result info and progress bar
        """

        if hasattr(self, 'sdbProcess') and self.sdbProcess.isRunning():
            self.sdbProcess.stop()
            self.sdbProcess.wait()
            self.resultText.clear()
            self.progressBar.setValue(0)
            self.resultText.setText('Processing has been stopped!')


    def copyLogFile(self):
        """
        Copy the original log file to the save location
        """

        if hasattr(self, 'savelocList') and self.savelocList.toPlainText():
            try:
                save_path = f'{
                    os.path.splitext(self.savelocList.toPlainText())[0]
                }.log'

                with open(LOG_NAME, 'r') as source, open(save_path, 'w') as target:
                    target.write(source.read())
                logger.info(f'log file copied to: {save_path}')
            except Exception as e:
                logger.error(f'failed to copy log file: {e}')


    def closeEvent(self, event):
        """
        Called when the widget is closed
        """

        self.saveSettings()

        logger.info(f'SDB GUI {SDB_GUI_VERSION} is closing')
        if hasattr(self, 'sdbProcess') and self.sdbProcess.isRunning():
            logger.info('stopping running process')
            self.sdbProcess.stop()
            self.sdbProcess.wait()

        self.copyLogFile()
        event.accept()


    def warningWithClear(self, warning_text):
        """
        Show warning dialog and customized warning text
        and then clear result info and progress bar after closing
        """

        warning = QErrorMessage()
        warning.setWindowTitle('WARNING')
        warning.setWindowIcon(
            QIcon(resource_path('icons/warning-pngrepo-com.png'))
        )
        warning.showMessage(warning_text)

        warning.exec_()
        self.resultText.clear()
        self.progressBar.setValue(0)


    def warningWithoutClear(self, warning_text):
        """
        Show warning dialog and customized warning text
        without clearing result info and progress bar after closing
        """

        warning = QErrorMessage()
        warning.setWindowTitle('WARNING')
        warning.setWindowIcon(
            QIcon(resource_path('icons/warning-pngrepo-com.png'))
        )
        warning.showMessage(warning_text)

        warning.exec_()


    def completeDialog(self):
        """
        Showing complete pop up dialog
        """

        complete = QDialog()
        complete.setWindowTitle('Complete')
        complete.setWindowIcon(
            QIcon(resource_path('icons/complete-pngrepo-com.png'))
        )
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
        """
        Saving option window
        """

        self.saveOptionDialog = QDialog()
        self.saveOptionDialog.setWindowTitle('Save Options')
        self.saveOptionDialog.setWindowIcon(
            QIcon(resource_path('icons/load-pngrepo-com.png'))
        )

        format_list = ['GeoTIFF (*.tif)','ASCII Gridded XYZ (*.xyz)']
        format_list.sort()

        dataTypeLabel = QLabel('Data Type:')
        self.dataTypeCB = QComboBox()
        self.dataTypeCB.addItems(format_list)
        self.dataTypeCB.setCurrentText('GeoTIFF (*.tif)')

        direction_list = list(DEPTH_DIR_DICT.keys())

        depthDirectionSaveLabel = QLabel('Depth Direction:')
        self.depthDirectionSaveCB = QComboBox()
        self.depthDirectionSaveCB.addItems(direction_list)

        medianFilterLabel = QLabel('Median Filter Size:')
        self.medianFilterSB = QSpinBox()
        self.medianFilterSB.setRange(3, 33)
        self.medianFilterSB.setValue(3)
        self.medianFilterSB.setSingleStep(2)
        self.medianFilterSB.setAlignment(Qt.AlignRight)

        self.medianFilterCheckBox = QCheckBox('Disable Median Filter')
        self.medianFilterCheckBox.setChecked(False)

        saveFileButton = QPushButton('Save File Location')
        saveFileButton.clicked.connect(
            lambda:self.fileDialog(
                command=QFileDialog.getSaveFileName,
                window_text='Save File',
                file_type=self.dataTypeCB.currentText(),
                text_browser=self.savelocList
            )
        )

        locLabel = QLabel('Location:')
        self.savelocList = QTextBrowser()

        self.scatterPlotCheckBox = QCheckBox('Save Scatter Plot')
        self.scatterPlotCheckBox.setChecked(False)

        self.trainTestDataCheckBox = QCheckBox('Save Training and Testing Data in')
        self.trainTestDataCheckBox.setChecked(False)

        self.trainTestFormatCB = QComboBox()
        self.trainTestFormatCB.addItems(['.csv', '.shp'])

        trainTestLabel = QLabel('format')

        self.saveDEMCheckBox = QCheckBox('Save DEM')
        self.saveDEMCheckBox.setChecked(True)

        self.reportCheckBox = QCheckBox('Save Report')
        self.reportCheckBox.setChecked(True)

        cancelButton = QPushButton('Cancel')
        cancelButton.clicked.connect(self.saveOptionDialog.close)
        saveButton = QPushButton('Save')
        saveButton.clicked.connect(self.saveAction)
        saveButton.clicked.connect(self.saveOptionDialog.close)

        grid = QGridLayout()
        grid.addWidget(dataTypeLabel, 1, 1, 1, 1)
        grid.addWidget(self.dataTypeCB, 1, 2, 1, 3)

        grid.addWidget(depthDirectionSaveLabel, 2, 1, 1, 1)
        grid.addWidget(self.depthDirectionSaveCB, 2, 2, 1, 3)

        grid.addWidget(medianFilterLabel, 3, 1, 1, 1)
        grid.addWidget(self.medianFilterSB, 3, 2, 1, 1)
        grid.addWidget(self.medianFilterCheckBox, 3, 3, 1, 2)

        grid.addWidget(saveFileButton, 4, 1, 1, 4)

        grid.addWidget(locLabel, 5, 1, 1, 4)
        grid.addWidget(self.savelocList, 6, 1, 1, 4)

        grid.addWidget(self.scatterPlotCheckBox, 7, 1, 1, 2)

        grid.addWidget(self.trainTestDataCheckBox, 8, 1, 1, 2)
        grid.addWidget(self.trainTestFormatCB, 8, 3, 1, 1)
        grid.addWidget(trainTestLabel, 8, 4, 1, 1)

        grid.addWidget(self.saveDEMCheckBox, 9, 1, 1, 1)
        grid.addWidget(self.reportCheckBox, 9, 2, 1, 1)
        grid.addWidget(saveButton, 9, 3, 1, 1)
        grid.addWidget(cancelButton, 9, 4, 1, 1)

        self.saveOptionDialog.setLayout(grid)

        self.saveOptionDialog.exec_()


    def saveAction(self):
        """
        Saving predicted depth, training and testing data, and/or report into file.
        Applying median filter (or not) to the predicted depth array before saving.
        """

        try:
            daz_filtered = end_results['daz_predict'].copy()

            if self.medianFilterCheckBox.isChecked() == False:
                print_filter_info = (
                    f'Median Filter Size:\t{self.medianFilterSB.value()}'
                )

                daz_filtered.values[0] = sdb.median_filter(
                    daz_filtered.values[0],
                    filter_size=self.medianFilterSB.value()
                )
                daz_filtered.band_name.values[0] = 'filtered'
            else:
                print_filter_info = 'Median Filter Size:\tDisabled'

            train_df_copy = end_results['train'].copy()
            test_df_copy = end_results['test'].copy()

            if DEPTH_DIR_DICT[self.depthDirectionSaveCB.currentText()][1]:
                daz_filtered.values[0] *=-1
                test_df_copy['z'] *=-1
                test_df_copy['z_predict'] *=-1
                train_df_copy['z'] *=-1

            if not self.savelocList.toPlainText():
                raise ValueError('empty save location')

            sdb.write_geotiff(
                daz_filtered,
                self.savelocList.toPlainText()
            )

            if self.saveDEMCheckBox.isChecked() == True:
                new_img_size = os.path.getsize(self.savelocList.toPlainText())
                print_dem_info = (
                    f'{print_filter_info}\n\n'
                    f'DEM Output:\t\t{self.savelocList.toPlainText()} '
                    f'({round(new_img_size / 2**10 / 2**10, 2)} MiB)\n'
                )
            elif self.saveDEMCheckBox.isChecked() == False:
                os.remove(self.savelocList.toPlainText())
                print_dem_info = (
                    'DEM Output:\t\tNot Saved\n'
                )

            if self.trainTestDataCheckBox.isChecked() == True:
                train_save_loc = (
                    f'{os.path.splitext(self.savelocList.toPlainText())[0]}'
                    f'_train{self.trainTestFormatCB.currentText()}'
                )
                test_save_loc = (
                    f'{os.path.splitext(self.savelocList.toPlainText())[0]}'
                    f'_test{self.trainTestFormatCB.currentText()}'
                )

                if self.trainTestFormatCB.currentText() == '.csv':
                    train_df_copy.to_csv(train_save_loc, index=False)
                    test_df_copy.to_csv(test_save_loc, index=False)
                elif self.trainTestFormatCB.currentText() == '.shp':
                    sdb.write_shapefile(
                        train_df_copy,
                        train_save_loc,
                        x_col_name='x',
                        y_col_name='y',
                        crs=end_results['sample_gdf'].crs
                    )
                    sdb.write_shapefile(
                        test_df_copy,
                        test_save_loc,
                        x_col_name='x',
                        y_col_name='y',
                        crs=end_results['sample_gdf'].crs
                    )

                train_data_size = os.path.getsize(train_save_loc)
                test_data_size = os.path.getsize(test_save_loc)

                print_train_test_info = (
                    f'Train Data Output:\t{train_save_loc} '
                    f'({round(train_data_size / 2**10 / 2**10, 2)} MiB)\n'
                    f'Test Data output:\t{test_save_loc} '
                    f'({round(test_data_size / 2**10 / 2**10, 2)} MiB)\n'
                )
            elif self.trainTestDataCheckBox.isChecked() == False:
                print_train_test_info = (
                    'Train Data Output:\tNot Saved\n'
                    'Test Data output:\tNot Saved\n'
                )

            if self.scatterPlotCheckBox.isChecked() == True:
                scatter_plot_loc = (
                    os.path.splitext(self.savelocList.toPlainText())[0] +
                    '_scatter_plot.png'
                )
                scatter_plot = sdb.scatter_plotter(
                    true_val=test_df_copy['z'],
                    pred_val=test_df_copy['z_predict'],
                    title=self.methodCB.currentText()
                )
                scatter_plot[0].savefig(scatter_plot_loc)

                scatter_plot_size = os.path.getsize(scatter_plot_loc)

                print_scatter_plot_info = (
                    f'Scatter Plot:\t{scatter_plot_loc} '
                    f'({round(scatter_plot_size / 2**10, 2)} KiB)\n'
                )
            elif self.scatterPlotCheckBox.isChecked() == False:
                print_scatter_plot_info = 'Scatter Plot:\tNotSaved\n'

            self.resultText.append(print_dem_info)
            self.resultText.append(print_train_test_info)
            self.resultText.append(print_scatter_plot_info)

            if self.reportCheckBox.isChecked() == True:
                report_save_loc = (
                    os.path.splitext(self.savelocList.toPlainText())[0] +
                    '_report.txt'
                )
                report = open(report_save_loc, 'w')

                report.write(
                    print_result_info +
                    print_dem_info +
                    print_train_test_info +
                    print_scatter_plot_info
                )
        except ValueError as e:
            if 'Allowed value: >= 3 or odd numbers' in str(e):
                self.saveOptionDialog.close()
                self.warningWithoutClear(
                    'Please insert odd number on filter size!'
                )
                self.saveOptionWindow()
            elif 'empty save location' in str(e):
                self.saveOptionDialog.close()
                self.warningWithoutClear(
                    'Please insert save location!'
                )
                self.saveOptionWindow()


    def licensesDialog(self):
        """
        Showing the license of SDB GUI and another library licenses
        """

        licenses = QDialog()
        licenses.setWindowTitle('Licenses')
        licenses.setWindowIcon(
            QIcon(resource_path('icons/information-pngrepo-com.png'))
        )
        licenses.resize(600, 380)

        license_dict = {
            'SDB GUI': 'LICENSE',
            'NumPy': 'licenses/numpy_license',
            'Scipy': 'licenses/scipy_license',
            'Pandas': 'licenses/pandas_license',
            'Xarray': 'licenses/xarray_license',
            'Rioxarray': 'licenses/rioxarray_license',
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
        """
        Selecting license file location
        """

        license_file = open(resource_path(location), 'r')
        self.licenseText.setText(license_file.read())



class Process(QThread):
    """
    Data processing class of SDB GUI.
    Sending inputs from SDBWidget to process in the background
    so the GUI won't freeze while processing data.
    """

    thread_signal = pyqtSignal(dict)
    time_signal = pyqtSignal(list)
    warning_with_clear = pyqtSignal(str)
    warning_without_clear = pyqtSignal(str)

    def __init__(self):

        QThread.__init__(self)

        self._is_running: bool = True


    def inputs(self, input_dict):
        """
        Pooling inputs from widget
        """

        self.depth_label = input_dict['depth_label']
        self.depth_direction = input_dict['depth_direction']
        self.limit_state = input_dict['limit_state']
        self.limit_a_value = input_dict['limit_a']
        self.limit_b_value = input_dict['limit_b']
        self.method = input_dict['method']
        self.train_select = input_dict['train_select']
        self.selection = input_dict['selection']


    def preprocess(self):
        """
        Preparing input values from widget to use on 
        training models and predicting depth by reprojecting 
        depth sample CRS, sampling raster value and depth value, 
        and then limiting or not limiting depth value.
        """


        if not self._is_running:
            return None

        logger.debug('preprocess started by clip and/or reproject sample data')
        time_start = datetime.datetime.now()
        start_list = [time_start, 'Clipping and Reprojecting...\n']
        self.time_signal.emit(start_list)
        clipped_sample = sdb.clip_vector(image_raw, sample_raw)

        if not self._is_running:
            return None

        logger.debug('filter depth sample input')
        time_clip = datetime.datetime.now()
        clip_list = [time_clip, 'Depth Filtering...\n']
        self.time_signal.emit(clip_list)
        depth_filtered_sample = sdb.in_depth_filter(
            vector=clipped_sample,
            header=self.depth_label,
            depth_direction=DEPTH_DIR_DICT[self.depth_direction][0],
            disable_depth_filter=self.limit_state,
            upper_limit=self.limit_a_value,
            lower_limit=self.limit_b_value
        )

        if not self._is_running:
            return None

        time_depth_filter = datetime.datetime.now()
        depth_filter_list = [time_depth_filter, 'Split Train and Test...\n']
        self.time_signal.emit(depth_filter_list)
        if self.train_select == 'Random Selection':
            logger.info('split depth sample randomly')
            f_train, f_test, z_train, z_test = sdb.split_random(
                raster=image_raw,
                vector=depth_filtered_sample,
                header=self.depth_label,
                train_size=self.selection['train_size'],
                random_state=self.selection['random_state']
            )
        elif self.train_select == 'Attribute Selection':
            logger.info('split depth sample by selected attribute')
            f_train, f_test, z_train, z_test = sdb.split_attribute(
                raster=image_raw,
                vector=depth_filtered_sample,
                depth_header=self.depth_label,
                split_header=self.selection['header'],
                group_name=self.selection['group']
            )

        results = {
            'f_train': f_train,
            'f_test': f_test,
            'z_train': z_train,
            'z_test': z_test,
            'sample_gdf': depth_filtered_sample
        }

        logging.debug('preprocess ended')
        return results


    def predict(self, method):
        """
        Preparing prediction using selected method/model
        and saving selected parameters for report
        """

        if not self._is_running:
            return None

        results = self.preprocess()

        logger.info(f'prediction started using {method}')
        if results is None or not self._is_running:
            return None

        time_split = datetime.datetime.now()
        split_list = [time_split, 'Modeling...\n']
        self.time_signal.emit(split_list)

        model_parameters = option_pool['method'][method]['model_parameters']
        logger.info(f'model parameters: {model_parameters}')

        global print_parameters_info
        print_parameters_info = ''
        for key, value in model_parameters.items():
            print_parameters_info += (
                f'{key.replace('_', ' ').title()}:\t\t{value}\n'
            )

        z_predict = sdb.prediction(
            model=method,
            unraveled_band=bands_df,
            features_train=results['f_train'].drop(columns=['x', 'y']),
            label_train=results['z_train'],
            backend=proc_op_dict['backend'],
            n_jobs=proc_op_dict['n_jobs'],
            **model_parameters
        )

        if not self._is_running:
            return None

        results.update({'z_predict': z_predict})

        logger.debug('prediction ended')
        return results


    def run(self):
        """
        Taking pre processed input and chosen method, then 
        fitting training data to chosen model and generate prediction
        based on trained model.
        """

        try:
            results = self.predict(method=self.method)

            if results is None or not self._is_running:
                return None
            logger.debug('run started')

            time_model = datetime.datetime.now()
            model_list = [time_model, 'Evaluating...\n']
            self.time_signal.emit(model_list)

            logger.debug('reshape prediction array to raster shape')
            az_predict = sdb.reshape_prediction(
                array=results['z_predict'],
                raster=image_raw
            )

            logger.debug('convert prediction array to dataarray')
            daz_predict = sdb.array_to_dataarray(
                array=az_predict,
                data_array=image_raw
            )

            daz_predict = daz_predict.assign_coords(
                band_name=('band', ['original'])
            )

            logger.debug('sampling predictin based on test data coordinates')
            dfz_predict = sdb.point_sampling(
                daz_predict,
                x=results['f_test'].x,
                y=results['f_test'].y,
                include_xy=False
            )

            logger.info('evaluating prediction')
            rmse, mae, r2 = sdb.evaluate(
                true_val=results['z_test'],
                pred_val=dfz_predict['band_1'].to_numpy()
            )
            logger.info(f'RMSE: {rmse}, MAE: {mae}, R2: {r2}')

            time_test = datetime.datetime.now()
            test_list = [time_test, 'Done.']
            self.time_signal.emit(test_list)

            train_df = results['f_train'].copy()
            train_df['z'] = results['z_train'].copy()

            test_df = results['f_test'].copy()
            test_df['z'] = results['z_test'].copy()
            test_df['z_predict'] = dfz_predict['band_1'].copy()

            results.update({
                'daz_predict': daz_predict,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'train': train_df,
                'test': test_df
            })

            logger.debug('run ended and sending results')
            self.thread_signal.emit(results)
        except NameError:
            self.warning_with_clear.emit(
                'No image data loaded. Please load your image data!'
            )
        except IndexError:
            self.warning_with_clear.emit(
                'Depth sample is out of image boundary'
            )
        except KeyError:
            self.warning_with_clear.emit(
                'Please select attribute header and group in Processing Options'
            )


    def stop(self):
        """
        Stop the processing thread.
        """
        self._is_running: bool = False
        self.quit()
        self.wait()



def main():

    global sdb_gui
    sdb_gui = SDBWidget()
    sdb_gui.show()


def default_values():
    """
    Default values container
    """

    random_selection = {
        'name': 'Random Selection',
        'parameters': OrderedDict([
            ('train_size', 0.75),
            ('random_state', 0)
        ])
    }

    attribute_selection = {
        'name': 'Attribute Selection',
        'parameters': OrderedDict([
            ('header', ''),
            ('group', '')
        ])
    }

    proc_op_dict = {
        'depth_limit': {
            'disable': False,
            'upper': 0.0,
            'lower': -15.0
        },
        'backend': 'threading',
        'n_jobs': -2,
        'selection' : OrderedDict([
            (random_selection['name'], random_selection),
            (attribute_selection['name'], attribute_selection)
        ]),
        'backend_set': (
            'loky', 'threading', 'multiprocessing'
        )
    }

    knn_op_dict = {
        'name': 'K-Nearest Neighbors',
        'model_parameters': OrderedDict([
            ('n_neighbors', 5),
            ('weights', 'distance'),
            ('algorithm', 'auto'),
            ('leaf_size', 30)
        ]),
        'weights_set': (
            'uniform', 'distance'
        ),
        'algorithm_set': (
            'auto', 'ball_tree', 'kd_tree', 'brute'
        )
    }

    mlr_op_dict = {
        'name': 'Multiple Linear Regression',
        'model_parameters': OrderedDict([
            ('fit_intercept', True),
            ('copy_X', True)
        ])
    }

    rf_op_dict = {
        'name': 'Random Forest',
        'model_parameters': OrderedDict([
            ('n_estimators', 300),
            ('criterion', 'squared_error'),
            ('bootstrap', True)
        ]),
        'criterion_set': (
            'squared_error', 'absolute_error', 'poisson', 'friedman_mse'
        )
    }

    default_dict = {
        'processing': proc_op_dict,
        'method': {
            knn_op_dict['name']: knn_op_dict,
            mlr_op_dict['name']: mlr_op_dict,
            rf_op_dict['name']: rf_op_dict
        }
    }

    return default_dict


def resource_path(relative_path):
    """
    Get the absolute path to the resource, works for dev and for PyInstaller
    """

    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS # type: ignore
    except Exception:
        base_path = os.path.abspath('.')
    return os.path.join(base_path, relative_path)


def acronym(phrase: str) -> str:
    """
    Generate an acronym from a phrase by taking the first letter of each word
    and converting it to uppercase.
    """

    words = re.split(r'[\s\-]+', phrase)
    return ''.join(word[0].upper() for word in words if word and word[0].isalpha())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    logger.info(f'SDB GUI {SDB_GUI_VERSION} started')
    main()
    exit_code = app.exec_()
    logger.info(f'SDB GUI {SDB_GUI_VERSION} exited with code {exit_code}')
    sys.exit(exit_code)