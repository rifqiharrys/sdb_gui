import datetime
import logging
import pprint
import re
import sys
import webbrowser
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from PyQt5.QtCore import QSettings, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog,
                             QDoubleSpinBox, QErrorMessage, QFileDialog,
                             QGridLayout, QLabel, QMessageBox, QProgressBar,
                             QPushButton, QScrollArea, QSpinBox, QTableWidget,
                             QTableWidgetItem, QTextBrowser, QVBoxLayout,
                             QWidget)

import sdb
from sdb.gui_utils import acronym, str2bool, to_title

## CONSTANTS ##
SDB_GUI_VERSION: str = '4.1.0'
LOG_NAME: str = 'sdb_gui.log'
PROGRESS_STEP: int = 6
DEPTH_DIRECTION: Dict[str, Tuple[str, bool]] = {
    'Positive Up': ('up', False),
    'Positive Down': ('down', True),
}
SELECTION_TYPES: Dict[str, str] = {
    'RANDOM': 'Random Selection',
    'ATTRIBUTE': 'Attribute Selection'
}
EVALUATION_TYPES: Dict[str, bool] = {
    'Use Current Prediction': False,
    'Recalculate from Test Data': True,
}
DEM_FORMATS: List[str] = [
    'GeoTIFF (*.tif)',
    'ASCII Gridded XYZ (*.xyz)',
]
DEM_FORMATS.sort()
TRAIN_TEST_SAVE: Dict[str, bool] = {
    '.csv': True,
    '.shp': True,
    '.gpkg': False,
}


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
        self._assignSettings()
        logger.debug(
            f'initial options: \n{pprint.pformat(option_pool, width=200)}'
        )

        self.dir_path = Path(self.settings.value('last_directory', Path.home()))
        self.initUI()


    def initUI(self):
        """
        Initialize User Interface for SDB GUI Widget
        """

        self.setGeometry(300, 100, 480, 640)
        self.setWindowTitle(f'Satellite Derived Bathymetry v{SDB_GUI_VERSION}')
        self.setWindowIcon(QIcon(resource_path('icons/satellite.png')))

        mainLayout = QVBoxLayout()

        grid1 = QGridLayout()
        row_grid1 = 1
        loadImageButton = QPushButton('Load Image')
        loadImageButton.clicked.connect(self.loadImageWindow)
        grid1.addWidget(loadImageButton, row_grid1, 1, 1, 1)

        self.loadImageLabel = QLabel()
        self.loadImageLabel.setText('No Image Loaded')
        self.loadImageLabel.setAlignment(Qt.AlignCenter)
        grid1.addWidget(self.loadImageLabel, row_grid1, 2, 1, 3)

        row_grid1 += 1
        loadSampleButton = QPushButton('Load Sample')
        loadSampleButton.clicked.connect(self.loadSampleWindow)
        grid1.addWidget(loadSampleButton, row_grid1, 1, 1, 1)

        self.loadSampleLabel = QLabel()
        self.loadSampleLabel.setText('No Sample Loaded')
        self.loadSampleLabel.setAlignment(Qt.AlignCenter)
        grid1.addWidget(self.loadSampleLabel, row_grid1, 2, 1, 3)

        row_grid1 += 1
        depthHeaderLabel = QLabel('Depth Header:')
        grid1.addWidget(depthHeaderLabel, row_grid1, 1, 1, 1)

        self.depthHeaderCB = QComboBox()
        grid1.addWidget(self.depthHeaderCB, row_grid1, 2, 1, 1)

        depthDirectionLabel = QLabel('Depth Direction:')
        grid1.addWidget(depthDirectionLabel, row_grid1, 3, 1, 1)

        self.depthDirectionCB = QComboBox()
        direction_list = list(DEPTH_DIRECTION.keys())
        self.depthDirectionCB.addItems(direction_list)
        self.depthDirectionCB.setCurrentText(main_set['direction'])
        grid1.addWidget(self.depthDirectionCB, row_grid1, 4, 1, 1)

        row_grid1 += 1
        self.table = QTableWidget()
        scroll = QScrollArea()
        scroll.setWidget(self.table)
        grid1.addWidget(self.table, row_grid1, 1, 5, 4)

        mainLayout.addLayout(grid1)

        grid2 = QGridLayout()
        row_grid2 = 1
        limitLabel = QLabel('Depth Limit Window:')
        grid2.addWidget(limitLabel, row_grid2, 1, 1, 2)

        limitALabel = QLabel('Upper Limit:')
        grid2.addWidget(limitALabel, row_grid2, 3, 1, 1)

        self.limitADSB = QDoubleSpinBox()
        self.limitADSB.setRange(-100, 100)
        self.limitADSB.setDecimals(1)
        self.limitADSB.setValue(main_set['depth_limit']['upper'])
        self.limitADSB.setSuffix(' m')
        self.limitADSB.setAlignment(Qt.AlignRight)
        grid2.addWidget(self.limitADSB, row_grid2, 4, 1, 1)

        row_grid2 += 1
        self.limitCheckBox = QCheckBox('Disable Depth Limitation')
        self.limitCheckBox.setChecked(main_set['depth_limit']['disable'])
        grid2.addWidget(self.limitCheckBox, row_grid2, 1, 1, 2)

        limitBLabel = QLabel('Lower Limit:')
        grid2.addWidget(limitBLabel, row_grid2, 3, 1, 1)

        self.limitBDSB = QDoubleSpinBox()
        self.limitBDSB.setRange(-100, 100)
        self.limitBDSB.setDecimals(1)
        self.limitBDSB.setValue(main_set['depth_limit']['lower'])
        self.limitBDSB.setSuffix(' m')
        self.limitBDSB.setAlignment(Qt.AlignRight)
        grid2.addWidget(self.limitBDSB, row_grid2, 4, 1, 1)

        mainLayout.addLayout(grid2)

        grid3 = QGridLayout()
        row_grid3 = 1
        methodLabel = QLabel('Regression Method:')
        grid3.addWidget(methodLabel, row_grid3, 1, 1, 1)

        self.methodCB = QComboBox()
        method_list = list(option_pool['method'].keys())
        self.methodCB.addItems(method_list)
        self.methodCB.setCurrentText(main_set['method'])
        grid3.addWidget(self.methodCB, row_grid3, 2, 1, 1)

        self.optionsButton = QPushButton('Method Options')
        self.optionsButton.clicked.connect(
            lambda: self._methodOptionWindow()
        )
        grid3.addWidget(self.optionsButton, row_grid3, 3, 1, 1)

        processingOptionsButton = QPushButton('Processing Options')
        processingOptionsButton.clicked.connect(self._processingOptionWindow)
        grid3.addWidget(processingOptionsButton, row_grid3, 4, 1, 1)

        mainLayout.addLayout(grid3)

        grid4 = QGridLayout()
        row_grid4 = 1
        makePredictionButton = QPushButton('Generate Prediction')
        makePredictionButton.clicked.connect(self._predict)
        grid4.addWidget(makePredictionButton, row_grid4, 1, 1, 2)

        resetSettingsButton = QPushButton('Reset Settings')
        resetSettingsButton.clicked.connect(self.resetToDefault)
        grid4.addWidget(resetSettingsButton, row_grid4, 3, 1, 2)

        row_grid4 += 1
        stopProcessingButton = QPushButton('Stop Processing')
        stopProcessingButton.clicked.connect(self._stopProcess)
        grid4.addWidget(stopProcessingButton, row_grid4, 1, 1, 2)

        saveFileButton = QPushButton('Save Into File')
        saveFileButton.clicked.connect(self._saveOptionWindow)
        grid4.addWidget(saveFileButton, row_grid4, 3, 1, 2)

        row_grid4 += 1
        resultInfo = QLabel('Result Information')
        grid4.addWidget(resultInfo, row_grid4, 1, 1, 2)

        row_grid4 += 1
        self.resultText = QTextBrowser()
        self.resultText.setAlignment(Qt.AlignRight)
        grid4.addWidget(self.resultText, row_grid4, 1, 1, 4)

        row_grid4 += 4
        self.progressBar = QProgressBar()
        self.progressBar.setFormat('%p%')
        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(PROGRESS_STEP)
        grid4.addWidget(self.progressBar, row_grid4, 1, 1, 4)

        row_grid4 += 1
        releaseButton =  QPushButton('Releases')
        releaseButton.clicked.connect(lambda: webbrowser.open(
            'https://github.com/rifqiharrys/sdb_gui/releases'
        ))
        grid4.addWidget(releaseButton, row_grid4, 1, 1, 1)

        licensesButton = QPushButton('Licenses')
        licensesButton.clicked.connect(self.licensesDialog)
        grid4.addWidget(licensesButton, row_grid4, 2, 1, 2)

        readmeButton = QPushButton('Readme')
        readmeButton.clicked.connect(lambda: webbrowser.open(
            'https://github.com/rifqiharrys/sdb_gui/blob/main/README.md'
        ))
        grid4.addWidget(readmeButton, row_grid4, 4, 1, 1)

        mainLayout.addLayout(grid4)

        self.setLayout(mainLayout)


    def loadSettings(self) -> dict:
        """
        Load settings or return defaults if none exist
        """

        try:
            if self.settings.value('options'):
                saved_settings = self.settings.value('options')

                try:
                    _ = saved_settings['main']['direction']
                    _ = saved_settings['main']['depth_limit']
                    _ = saved_settings['main']['method']

                    _ = saved_settings['save']

                    _ = saved_settings['processing']

                    _ = saved_settings['method']

                    return saved_settings
                except KeyError as e:
                    logger.warning(f'Missing or invalid settings structure: {e}, loading defaults')
                    return default_values()

            return default_values()
        except Exception as e:
            logger.error(f'Error loading settings: {e}')
            return default_values()


    def _assignSettings(self) -> None:
        """
        Assign loaded settings to global variables
        """

        global option_pool, proc_op_dict, main_set, save_set
        option_pool = self.loadSettings()
        proc_op_dict = option_pool['processing']
        main_set = option_pool['main']
        save_set = option_pool['save']


    def saveSettings(self) -> None:
        """
        Save all current settings
        """

        if self.limitADSB.value() < self.limitBDSB.value():
            a = self.limitADSB.value()
            b = self.limitBDSB.value()

            self.limitADSB.setValue(b)
            self.limitBDSB.setValue(a)

        main_set.update({
            'method': self.methodCB.currentText(),
            'direction': self.depthDirectionCB.currentText(),
            'depth_limit': {
                'disable': self.limitCheckBox.isChecked(),
                'upper': self.limitADSB.value(),
                'lower': self.limitBDSB.value()
            },
        })

        self.settings.setValue('options', option_pool)
        self.settings.setValue('last_directory', self.dir_path)


    def resetToDefault(self) -> None:
        """
        Reset all settings to default values
        """

        resetWindow = QMessageBox()
        resetWindow.setWindowTitle('Reset Settings')
        resetWindow.setText(
            'Are you sure you want to reset all settings to default values?'
        )
        resetWindow.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        resetWindow.setDefaultButton(QMessageBox.No)
        resetWindow.setWindowIcon(
            QIcon(resource_path('icons/warning-pngrepo-com.png'))
        )

        reply = resetWindow.exec_()

        if reply == QMessageBox.Yes:
            self.settings.clear()
            self._assignSettings()

            self.depthDirectionCB.setCurrentText(main_set['direction'])
            self.limitCheckBox.setChecked(main_set['depth_limit']['disable'])
            self.limitADSB.setValue(main_set['depth_limit']['upper'])
            self.limitBDSB.setValue(main_set['depth_limit']['lower'])
            self.methodCB.setCurrentText(main_set['method'])

            home_dir = Path.home()
            self.dir_path = home_dir
            self.settings.setValue('last_directory', home_dir)
            logger.info('last directory and options reset to default')
            logger.debug(
                f'reset to default options: \n{pprint.pformat(option_pool, width=200)}'
            )

            complete = QMessageBox()
            complete.setWindowTitle('Settings Reset')
            complete.setText('All settings have been reset to default values.')
            complete.setWindowIcon(
                QIcon(resource_path('icons/complete-pngrepo-com.png'))
            )
            complete.setWindowModality(Qt.ApplicationModal)
            complete.exec_()


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
        fname = command(
            self,
            window_text,
            str(self.dir_path),
            fileFilter,
            selectedFilter
        )

        if fname[0]:
            selected_path = Path(fname[0])

            # For save dialogs, ensure the extension matches the selected filter
            if 'Save' in window_text and file_type in DEM_FORMATS:
                extension = self._getDEMExtension(file_type)
                if selected_path.suffix.lower() != extension.lower():
                    selected_path = selected_path.with_suffix(extension)

            text_browser.setText(str(selected_path))
            self.dir_path = selected_path.parent
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

        grid = QGridLayout()
        row = 1
        openFilesButton = QPushButton('Open File')
        openFilesButton.clicked.connect(
            lambda: self.fileDialog(
                command=QFileDialog.getOpenFileName,
                window_text='Open Image File',
                file_type='GeoTIFF (*.tif)',
                text_browser=self.imglocList
            )
        )
        grid.addWidget(openFilesButton, row, 1, 1, 4)

        row += 1
        locLabel = QLabel('Location:')
        grid.addWidget(locLabel, row, 1, 1, 1)

        row += 1
        self.imglocList = QTextBrowser()
        grid.addWidget(self.imglocList, row, 1, 10, 4)

        row += 10
        loadButton = QPushButton('Load')
        loadButton.clicked.connect(self._loadImageAction)
        loadButton.clicked.connect(self.loadImageDialog.close)
        grid.addWidget(loadButton, 15, 3, 1, 1)

        cancelButton = QPushButton('Cancel')
        cancelButton.clicked.connect(self.loadImageDialog.close)
        grid.addWidget(cancelButton, 15, 4, 1, 1)

        self.loadImageDialog.setLayout(grid)

        self.loadImageDialog.exec_()


    def _loadImageAction(self):
        """
        Loading selected image and retrieve some metadata such as file size,
        band quantity, array size, pixel size, etc. Then, recreate image 3D
        array into a simple column and row array.
        """

        try:
            if not self.imglocList.toPlainText():
                logger.critical('no image filepath')
                raise ValueError('empty file path')

            logger.debug(
                f'loading image from: {Path(self.imglocList.toPlainText())}'
            )

            self.img_size = Path(self.imglocList.toPlainText()).stat().st_size

            global image_raw
            image_raw = sdb.read_geotiff(self.imglocList.toPlainText())

            global bands_df
            bands_df = sdb.unravel(image_raw)

            self.loadImageLabel.setText(Path(self.imglocList.toPlainText()).name)

            logger.info(f'load image successfully of size: {self.img_size} B')
            logger.info(f'image CRS: {image_raw.rio.crs}')
        except ValueError as e:
            if 'empty file path' in str(e):
                self.loadImageDialog.close()
                self._warningWithClear(
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

        grid = QGridLayout()
        row = 1
        openFilesButton = QPushButton('Open File')
        openFilesButton.clicked.connect(
            lambda: self.fileDialog(
                command=QFileDialog.getOpenFileName,
                window_text='Open Depth Sample File',
                file_type='ESRI Shapefile (*.shp)',
                text_browser=self.samplelocList
            )
        )
        grid.addWidget(openFilesButton, row, 1, 1, 4)

        row += 1
        locLabel = QLabel('Location:')
        grid.addWidget(locLabel, row, 1, 1, 1)

        row += 1
        self.samplelocList = QTextBrowser()
        grid.addWidget(self.samplelocList, row, 1, 10, 4)

        row += 10
        self.showCheckBox = QCheckBox('Show All Data to Table')
        self.showCheckBox.setChecked(False)
        grid.addWidget(self.showCheckBox, row, 1, 1, 2)

        loadButton = QPushButton('Load')
        loadButton.clicked.connect(self._loadSampleAction)
        loadButton.clicked.connect(self.loadSampleDialog.close)
        grid.addWidget(loadButton, row, 3, 1, 1)

        cancelButton = QPushButton('Cancel')
        cancelButton.clicked.connect(self.loadSampleDialog.close)
        grid.addWidget(cancelButton, row, 4, 1, 1)

        self.loadSampleDialog.setLayout(grid)

        self.loadSampleDialog.exec_()


    def _loadSampleAction(self):
        """
        Loading selected sample and retrieve file size.
        Then, some or all data on selected sample to the widget.
        """

        try:
            if not self.samplelocList.toPlainText():
                logger.critical('no sample filepath')
                raise ValueError('empty file path')

            logger.debug(
                f'loading sample data from: {
                    Path(self.samplelocList.toPlainText())
                }'
            )

            global sample_size
            sample_size = Path(self.samplelocList.toPlainText()).stat().st_size

            global sample_raw
            sample_raw = sdb.read_shapefile(self.samplelocList.toPlainText())

            proc_op_dict.update({
                'current_selection': SELECTION_TYPES['RANDOM']
            })
            proc_op_dict[
                'selection'
            ][
                SELECTION_TYPES['ATTRIBUTE']
            ][
                'parameters'
            ].update({
                'header': '',
                'group': ''
            })

            logger.debug('reset attribute selection parameters for new dataset')

            self.loadSampleLabel.setText(
                Path(self.samplelocList.toPlainText()).name
            )

            if (sample_raw.geom_type != 'Point').any():
                logger.critical('sample is not point type')
                del sample_raw
                self.loadSampleLabel.setText('Sample Retracted')
                self.depthHeaderCB.clear()
                self.table.clearContents()

                self.loadSampleDialog.close()
                self._warningWithoutClear(
                    'Your data is not Point type. Please load another data!'
                )
                self.loadSampleWindow()
            else:
                raw = sample_raw.copy()

                if self.showCheckBox.isChecked():
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
                self._warningWithClear(
                    'No data loaded. Please load your data!'
                )
                self.loadSampleWindow()


    def _methodOptionWindow(self):
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
            label = QLabel(to_title(param) + ':')
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
                set_name = f'{param}_set'
                if set_name in method_options:
                    widget.addItems(method_options[set_name])
                else:
                    widget.addItems([value])
                widget.setCurrentText(value)

            self.option_widgets[param] = widget
            grid.addWidget(widget, row, 3, 1, 2)
            row += 1

        loadButton = QPushButton('Load')
        loadButton.clicked.connect(self._loadMethodOptionAction)
        loadButton.clicked.connect(optionDialog.close)
        grid.addWidget(loadButton, row, 3, 1, 1)

        cancelButton = QPushButton('Cancel')
        cancelButton.clicked.connect(optionDialog.close)
        grid.addWidget(cancelButton, row, 4, 1, 1)

        optionDialog.setLayout(grid)
        optionDialog.exec_()


    def _loadMethodOptionAction(self) -> None:
        """
        Update settings for any method and save them
        """

        method = self.methodCB.currentText()

        for param, widget in self.option_widgets.items():
            if isinstance(widget, QComboBox):
                value = (str2bool(widget.currentText())
                        if widget.currentText() in ('True', 'False')
                        else widget.currentText())
            else:
                value = widget.value()
                
            option_pool['method'][method]['model_parameters'][param] = value
        logger.info(f'{method} parameters updated')

        self.saveSettings()


    def _processingOptionWindow(self):
        """
        Processing option User Interface
        """

        self.processingOptionDialog = QDialog()
        self.processingOptionDialog.setWindowTitle('Processing Options')
        self.processingOptionDialog.setWindowIcon(
            QIcon(resource_path('icons/setting-tool-pngrepo-com.png'))
        )

        grid = QGridLayout()
        row = 1
        backendLabel = QLabel('Parallel Backend:')
        grid.addWidget(backendLabel, row, 1, 1, 2)

        self.backendCB = QComboBox()
        self.backendCB.addItems(proc_op_dict['backend_set'])
        self.backendCB.setCurrentText(proc_op_dict['backend'])
        grid.addWidget(self.backendCB, row, 3, 1, 2)

        row += 1
        njobsLabel = QLabel('Processing Cores:')
        grid.addWidget(njobsLabel, row, 1, 1, 2)

        self.njobsSB = QSpinBox()
        self.njobsSB.setRange(-100, 100)
        self.njobsSB.setValue(proc_op_dict['n_jobs'])
        self.njobsSB.setAlignment(Qt.AlignRight)
        grid.addWidget(self.njobsSB, row, 3, 1, 2)

        row += 1
        evalTypeLabel = QLabel('Evaluation Type:')
        grid.addWidget(evalTypeLabel, row, 1, 1, 2)

        self.evalTypeCB = QComboBox()
        self.evalTypeCB.addItems(list(EVALUATION_TYPES.keys()))
        self.evalTypeCB.setCurrentText(proc_op_dict['current_eval'])
        grid.addWidget(self.evalTypeCB, row, 3, 1, 2)

        row += 1
        trainSelectLabel = QLabel('Train Data Selection:')
        grid.addWidget(trainSelectLabel, row, 1, 1, 2)

        self.trainSelectCB = QComboBox()
        selection_list = list(proc_op_dict['selection'].keys())
        self.trainSelectCB.addItems(selection_list)
        self.trainSelectCB.setCurrentText(proc_op_dict['current_selection'])
        self.trainSelectCB.currentTextChanged.connect(self._updateTrainSelection)
        grid.addWidget(self.trainSelectCB, row, 3, 1, 2)

        row += 1
        self.dynamicContainer = QWidget()
        self.dynamicLayout = QGridLayout()
        self.dynamicContainer.setLayout(self.dynamicLayout)
        self._updateTrainSelection(self.trainSelectCB.currentText())
        grid.addWidget(self.dynamicContainer, row, 1, 1, 4)

        row += 1
        loadButton = QPushButton('Load')
        loadButton.clicked.connect(self._loadProcessingOptionAction)
        loadButton.clicked.connect(self.processingOptionDialog.close)
        grid.addWidget(loadButton, row, 3, 1, 1)

        cancelButton = QPushButton('Cancel')
        cancelButton.clicked.connect(self.processingOptionDialog.close)
        grid.addWidget(cancelButton, row, 4, 1, 1)

        self.processingOptionDialog.setLayout(grid)

        self.processingOptionDialog.exec_()


    def _loadProcessingOptionAction(self):
        """
        Loading defined processing option input
        """

        if self.njobsSB.value() == 0:
            self.processingOptionDialog.close()
            self._warningWithoutClear(
                'Do not insert zero on Processing Cores!'
            )
            self._processingOptionWindow()
            return

        proc_op_dict['backend'] = self.backendCB.currentText()
        proc_op_dict['n_jobs'] = self.njobsSB.value()
        proc_op_dict['current_eval'] = self.evalTypeCB.currentText()
        proc_op_dict['current_selection'] = self.trainSelectCB.currentText()

        selection_params = proc_op_dict['selection'][
            self.trainSelectCB.currentText()
        ]['parameters']
        
        for param, widget in self.selection_widgets.items():
            if isinstance(widget, QDoubleSpinBox):
                selection_params[param] = widget.value()
            elif isinstance(widget, QSpinBox):
                selection_params[param] = widget.value()
            elif isinstance(widget, QComboBox):
                selection_params[param] = widget.currentText()

        logger.info('processing options updated')
        self.saveSettings()


    def _updateTrainSelection(self, selection: str) -> None:
        """
        Update dynamic UI based on train selection type
        """

        try:
            for i in reversed(range(self.dynamicLayout.count())):
                widget = self.dynamicLayout.itemAt(i).widget()
                if widget:
                    widget.setParent(None)

            self.currentSelection = proc_op_dict['selection'][selection]
            self.selection_widgets = {}
            
            row = 0
            for param, value in self.currentSelection['parameters'].items():
                label = QLabel(to_title(param) + ':')
                self.dynamicLayout.addWidget(label, row, 1, 1, 2)

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
                        widget.activated.connect(self._updateGroupSelection)
                        if value:
                            widget.setCurrentText(value)
                            header = self.currentSelection['parameters']['header']
                        else:
                            widget.setCurrentText(widget.itemText(0))
                            header = widget.currentText()
                    elif param == 'group':
                        groups = list(sample_raw.groupby(header).groups.keys())
                        widget.addItems(groups)
                        if value:
                            widget.setCurrentText(value)
                        elif groups:
                            widget.setCurrentText(groups[0])

                self.selection_widgets[param] = widget
                self.dynamicLayout.addWidget(widget, row, 3, 1, 2)
                row += 1
        except NameError:
            self.trainSelectCB.setCurrentText(SELECTION_TYPES['RANDOM'])
            self._warningWithClear(
                'No depth sample loaded. Please load your depth sample!'
            )


    def _updateGroupSelection(self):
        """
        Update list in group selection when header selection changes
        """

        try:
            selected_header = self.selection_widgets['header'].currentText()

            if selected_header:
                object_only = sample_raw.select_dtypes(include=['object'])
                group_list = list(
                    object_only.groupby(selected_header).groups.keys()
                )

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


    def _predict(self):
        """
        Sending parameters and inputs from widget to Process Class
        """

        logging.debug('Sending user inputs to process class')
        self.saveSettings()

        self.resultText.clear()
        self.progressBar.setValue(0)

        global time_list
        time_list = []
        init_input = {
            'depth_label': self.depthHeaderCB.currentText(),
            'depth_direction': self.depthDirectionCB.currentText(), 
            'limit_state': self.limitCheckBox.isChecked(),
            'limit_a': self.limitADSB.value(),
            'limit_b': self.limitBDSB.value(),
            'method': self.methodCB.currentText(),
            'train_select': proc_op_dict['current_selection'],
            'selection': proc_op_dict['selection'][
                proc_op_dict['current_selection']
            ]['parameters'],
            'eval_type': proc_op_dict['current_eval'],
        }

        try:
            if sample_raw[self.depthHeaderCB.currentText()].dtype == 'float':
                self.sdbProcess = Process()
                self.widget_signal.connect(self.sdbProcess.inputs)
                self.widget_signal.emit(init_input)
                self.sdbProcess.start()
                self.sdbProcess.time_signal.connect(self._timeCounting)
                self.sdbProcess.thread_signal.connect(self._results)
                self.sdbProcess.warning_with_clear.connect(self._warningWithClear)
            else:
                self._warningWithClear(
                    'Please select headers correctly!'
                )
        except NameError:
            self._warningWithClear(
                'No depth sample loaded. Please load your depth sample!'
            )


    def _timeCounting(self, time_text: List[Union[datetime.datetime, str]]) -> None:
        """
        Receive time value on every step and its corresponding processing
        text to show in result text browser and increase progress bar.
        """

        time_list.append(time_text[0])
        self.resultText.append(time_text[1])
        self.progressBar.setValue(self.progressBar.value() + 1)

        if self.progressBar.value() == self.progressBar.maximum():
            self._completeDialog()


    def _results(self, result_dict: Dict[str, Any]) -> None:
        """
        Recieve processing results and filter the predicted value to depth
        limit window (if enabled).
        Counting runtimes using saved time values and printing result info.
        """

        if EVALUATION_TYPES[proc_op_dict['current_eval']]:
            print_eval_type = (
                'Evaluated using predicted values that was generated from '
                'recalculation of depth prediction using the existing model'
                ' and test data'
            )
        elif not EVALUATION_TYPES[proc_op_dict['current_eval']]:
            print_eval_type = (
                'Evaluated using predicted values that was generated from '
                'point samples of the existing predicted values'
            )

        print_selection_info = (
            f'Parallel Backend:\t{proc_op_dict["backend"]}\n'
            f'Processing Cores:\t{proc_op_dict["n_jobs"]}\n'
            f'Train Data Selection:\t{proc_op_dict["current_selection"]}\n'
        )
        parameters = proc_op_dict['selection'][proc_op_dict['current_selection']]
        for param, value in parameters['parameters'].items():
            print_selection_info += (
                f'{to_title(param)}:\t\t{value}\n'
            )

        global end_results
        end_results = result_dict

        daz_predict = end_results['daz_predict']
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

        if not self.limitCheckBox.isChecked():
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
            f'Image Input:\t\t{Path(self.imglocList.toPlainText())} '
            f'({round(self.img_size / 2**20, 2)} MiB)\n'
            f'Sample Data:\t\t{Path(self.samplelocList.toPlainText())} '
            f'({round(sample_size / 2**20, 2)} MiB)\n'
            f'Selected Header:\t{self.depthHeaderCB.currentText()}\n'
            f'Depth Direction:\t\t{self.depthDirectionCB.currentText()}\n'
            'Min/Max:\t\t'
            f'{sample_raw[self.depthHeaderCB.currentText()].min():.2f}/'
            f'{sample_raw[self.depthHeaderCB.currentText()].max():.2f}\n\n'
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
            f'{print_eval_type}\n'
            f'RMSE:\t\t{round(rmse, 3)}\n'
            f'MAE:\t\t{round(mae, 3)}\n'
            f'R\u00B2:\t\t{round(r2, 3)}\n\n'
            f'{print_selection_info}\n'
            f'Clipping Runtime:\t{runtime[0]}\n'
            f'Filtering Runtime:\t{runtime[1]}\n'
            f'Splitting Runtime:\t{runtime[2]}\n'
            f'Modeling Runtime:\t{runtime[3]}\n'
            f'Evaluation Runtime:\t{runtime[4]}\n'
            f'Overall Runtime:\t{runtime[5]}\n\n'
            f'CRS:\t\t{daz_predict.rio.crs}\n'
            f'Dimensions:\t\t{daz_predict.rio.width} x '
            f'{daz_predict.rio.height} pixels\n'
            f'Pixel Size:\t\t{abs(daz_predict.rio.resolution()[0])} , '
            f'{abs(daz_predict.rio.resolution()[1])}\n'
            'Min/Max:\t\t'
            f'{daz_predict.values[0].min():.2f}/'
            f'{daz_predict.values[0].max():.2f}\n\n'
        )

        self.resultText.setText(print_result_info)


    def _stopProcess(self):
        """
        Stop processing and clear result info and progress bar
        """

        if hasattr(self, 'sdbProcess') and self.sdbProcess.isRunning():
            self.sdbProcess.stop()
            self.sdbProcess.wait()
            self.resultText.clear()
            self.progressBar.setValue(0)
            self.resultText.setText('Processing has been stopped!')


    def _copyLogFile(self):
        """
        Copy the original log file to the save location
        """

        if hasattr(self, 'savelocList') and self.savelocList.toPlainText():
            try:
                save_path = Path(
                    self.savelocList.toPlainText()
                ).with_suffix('.log')

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

        self._copyLogFile()
        event.accept()


    def _warningWithClear(self, warning_text):
        """
        Show warning dialog and customized warning text
        and then clear result info and progress bar after closing
        """

        warning = QErrorMessage()
        warning.setWindowTitle('WARNING')
        warning.setWindowIcon(
            QIcon(resource_path('icons/warning-pngrepo-com.png'))
        )
        warning.setWindowModality(Qt.ApplicationModal)
        warning.showMessage(warning_text)

        warning.exec_()
        self.resultText.clear()
        self.progressBar.setValue(0)


    def _warningWithoutClear(self, warning_text):
        """
        Show warning dialog and customized warning text
        without clearing result info and progress bar after closing
        """

        warning = QErrorMessage()
        warning.setWindowTitle('WARNING')
        warning.setWindowIcon(
            QIcon(resource_path('icons/warning-pngrepo-com.png'))
        )
        warning.setWindowModality(Qt.ApplicationModal)
        warning.showMessage(warning_text)

        warning.exec_()


    def _completeDialog(self):
        """
        Showing complete pop up dialog
        """

        complete = QDialog()
        complete.setWindowTitle('Complete')
        complete.setWindowIcon(
            QIcon(resource_path('icons/complete-pngrepo-com.png'))
        )
        complete.setWindowModality(Qt.ApplicationModal)
        complete.resize(180,30)

        grid = QGridLayout()
        textLabel = QLabel('Tasks has been completed')
        textLabel.setAlignment(Qt.AlignCenter)
        grid.addWidget(textLabel, 1, 1, 1, 4)

        okButton = QPushButton('OK')
        okButton.clicked.connect(complete.close)
        grid.addWidget(okButton, 2, 2, 1, 2)

        complete.setLayout(grid)

        complete.exec_()


    def _saveOptionWindow(self):
        """
        Saving option window
        """

        self.saveOptionDialog = QDialog()
        self.saveOptionDialog.setWindowTitle('Save Options')
        self.saveOptionDialog.setWindowIcon(
            QIcon(resource_path('icons/load-pngrepo-com.png'))
        )

        def dialogCloseEvent(event):
            """Save settings before closing dialog"""

            save_set.update({
                'type': self.dataTypeCB.currentText(),
                'direction': self.depthDirectionSaveCB.currentText(),
                'depth_limit': {
                    'upper': self.saveLimitADSB.value(),
                    'lower': self.saveLimitBDSB.value(),
                },
                'filter': {
                    'disable': self.medianFilterCheckBox.isChecked(),
                    'size': self.medianFilterSB.value(),
                },
                'scatter_plot': self.scatterPlotCheckBox.isChecked(),
                'train_test': {
                    'save': self.trainTestDataCheckBox.isChecked(),
                    'format': self.trainTestFormatCB.currentText(),
                },
                'dem': self.saveDEMCheckBox.isChecked(),
                'report': self.reportCheckBox.isChecked(),
            })

            self.settings.setValue('options', option_pool)
            logger.debug('save options updated')
            event.accept()

        self.saveOptionDialog.closeEvent = dialogCloseEvent

        grid = QGridLayout()
        row = 1
        dataTypeLabel = QLabel('Data Type:')
        grid.addWidget(dataTypeLabel, row, 1, 1, 1)

        self.dataTypeCB = QComboBox()
        self.dataTypeCB.addItems(DEM_FORMATS)
        self.dataTypeCB.setCurrentText(save_set['type'])
        grid.addWidget(self.dataTypeCB, row, 2, 1, 3)

        row += 1
        depthDirectionSaveLabel = QLabel('Depth Direction:')
        grid.addWidget(depthDirectionSaveLabel, row, 1, 1, 1)

        self.depthDirectionSaveCB = QComboBox()
        direction_list = list(DEPTH_DIRECTION.keys())
        self.depthDirectionSaveCB.addItems(direction_list)
        self.depthDirectionSaveCB.setCurrentText(save_set['direction'])
        grid.addWidget(self.depthDirectionSaveCB, row, 2, 1, 3)

        row += 1
        limitALabel = QLabel('Upper Limit:')
        grid.addWidget(limitALabel, row, 1, 1, 1)

        self.saveLimitADSB = QDoubleSpinBox()
        self.saveLimitADSB.setRange(-100, 100)
        self.saveLimitADSB.setDecimals(1)
        self.saveLimitADSB.setValue(save_set['depth_limit']['upper'])
        self.saveLimitADSB.setSuffix(' m')
        self.saveLimitADSB.setAlignment(Qt.AlignRight)
        grid.addWidget(self.saveLimitADSB, row, 2, 1, 1)

        limitBLabel = QLabel('Lower Limit:')
        grid.addWidget(limitBLabel, row, 3, 1, 1)

        self.saveLimitBDSB = QDoubleSpinBox()
        self.saveLimitBDSB.setRange(-100, 100)
        self.saveLimitBDSB.setDecimals(1)
        self.saveLimitBDSB.setValue(save_set['depth_limit']['lower'])
        self.saveLimitBDSB.setSuffix(' m')
        self.saveLimitBDSB.setAlignment(Qt.AlignRight)
        grid.addWidget(self.saveLimitBDSB, row, 4, 1, 1)

        row += 1
        medianFilterLabel = QLabel('Median Filter Size:')
        grid.addWidget(medianFilterLabel, row, 1, 1, 1)

        self.medianFilterSB = QSpinBox()
        self.medianFilterSB.setRange(3, 33)
        self.medianFilterSB.setValue(save_set['filter']['size'])
        self.medianFilterSB.setSingleStep(2)
        self.medianFilterSB.setAlignment(Qt.AlignRight)
        grid.addWidget(self.medianFilterSB, row, 2, 1, 1)

        self.medianFilterCheckBox = QCheckBox('Disable Median Filter')
        self.medianFilterCheckBox.setChecked(save_set['filter']['disable'])
        grid.addWidget(self.medianFilterCheckBox, row, 3, 1, 2)

        row += 1
        saveFileButton = QPushButton('Save File Location')
        saveFileButton.clicked.connect(
            lambda:self.fileDialog(
                command=QFileDialog.getSaveFileName,
                window_text='Save File',
                file_type=self.dataTypeCB.currentText(),
                text_browser=self.savelocList
            )
        )
        grid.addWidget(saveFileButton, row, 1, 1, 4)

        row += 1
        locLabel = QLabel('Location:')
        grid.addWidget(locLabel, row, 1, 1, 4)

        row += 1
        self.savelocList = QTextBrowser()
        grid.addWidget(self.savelocList, row, 1, 1, 4)

        row += 1
        self.scatterPlotCheckBox = QCheckBox('Save Scatter Plot')
        self.scatterPlotCheckBox.setChecked(save_set['scatter_plot'])
        grid.addWidget(self.scatterPlotCheckBox, row, 1, 1, 2)

        row += 1
        self.trainTestDataCheckBox = QCheckBox('Save Training and Testing Data in')
        self.trainTestDataCheckBox.setChecked(save_set['train_test']['save'])
        grid.addWidget(self.trainTestDataCheckBox, row, 1, 1, 2)

        self.trainTestFormatCB = QComboBox()
        self.trainTestFormatCB.addItems(TRAIN_TEST_SAVE.keys())
        self.trainTestFormatCB.setCurrentText(save_set['train_test']['format'])
        grid.addWidget(self.trainTestFormatCB, row, 3, 1, 1)

        trainTestLabel = QLabel('format')
        grid.addWidget(trainTestLabel, row, 4, 1, 1)

        row += 1
        self.saveDEMCheckBox = QCheckBox('Save DEM')
        self.saveDEMCheckBox.setChecked(save_set['dem'])
        grid.addWidget(self.saveDEMCheckBox, row, 1, 1, 1)

        self.reportCheckBox = QCheckBox('Save Report')
        self.reportCheckBox.setChecked(save_set['report'])
        grid.addWidget(self.reportCheckBox, row, 2, 1, 1)

        saveButton = QPushButton('Save')
        saveButton.clicked.connect(self._saveAction)
        saveButton.clicked.connect(self.saveOptionDialog.close)
        grid.addWidget(saveButton, row, 3, 1, 1)

        cancelButton = QPushButton('Cancel')
        cancelButton.clicked.connect(self.saveOptionDialog.close)
        grid.addWidget(cancelButton, row, 4, 1, 1)

        self.saveOptionDialog.setLayout(grid)

        self.saveOptionDialog.exec_()


    def _saveAction(self):
        """
        Saving predicted depth, training and testing data, and/or report into file.
        Applying median filter (or not) to the predicted depth array before saving.
        """

        try:
            daz_filtered = end_results['daz_predict'].copy()

            if not self.medianFilterCheckBox.isChecked():
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

            daz_filtered.values[0] = sdb.out_depth_filter(
                array=daz_filtered.values[0],
                top_limit=self.saveLimitADSB.value(),
                bottom_limit=self.saveLimitBDSB.value()
            )

            train_df_copy = end_results['train'].copy()
            test_df_copy = end_results['test'].copy()

            if DEPTH_DIRECTION[self.depthDirectionSaveCB.currentText()][1]:
                daz_filtered.values[0] *=-1
                test_df_copy['z'] *=-1
                test_df_copy['z_validate'] *=-1
                train_df_copy['z'] *=-1

            if not self.savelocList.toPlainText():
                raise ValueError('empty save location')

            save_loc = Path(self.savelocList.toPlainText())

            if self.saveDEMCheckBox.isChecked():
                sdb.write_geotiff(
                    daz_filtered,
                    save_loc
                )
                new_img_size = Path(save_loc).stat().st_size
                print_dem_info = (
                    f'{print_filter_info}\n\n'
                    f'DEM Output:\t\t{save_loc} '
                    f'({round(new_img_size / 2**10 / 2**10, 2)} MiB)\n'
                )
                logger.info(
                    f'DEM with the size of {new_img_size} B has been saved'
                )
                logger.debug(f'DEM location: {save_loc}')
            elif not self.saveDEMCheckBox.isChecked():
                print_dem_info = (
                    'DEM Output:\t\tNot Saved\n'
                )

            if self.trainTestDataCheckBox.isChecked():
                print_train_test_info = self._trainTestSave(
                    train_data=train_df_copy,
                    test_data=test_df_copy,
                    save_location=save_loc,
                    data_format=self.trainTestFormatCB.currentText(),
                    split=TRAIN_TEST_SAVE[self.trainTestFormatCB.currentText()],
                )

                logger.info(
                    f'splitted train and test data with {
                        self.trainTestFormatCB.currentText()
                    } format has been saved'
                )
            elif not self.trainTestDataCheckBox.isChecked():
                print_train_test_info = (
                    'Train dna Test Data Output:\tNot Saved\n'
                )

            if self.scatterPlotCheckBox.isChecked():
                scatter_plot_loc = Path(save_loc).with_name(
                    f'{Path(save_loc).stem}_scatter_plot.png'
                )
                scatter_plot = sdb.scatter_plotter(
                    true_val=test_df_copy['z'],
                    pred_val=test_df_copy['z_validate'],
                    title=self.methodCB.currentText()
                )
                scatter_plot[0].savefig(scatter_plot_loc)

                scatter_plot_size = Path(scatter_plot_loc).stat().st_size

                print_scatter_plot_info = (
                    f'Scatter Plot:\t{scatter_plot_loc} '
                    f'({round(scatter_plot_size / 2**10, 2)} KiB)\n'
                )
                logger.info('scatter plot has been saved')
                logger.debug(f'scatter plot location: {scatter_plot_loc}')
            elif not self.scatterPlotCheckBox.isChecked():
                print_scatter_plot_info = 'Scatter Plot:\t\tNotSaved\n'

            self.resultText.append(print_dem_info)
            self.resultText.append(print_train_test_info)
            self.resultText.append(print_scatter_plot_info)

            if self.reportCheckBox.isChecked():
                report_save_loc = Path(save_loc).with_name(
                    f'{Path(save_loc).stem}_report.txt'
                )
                report = open(report_save_loc, 'w')

                report.write(
                    print_result_info +
                    print_dem_info +
                    print_train_test_info +
                    print_scatter_plot_info
                )
                logger.info('report has been saved')
                logger.debug(f'report location: {report_save_loc}')
        except ValueError as e:
            if 'Allowed value: >= 3 or odd numbers' in str(e):
                self.saveOptionDialog.close()
                self._warningWithoutClear(
                    'Please insert odd number on filter size!'
                )
                self._saveOptionWindow()
            elif 'empty save location' in str(e):
                self.saveOptionDialog.close()
                self._warningWithoutClear(
                    'Please insert save location!'
                )
                self._saveOptionWindow()


    def _trainTestSave(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        save_location: Path | str,
        data_format: str,
        split: bool,
    ) -> str:
        """
        Saving splitted train and test data into file
        and returning the print info to be shown in result text browser.
        """

        if split:
            train_save_loc = Path(save_location).with_name(
                f'{Path(save_location).stem}_train{data_format}'
            )
            test_save_loc = Path(save_location).with_name(
                f'{Path(save_location).stem}_test{data_format}'
            )
            if data_format == '.csv':
                train_data.to_csv(train_save_loc, index=False)
                test_data.to_csv(test_save_loc, index=False)
            elif data_format == '.shp':
                sdb.write_shapefile(
                    train_data,
                    train_save_loc,
                    x_col_name='x',
                    y_col_name='y',
                    crs=end_results['sample_gdf'].crs
                )
                sdb.write_shapefile(
                    test_data,
                    test_save_loc,
                    x_col_name='x',
                    y_col_name='y',
                    crs=end_results['sample_gdf'].crs
                )

            train_data_size = Path(train_save_loc).stat().st_size
            test_data_size = Path(test_save_loc).stat().st_size
            print_info = (
                f'Train Data Output:\t{train_save_loc} '
                f'{round(train_data_size / 2**10 / 2**10, 2)} MiB\n'
                f'Test Data output:\t{test_save_loc} '
                f'{round(test_data_size / 2**10 / 2**10, 2)} MiB\n'
            )
            logger.debug(f'train data location: {train_save_loc}')
            logger.debug(f'test data location: {test_save_loc}')
        else:
            merge_data_loc = Path(save_location).with_name(
                f'{Path(save_location).stem}_splitted_data{data_format}'
            )
            if data_format =='.gpkg':
                sdb.write_shapefile(
                    train_data,
                    merge_data_loc,
                    x_col_name='x',
                    y_col_name='y',
                    crs=end_results['sample_gdf'].crs,
                    layer='train_data'
                )
                sdb.write_shapefile(
                    test_data,
                    merge_data_loc,
                    x_col_name='x',
                    y_col_name='y',
                    crs=end_results['sample_gdf'].crs,
                    layer='test_data'
                )

            train_test_data_size = Path(merge_data_loc).stat().st_size
            print_info = (
                f'Train and Test Data Output:\t{merge_data_loc} '
                f'{round(train_test_data_size / 2**10 / 2**10, 2)} MiB\n'
            )
            logger.debug(f'train & test data location: {merge_data_loc}')

        return print_info


    def _getDEMExtension(self, text: str) -> str:
        """
        Get file extension from filetype text
        """
        match = re.search(r'\(\*(\.\w+)\)', text)
        if match:
            return match.group(1)
        raise ValueError(f'Could not extract extension from: {text}')


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

        grid = QGridLayout()
        licenseCB = QComboBox()
        licenseCB.addItems(list(license_dict))
        licenseCB.activated.connect(
            lambda: self._licenseSelection(
                location=license_dict[licenseCB.currentText()]
            )
        )
        grid.addWidget(licenseCB, 1, 1, 1, 4)

        self.licenseText = QTextBrowser()
        license_file = open(resource_path('LICENSE'), 'r')
        self.licenseText.setText(license_file.read())
        grid.addWidget(self.licenseText, 2, 1, 1, 4)

        okButton = QPushButton('OK')
        okButton.clicked.connect(licenses.close)
        grid.addWidget(okButton, 3, 4, 1, 1)

        licenses.setLayout(grid)

        licenses.exec_()


    def _licenseSelection(self, location):
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
        self.eval_type = input_dict['eval_type']


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
            depth_direction=DEPTH_DIRECTION[self.depth_direction][0],
            disable_depth_filter=self.limit_state,
            upper_limit=self.limit_a_value,
            lower_limit=self.limit_b_value
        )

        if not self._is_running:
            return None

        time_depth_filter = datetime.datetime.now()
        depth_filter_list = [time_depth_filter, 'Split Train and Test...\n']
        self.time_signal.emit(depth_filter_list)
        logger.info(f'split depth sample by {self.train_select}: {self.selection}')
        if self.train_select == SELECTION_TYPES['RANDOM']:
            f_train, f_test, z_train, z_test = sdb.split_random(
                raster=image_raw,
                vector=depth_filtered_sample,
                header=self.depth_label,
                train_size=self.selection['train_size'],
                random_state=self.selection['random_state']
            )
        elif self.train_select == SELECTION_TYPES['ATTRIBUTE']:
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
                f'{to_title(key)}:\t\t{value}\n'
            )

        if EVALUATION_TYPES[self.eval_type]:
            logger.debug('recalculate prediction using test data')
            f_test = results['f_test'].drop(columns=['x', 'y'])
        else:
            logger.debug('using prediction data to later use against z_test')
            f_test = None

        z_predict, z_validate = sdb.prediction(
            model=method,
            unraveled_band=bands_df,
            features_train=results['f_train'].drop(columns=['x', 'y']),
            label_train=results['z_train'],
            features_test=f_test,
            backend=proc_op_dict['backend'],
            n_jobs=proc_op_dict['n_jobs'],
            **model_parameters
        )

        if not self._is_running:
            return None

        results.update({
            'z_predict': z_predict,
            'z_validate': z_validate
        })

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

            if not EVALUATION_TYPES[self.eval_type]:
                logger.debug('sampling prediction based on test data coordinates')
                dfz_predict = sdb.point_sampling(
                    daz_predict,
                    x=results['f_test'].x,
                    y=results['f_test'].y,
                    include_xy=False
                )
                results['z_validate'] = dfz_predict['band_1'].to_numpy()

            logger.info('evaluating prediction')
            rmse, mae, r2 = sdb.evaluate(
                true_val=results['z_test'],
                pred_val=results['z_validate']
            )
            logger.info(f'RMSE: {rmse}, MAE: {mae}, R2: {r2}')

            time_test = datetime.datetime.now()
            test_list = [time_test, 'Done.']
            self.time_signal.emit(test_list)

            train_df = results['f_train'].copy()
            train_df['z'] = results['z_train'].copy()

            test_df = results['f_test'].copy()
            test_df['z'] = results['z_test'].copy()
            test_df['z_validate'] = results['z_validate'].copy()

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
        'name': SELECTION_TYPES['RANDOM'],
        'parameters': OrderedDict([
            ('train_size', 0.75),
            ('random_state', 0)
        ])
    }

    attribute_selection = {
        'name': SELECTION_TYPES['ATTRIBUTE'],
        'parameters': OrderedDict([
            ('header', ''),
            ('group', '')
        ])
    }

    proc_op_dict = {
        'saved_depth': {
            'upper': 2.0,
            'lower': -15.0
        },
        'backend': 'threading',
        'n_jobs': -2,
        'current_eval': 'Use Current Prediction',
        'selection' : OrderedDict([
            (random_selection['name'], random_selection),
            (attribute_selection['name'], attribute_selection)
        ]),
        'current_selection': random_selection['name'],
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

    main_dict = {
        'method': knn_op_dict['name'],
        'direction': list(DEPTH_DIRECTION.keys())[0],
        'depth_limit': {
            'disable': False,
            'upper': 2.0,
            'lower': -15.0
        },
    }

    save_dict = {
        'type': 'GeoTIFF (*.tif)',
        'direction': list(DEPTH_DIRECTION.keys())[0],
        'depth_limit': {
            'upper': 2.0,
            'lower': -15.0
        },
        'filter': {
            'disable': False,
            'size': 3,
        },
        'scatter_plot': False,
        'train_test': {
            'save': False,
            'format': list(TRAIN_TEST_SAVE.keys())[0],
        },
        'dem': True,
        'report': True,
    }

    default_dict = {
        'main': main_dict,
        'processing': proc_op_dict,
        'method': {
            knn_op_dict['name']: knn_op_dict,
            mlr_op_dict['name']: mlr_op_dict,
            rf_op_dict['name']: rf_op_dict
        },
        'save': save_dict
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
        # Use the script's directory, not the current working directory
        base_path = Path(__file__).parent.resolve()
    return str(Path(base_path) / relative_path)


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


if __name__ == '__main__':
    app = QApplication(sys.argv)
    logger.info(f'SDB GUI {SDB_GUI_VERSION} started')
    main()
    exit_code = app.exec_()
    logger.info(f'SDB GUI {SDB_GUI_VERSION} exited with code {exit_code}')
    sys.exit(exit_code)