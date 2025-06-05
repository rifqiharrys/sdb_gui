"""Main widget for SDB GUI application"""

import logging
import os
from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QPushButton, QLabel, QComboBox, QSpinBox,
    QDoubleSpinBox, QCheckBox, QProgressBar, QTextBrowser,
    QTableWidget, QGridLayout, QVBoxLayout, QScrollArea
)
from PyQt5.QtCore import Qt, QSettings, pyqtSignal
from PyQt5.QtGui import QIcon

from ..config.constants import DEPTH_DIRECTION, SDB_GUI_VERSION
from ..core.settings import load_settings
from ..core.utils import resource_path
from ..core.process import Process
from .dialogs import LoadImageDialog, LoadSampleDialog, MethodDialog
from .dialogs import ProcessDialog, SaveDialog

logger = logging.getLogger(__name__)

class SDBWidget(QWidget):
    """Main widget class for SDB GUI"""
    widget_signal = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.settings = QSettings('SDB', 'SDB GUI')
        self.dir_path = self.settings.value(
            'last_directory',
            os.path.abspath(Path.home())
        )
        self.initUI()

    def initUI(self):
        """Initialize User Interface"""
        self.setGeometry(300, 100, 480, 640)
        self.setWindowTitle(f'Satellite Derived Bathymetry v{SDB_GUI_VERSION}')
        self.setWindowIcon(QIcon(resource_path('icons/satellite.png')))

        # Main layout setup
        mainLayout = QVBoxLayout()
        self.setLayout(mainLayout)

        # Add UI components
        self.setupDataSection(mainLayout)
        self.setupMethodSection(mainLayout)
        self.setupProcessSection(mainLayout)

    def setupDataSection(self, mainLayout):
        """Setup data loading section"""
        grid = QGridLayout()
        row = 1

        # Image loading
        loadImageButton = QPushButton('Load Image')
        loadImageButton.clicked.connect(LoadImageDialog)
        grid.addWidget(loadImageButton, row, 1, 1, 1)

        self.loadImageLabel = QLabel('No Image Loaded')
        self.loadImageLabel.setAlignment(Qt.AlignCenter)
        grid.addWidget(self.loadImageLabel, row, 2, 1, 3)

        # Sample loading
        row += 1
        loadSampleButton = QPushButton('Load Sample')
        loadSampleButton.clicked.connect(LoadSampleDialog)
        grid.addWidget(loadSampleButton, row, 1, 1, 1)

        self.loadSampleLabel = QLabel('No Sample Loaded')
        self.loadSampleLabel.setAlignment(Qt.AlignCenter)
        grid.addWidget(self.loadSampleLabel, row, 2, 1, 3)

        # Depth header
        row += 1
        depthHeaderLabel = QLabel('Depth Header:')
        grid.addWidget(depthHeaderLabel, row, 1, 1, 1)

        self.depthHeaderCB = QComboBox()
        grid.addWidget(self.depthHeaderCB, row, 2, 1, 1)

        # Depth direction
        depthDirectionLabel = QLabel('Depth Direction:')
        grid.addWidget(depthDirectionLabel, row, 3, 1, 1)

        self.depthDirectionCB = QComboBox()
        self.depthDirectionCB.addItems(list(DEPTH_DIRECTION.keys()))
        grid.addWidget(self.depthDirectionCB, row, 4, 1, 1)

        # Data table
        row += 1
        self.table = QTableWidget()
        scroll = QScrollArea()
        scroll.setWidget(self.table)
        grid.addWidget(self.table, row, 1, 5, 4)

        # Depth limit
        row += 5
        limitLabel = QLabel('Depth Limit Window:')
        grid.addWidget(limitLabel, row, 1, 1, 2)

        limitALabel = QLabel('Upper Limit:')
        grid.addWidget(limitALabel, row, 3, 1, 1)

        self.limitADSB = QDoubleSpinBox()
        self.limitADSB.setRange(-100, 100)
        self.limitADSB.setSingleStep(0.1)
        self.limitADSB.setAlignment(Qt.AlignRight)
        grid.addWidget(self.limitADSB, row, 4, 1, 1)

        row += 1
        self.limitCheckBox = QCheckBox('Disable Depth Limitation')
        # self.limitCheckBox.stateChanged.connect(self.depthLimitState)
        grid.addWidget(self.limitCheckBox, row, 1, 1, 2)

        limitBLabel = QLabel('Lower Limit:')
        grid.addWidget(limitBLabel, row, 3, 1, 1)

        self.limitBDSB = QDoubleSpinBox()
        self.limitBDSB.setRange(-100, 100)
        self.limitBDSB.setSingleStep(0.1)
        self.limitBDSB.setAlignment(Qt.AlignRight)
        grid.addWidget(self.limitBDSB, row, 4, 1, 1)

        mainLayout.addLayout(grid)

    def setupMethodSection(self, mainLayout):
        """Setup method selection section"""
        grid = QGridLayout()
        row = 1

        # Method selection
        methodLabel = QLabel('Method:')
        grid.addWidget(methodLabel, row, 1, 1, 1)

        self.methodCB = QComboBox()
        self.methodCB.addItems(['K-Nearest Neighbors', 'Multiple Linear Regression', 'Random Forest'])
        grid.addWidget(self.methodCB, row, 2, 1, 2)

        methodOptionButton = QPushButton('Method Options')
        # methodOptionButton.clicked.connect(self.methodOptionWindow)
        grid.addWidget(methodOptionButton, row, 4, 1, 1)

        # Processing Options
        row += 1
        processingOptionButton = QPushButton('Processing Options')
        # processingOptionButton.clicked.connect(self.processingOptionWindow)
        grid.addWidget(processingOptionButton, row, 1, 1, 2)

        resetButton = QPushButton('Reset Settings')
        # resetButton.clicked.connect(self.resetSettings)
        grid.addWidget(resetButton, row, 3, 1, 2)

        mainLayout.addLayout(grid)

    def setupProcessSection(self, mainLayout):
        """Setup control buttons section"""
        grid = QGridLayout()
        row = 1

        # Progress bar and status
        self.progressBar = QProgressBar()
        grid.addWidget(self.progressBar, row, 1, 1, 4)

        row += 1
        self.resultText = QTextBrowser()
        self.resultText.setMinimumHeight(100)
        grid.addWidget(self.resultText, row, 1, 3, 4)

        # Control buttons
        row += 3
        generateButton = QPushButton('Generate')
        # generateButton.clicked.connect(self.predict)
        grid.addWidget(generateButton, row, 1, 1, 1)

        stopButton = QPushButton('Stop')
        # stopButton.clicked.connect(self.stopProcess)
        grid.addWidget(stopButton, row, 2, 1, 1)

        resetResultButton = QPushButton('Reset Result')
        # resetResultButton.clicked.connect(self.resetResults)
        grid.addWidget(resetResultButton, row, 3, 1, 1)

        saveButton = QPushButton('Save')
        # saveButton.clicked.connect(self.saveWindow)
        grid.addWidget(saveButton, row, 4, 1, 1)

        mainLayout.addLayout(grid)