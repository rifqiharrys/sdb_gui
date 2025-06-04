"""Processing options dialog for SDB GUI"""

import logging
from PyQt5.QtWidgets import (
    QDialog, QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
    QPushButton, QGridLayout, QWidget, QScrollArea
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from gui.core.utils import resource_path
from gui.config.constants import SELECTION_TYPES

logger = logging.getLogger(__name__)

class ProcessDialog(QDialog):
    """Dialog for processing options"""
    
    def __init__(self, parent=None, proc_options=None):
        super().__init__(parent)
        self.proc_options = proc_options
        self.selection_widgets = {}
        self.setupUI()

    def setupUI(self):
        """Initialize dialog UI"""
        self.setWindowTitle('Processing Options')
        self.setWindowIcon(
            QIcon(resource_path('icons/setting-tool-pngrepo-com.png'))
        )
        self.setFixedWidth(400)

        grid = QGridLayout()
        row = 1

        # Backend settings
        backendLabel = QLabel('Parallel Backend:')
        self.backendCB = QComboBox()
        self.backendCB.addItems(self.proc_options['backend_set'])
        self.backendCB.setCurrentText(self.proc_options['backend'])
        grid.addWidget(backendLabel, row, 1, 1, 2)
        grid.addWidget(self.backendCB, row, 3, 1, 2)

        # ... continue with other process dialog UI components ...