"""Save options dialog"""

import logging
from PyQt5.QtWidgets import (
    QDialog, QLabel, QTextBrowser, QPushButton,
    QGridLayout, QFileDialog
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from gui.core.utils import resource_path

logger = logging.getLogger(__name__)

class SaveDialog(QDialog):
    """Dialog for save options"""
    
    def __init__(self, parent=None, dir_path=None):
        super().__init__(parent)
        self.dir_path = dir_path
        self.save_locations = []
        self.setupUI()

    def setupUI(self):
        """Initialize dialog UI"""
        self.setWindowTitle('Save Options')
        self.setWindowIcon(
            QIcon(resource_path('icons/save-pngrepo-com.png'))
        )

        grid = QGridLayout()
        row = 1

        # Save location list
        savelocLabel = QLabel('Save Locations:')
        grid.addWidget(savelocLabel, row, 1, 1, 2)

        row += 1
        self.savelocList = QTextBrowser()
        grid.addWidget(self.savelocList, row, 1, 1, 4)

        # Buttons
        row += 1
        browseButton = QPushButton('Browse')
        browseButton.clicked.connect(self.browseSaveLocation)
        grid.addWidget(browseButton, row, 1, 1, 1)

        saveButton = QPushButton('Save')
        saveButton.clicked.connect(self.accept)
        grid.addWidget(saveButton, row, 3, 1, 1)

        cancelButton = QPushButton('Cancel')
        cancelButton.clicked.connect(self.reject)
        grid.addWidget(cancelButton, row, 4, 1, 1)

        self.setLayout(grid)

    def browseSaveLocation(self):
        """Open file dialog for save location"""
        save_path = QFileDialog.getSaveFileName(
            self, 'Save File',
            self.dir_path,
            'GeoTIFF (*.tif)'
        )[0]
        if save_path:
            self.save_locations.append(save_path)
            self.savelocList.setText('\n'.join(self.save_locations))