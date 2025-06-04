"""About and licenses dialog"""

import logging
from PyQt5.QtWidgets import (
    QDialog, QLabel, QTextBrowser, QPushButton,
    QGridLayout
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from gui.core.utils import resource_path

logger = logging.getLogger(__name__)

class LicensesDialog(QDialog):
    """Dialog for showing licenses information"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUI()

    def setupUI(self):
        """Initialize dialog UI"""
        self.setWindowTitle('Licenses')
        self.setWindowIcon(
            QIcon(resource_path('icons/info-pngrepo-com.png'))
        )

        grid = QGridLayout()
        row = 1

        licensesBrowser = QTextBrowser()
        try:
            with open(resource_path('licenses/LICENSES'), 'r') as f:
                licensesBrowser.setText(f.read())
        except Exception as e:
            logger.error(f'Failed to load licenses: {e}')
            licensesBrowser.setText('Failed to load licenses information')
        
        grid.addWidget(licensesBrowser, row, 1, 1, 4)

        row += 1
        okButton = QPushButton('OK')
        okButton.clicked.connect(self.accept)
        grid.addWidget(okButton, row, 4, 1, 1)

        self.setLayout(grid)