"""Method options dialog for SDB GUI"""

import logging
from collections import OrderedDict
from PyQt5.QtWidgets import (
    QDialog, QLabel, QComboBox, QSpinBox, QPushButton,
    QGridLayout
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from gui.core.utils import resource_path, to_title

logger = logging.getLogger(__name__)

class MethodDialog(QDialog):
    """Dialog for method-specific options"""
    
    def __init__(self, parent=None, method_options=None):
        super().__init__(parent)
        self.method_options = method_options
        self.option_widgets = {}
        self.setupUI()

    def setupUI(self):
        """Initialize dialog UI"""
        self.setWindowTitle('Method Options')
        self.setWindowIcon(
            QIcon(resource_path('icons/setting-tool-pngrepo-com.png'))
        )

        grid = QGridLayout()
        row = 1

        # Create widgets for each parameter
        for param, value in self.method_options['model_parameters'].items():
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
                if set_name in self.method_options:
                    widget.addItems(self.method_options[set_name])
                else:
                    widget.addItems([value])
                widget.setCurrentText(value)

            self.option_widgets[param] = widget
            grid.addWidget(widget, row, 3, 1, 2)
            row += 1

        # Add buttons
        loadButton = QPushButton('Load')
        loadButton.clicked.connect(self.accept)
        grid.addWidget(loadButton, row, 3, 1, 1)

        cancelButton = QPushButton('Cancel')
        cancelButton.clicked.connect(self.reject)
        grid.addWidget(cancelButton, row, 4, 1, 1)

        self.setLayout(grid)