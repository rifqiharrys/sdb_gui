"""
GUI Widgets for SDB

This package contains all the Qt widgets and dialogs
used to build the SDB GUI interface.
"""

from .main_widget import SDBWidget
from .dialogs.method import MethodDialog
from .dialogs.process import ProcessDialog
from .dialogs.load import LoadImageDialog, LoadSampleDialog
from .dialogs.save import SaveDialog
from .dialogs.about import LicensesDialog

__all__ = [
    'SDBWidget',
    'MethodDialog',
    'ProcessDialog',
    'LoadImageDialog',
    'LoadSampleDialog',
    'SaveDialog',
    'LicensesDialog'
]