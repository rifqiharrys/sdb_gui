"""
Satellite Derived Bathymetry GUI Package

This package provides a graphical user interface for the SDB library.
It handles image processing, model training, and depth prediction
for bathymetric analysis of satellite imagery.
"""

from gui.widgets.main_widget import SDBWidget
from gui.core.process import Process
from gui.core.settings import load_settings, save_settings

__version__ = '4.1.0'

__all__ = [
    'SDBWidget',
    'Process',
    'load_settings',
    'save_settings',
]