"""Load dialogs for image and sample data"""

import logging
from PyQt5.QtWidgets import QDialog, QFileDialog
from PyQt5.QtCore import Qt
import pandas as pd
import rasterio

logger = logging.getLogger(__name__)

class LoadImageDialog(QDialog):
    """Dialog for loading satellite imagery"""
    
    def __init__(self, parent=None, dir_path=None):
        super().__init__(parent)
        self.dir_path = dir_path
        self.image_path = None
        self.showDialog()

    def showDialog(self):
        """Show file selection dialog"""
        file_filter = "GeoTIFF (*.tif);;All Files (*.*)"
        self.image_path, _ = QFileDialog.getOpenFileName(
            self, 'Open Image', self.dir_path, file_filter
        )
        if self.image_path:
            try:
                with rasterio.open(self.image_path) as src:
                    logger.info(f'Image loaded: {self.image_path}')
                    self.accept()
            except Exception as e:
                logger.error(f'Failed to load image: {e}')
                self.reject()
        else:
            self.reject()

class LoadSampleDialog(QDialog):
    """Dialog for loading depth samples"""
    
    def __init__(self, parent=None, dir_path=None):
        super().__init__(parent)
        self.dir_path = dir_path
        self.sample_path = None
        self.showDialog()

    def showDialog(self):
        """Show file selection dialog"""
        file_filter = "CSV (*.csv);;Excel (*.xlsx);;All Files (*.*)"
        self.sample_path, _ = QFileDialog.getOpenFileName(
            self, 'Open Sample', self.dir_path, file_filter
        )
        if self.sample_path:
            try:
                if self.sample_path.endswith('.csv'):
                    pd.read_csv(self.sample_path)
                else:
                    pd.read_excel(self.sample_path)
                logger.info(f'Sample loaded: {self.sample_path}')
                self.accept()
            except Exception as e:
                logger.error(f'Failed to load sample: {e}')
                self.reject()
        else:
            self.reject()