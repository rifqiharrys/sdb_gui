import os
import sys

if getattr(sys, 'frozen', False):
    gdal_data = os.path.join(sys._MEIPASS, 'gdal') # type: ignore
    if os.path.isdir(gdal_data):
        os.environ['GDAL_DATA'] = gdal_data