import os
import sys

if getattr(sys, 'frozen', False):
    os.environ['GDAL_DATA'] = os.path.join(sys._MEIPASS, 'gdal')