"""Process thread handling for SDB GUI"""

import logging
import numpy as np
import pandas as pd
from PyQt5.QtCore import QThread, pyqtSignal
import sdb

logger = logging.getLogger(__name__)

class Process(QThread):
    """Processing thread for SDB calculations"""
    thread_signal = pyqtSignal(dict)
    time_signal = pyqtSignal(str)
    warning_with_clear = pyqtSignal(str)

    def __init__(self):
        super(Process, self).__init__()
        self._is_running = True

    def inputs(self, input_dict):
        """Pooling inputs from widget"""
        self.depth_label = input_dict['depth_label']
        self.depth_direction = input_dict['depth_direction']
        self.limit_state = input_dict['limit_state']
        self.limit_a_value = input_dict['limit_a']
        self.limit_b_value = input_dict['limit_b']
        self.method = input_dict['method']
        self.train_select = input_dict['train_select']
        self.selection = input_dict['selection']

    def run(self):
        """Main processing thread"""
        try:
            # ...existing process run code...
            results = {}
            self.thread_signal.emit(results)
        except Exception as e:
            logger.error(f"Processing error: {e}")
            self.warning_with_clear.emit(str(e))

    def stop(self):
        """Stop the processing thread"""
        self._is_running = False
        self.quit()
        self.wait()