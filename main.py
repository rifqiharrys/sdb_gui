import sys
import logging
import yaml
from PyQt5.QtWidgets import QApplication
from gui.widgets.main_widget import SDBWidget
from gui.core.utils import setup_logging

logger = logging.getLogger(__name__)

def main():
    app = QApplication(sys.argv)
    try:
        with open('gui/config/logging.yaml', 'r') as f:
            logging_config = yaml.safe_load(f)
            # Pass only the filename for basic setup, or config dict for custom setup
            setup_logging(log_file='sdb_gui.log', config=logging_config)
    except Exception as e:
        logger.error(f"Failed to load logging config: {e}")
        setup_logging(log_file='sdb_gui.log')  # Fallback to default
    
    logger.info('Starting application')
    widget = SDBWidget()
    widget.show()
    return app.exec_()

if __name__ == '__main__':
    sys.exit(main())