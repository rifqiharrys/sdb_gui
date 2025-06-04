"""Constant values for SDB GUI"""

from collections import OrderedDict
from typing import Dict, Tuple

# Application constants
SDB_GUI_VERSION: str = '4.1.0'
LOG_NAME: str = 'sdb_gui.log'
PROGRESS_STEP: int = 6

# Interface constants
DEPTH_DIRECTION: Dict[str, Tuple[str, bool]] = {
    'Positive Up': ('up', False),
    'Positive Down': ('down', True),
}

SELECTION_TYPES: Dict[str, str] = {
    'RANDOM': 'Random Selection',
    'ATTRIBUTE': 'Attribute Selection'
}

METHOD_TYPES = OrderedDict([
    ('K-Nearest Neighbors', 'KNN'),
    ('Multiple Linear Regression', 'MLR'),
    ('Random Forest', 'RF')
])