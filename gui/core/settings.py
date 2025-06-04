"""Settings management for SDB GUI"""

import json
import logging
import os
from collections import OrderedDict
from pathlib import Path
from PyQt5.QtCore import QSettings
from ..config.constants import SELECTION_TYPES

logger = logging.getLogger(__name__)

def default_values() -> dict:
    """Load default values from JSON config or fallback to hardcoded defaults"""
    try:
        config_path = Path(__file__).parent.parent / 'config' / 'defaults.json'
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading defaults: {e}")
        return _fallback_defaults()

def load_settings(settings: QSettings) -> dict:
    """Load settings or return defaults if none exist"""
    if settings.contains('options'):
        saved_settings = settings.value('options')
        if isinstance(saved_settings, dict):
            return saved_settings
    return default_values()

def save_settings(settings: QSettings, options: dict) -> None:
    """Save current settings"""
    settings.setValue('options', options)
    settings.sync()

def _fallback_defaults() -> dict:
    """Fallback default values if JSON fails to load"""
    random_selection = {
        'name': SELECTION_TYPES['RANDOM'],
        'parameters': OrderedDict([
            ('train_size', 0.75),
            ('random_state', 0)
        ])
    }

    attribute_selection = {
        'name': SELECTION_TYPES['ATTRIBUTE'],
        'parameters': OrderedDict([
            ('header', ''),
            ('group', '')
        ])
    }

    return {
        'processing': {
            'depth_limit': {
                'disable': False,
                'upper': 0.0,
                'lower': -15.0
            },
            'backend': 'threading',
            'n_jobs': -2,
            'selection': OrderedDict([
                (random_selection['name'], random_selection),
                (attribute_selection['name'], attribute_selection)
            ]),
            'current_selection': random_selection['name'],
            'backend_set': ('loky', 'threading', 'multiprocessing')
        }
    }