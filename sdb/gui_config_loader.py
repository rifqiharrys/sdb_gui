from typing import Any, Dict

import yaml

from .gui_utils import resource_path


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load GUI constants from a YAML configuration file.
    """
    try:
        config_file = resource_path(config_path)
        with open(config_file, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f'Configuration file not found: {config_file}')
    except yaml.YAMLError as e:
        raise RuntimeError(f'Error parsing YAML file: {e}')


CONSTANTS = load_config('config/constants.yaml')
DEFAULTS = load_config('config/defaults.yaml')