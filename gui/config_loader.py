from collections import OrderedDict
from typing import Any

import tomllib
import yaml

from .utils import resource_path


def load_config(config_path: str) -> dict[str, Any]:
    """
    Load GUI constants from a YAML or TOML configuration file.
    File type is determined by the extension (.yaml, .yml, or .toml).
    """
    try:
        config_file = resource_path(config_path)
        
        if config_path.endswith(('.toml')):
            with open(config_file, 'rb') as file:
                return tomllib.load(file)
        elif config_path.endswith(('.yaml', '.yml')):
            with open(config_file, 'r') as file:
                return yaml.safe_load(file)
        else:
            raise ValueError(f'Unsupported file format. Expected .yaml, .yml, or .toml but got: {config_path}')
    except FileNotFoundError:
        raise FileNotFoundError(f'Configuration file not found: {config_file}')
    except yaml.YAMLError as e:
        raise RuntimeError(f'Error parsing YAML file: {e}')
    except tomllib.TOMLDecodeError as e:
        raise RuntimeError(f'Error parsing TOML file: {e}')


def structure_defaults(defaults_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Convert loaded defaults from YAML into the expected structure
    with proper nested OrderedDicts for compatibility with existing code.
    """
    
    if 'processing' in defaults_dict and 'selection' in defaults_dict['processing']:
        selection_dict = OrderedDict()
        for selection_name, selection_data in defaults_dict['processing']['selection'].items():
            selection_dict[selection_name] = {
                'alias': selection_data.get('alias', ''),
                'parameters': OrderedDict(selection_data.get('parameters', {}))
            }
        defaults_dict['processing']['selection'] = selection_dict

    if 'method' in defaults_dict:
        method_dict = {}
        for method_name, method_data in defaults_dict['method'].items():
            method_dict[method_name] = {
                'model_parameters': OrderedDict(method_data.get('model_parameters', {})),
                **{k: v for k, v in method_data.items() if k != 'model_parameters'}
            }
        defaults_dict['method'] = method_dict

    return defaults_dict


PROJECT = load_config('pixi.toml')
CONSTANTS = load_config('config/constants.yaml')
DEFAULTS = structure_defaults(load_config('config/defaults.yaml'))