import pprint
from collections import OrderedDict
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

def structure_defaults(defaults_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert loaded defaults from YAML into the expected structure
    with proper nested OrderedDicts for compatibility with existing code.
    """
    
    # Convert selection to proper nested structure with 'name' key
    if 'processing' in defaults_dict and 'selection' in defaults_dict['processing']:
        selection_dict = OrderedDict()
        for selection_name, selection_data in defaults_dict['processing']['selection'].items():
            selection_dict[selection_name] = {
                'name': selection_name,
                'alias': selection_data.get('alias', ''),  # ADD THIS LINE
                'parameters': OrderedDict(selection_data.get('parameters', {}))
            }
        defaults_dict['processing']['selection'] = selection_dict
    
    # # Ensure processing dict has backend_set for backward compatibility
    # if 'processing' in defaults_dict:
    #     defaults_dict['processing'] = OrderedDict(defaults_dict['processing'])
    
    # Convert method dicts to include names and OrderedDict parameters
    if 'method' in defaults_dict:
        method_dict = OrderedDict()
        for method_name, method_data in defaults_dict['method'].items():
            method_dict[method_name] = {
                'name': method_name,
                'model_parameters': OrderedDict(method_data.get('model_parameters', {})),
                **{k: v for k, v in method_data.items() if k != 'model_parameters'}
            }
        defaults_dict['method'] = method_dict
    
    # Convert main to OrderedDict
    if 'main' in defaults_dict:
        defaults_dict['main'] = OrderedDict(defaults_dict['main'])
    
    # Convert save to OrderedDict
    if 'save' in defaults_dict:
        defaults_dict['save'] = OrderedDict(defaults_dict['save'])
    
    return defaults_dict


CONSTANTS = load_config('config/constants.yaml')
DEFAULTS = load_config('config/defaults.yaml')
# DEFAULTS = structure_defaults(load_config('config/defaults.yaml'))