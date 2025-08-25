import json
import time
import yaml
import numpy as np

def run_log(logpath = None, **kwargs):

    log_dict = {}
    for k, v in kwargs.items():
        log_dict[k] = v

    if logpath is None:
        logpath = f"illum_stats_{int(time.time())}.json"

    with open(logpath, "w") as fp:
        json.dump(log_dict, fp, indent=4)

def load_config_yaml(file_path):
    """Load configuration from a YAML file."""
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)  # safe_load prevents execution of arbitrary code
    return config

def promote_to_array_if_necessary(value, shape, dtype=None):
    if isinstance(value, (int, float)):
        return np.full(shape, value, dtype)
    elif isinstance(value, np.ndarray):
        if value.shape != shape:
            raise ValueError('invalid shape')
        return value
    else:
        raise TypeError(f'bad type: {type(value)}')