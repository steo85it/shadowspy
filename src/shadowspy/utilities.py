import json
import time
import yaml

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
