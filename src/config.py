import argparse
import logging
import os
import sys
from importlib import resources
import yaml
from jinja2 import Template
from pathlib import Path
# from shadowspy.utilities import load_config_yaml

logging.basicConfig(level=logging.INFO)

class ShSpOpt:
    _instance = None

    def __new__(cls, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.__init__(**kwargs)
        return cls._instance

    def __init__(self, **kwargs):
        if not hasattr(self, 'initialized'):
            self.__dict__.update(kwargs)
            self.initialized = True

    def setup_config(self, **kwargs):
        # # Load default configuration
        # with resources.open_text(__package__, 'default_config.yaml') as f:
        #     default_config = yaml.safe_load(f)
        cfg_path = Path(__file__).parent / "default_config.yaml"
        with cfg_path.open("r") as f:
            default_config = yaml.safe_load(f)

        self.update_config(**default_config)

        # Handling command-line arguments to override
        if hasattr(sys, 'argv'):
            parser = argparse.ArgumentParser(description="Dynamic Configuration for ShSpOpt")
            for key, value in self.__dict__.items():
                if key != "initialized":
                    parser.add_argument(f"--{key}", type=type(value), default=None, help=f"Set {key}")
            args, unknown = parser.parse_known_args()  # this allows the notebook to ignore argv that Jupyter uses
            cli_config = {k: v for k, v in vars(args).items() if v is not None}
            self.update_config(**cli_config)

        # Optionally load user configuration if specified
        config_file = kwargs.get('config_file', None) or getattr(self, 'config_file', None)
        if config_file:
            user_config = self.load_config_yaml(config_file, args)
            self.update_config(**user_config)

        # Display final configuration
        if 'show' in kwargs:
            self.display()

    def update_config(self, **config):
        self.__dict__.update(config)
        # Handling .wkt files or other special cases
        self.load_wkt_configs()

    def load_wkt_configs(self):
        """Method to load additional .wkt configurations if applicable."""
        pass

    def load_config_yaml(self, path, args):
        # Step 1: Load the YAML content
        with open(path, "r") as f:
            template_content = f.read()

        # Step 2: Load the variables section from the YAML (initial parsing)
        variables_section = yaml.safe_load(template_content) #.get('variables', {})
        # Step 3: Render the template using Jinja2 with the extracted variables
        template = Template(template_content)
        rendered_content = template.render(variables=variables_section, siteid=args.siteid)
        # Step 4: Load the final YAML after rendering the variables
        config = yaml.safe_load(rendered_content)

        return config

    def display(self):
        print("Current Configuration:", self.__dict__)

    @staticmethod
    def get_instance():
        if ShSpOpt._instance is None:
            ShSpOpt()
        return ShSpOpt._instance

    def display(self):
        for key in self.__dict__:
            print(f"{key} = {getattr(self, key)}")

    @staticmethod
    def check_consistency():
        return

    def get(self, name):
        return self.__dict__[name]

    def set(self, **kwargs):
        for name, value in kwargs.items():
            self.__dict__[name] = value
            print(f"### ShSpOpt.{name} updated to {value}.")

    def to_yaml(self, file_path):
        """Dumps the configuration dictionary to a YAML file."""
        with open(file_path, 'w') as file:
            yaml.dump(self.__dict__, file, default_flow_style=False)
        print(f"Configuration saved to {file_path}")

    def from_yaml(file_path):
        """Reads the configuration dictionary from a YAML file."""
        with open(file_path, 'r') as file:
            loaded_config = yaml.safe_load(file)
        return ShSpOpt(**loaded_config)
       
    @staticmethod
    def to_dict():
        return ShSpOpt.__dict__

    @staticmethod
    def clone(opts):
        ShSpOpt.__conf = opts.copy()

