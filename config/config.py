import json
import os

# load configs from JSON file
def load_config(config_path):
  
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file(s) not found at {config_path}")
  
    with open(config_path, 'r') as f:
        config = json.load(f)

    return config

# get configs 
def get_config(configs, path):

    # split up the path (form: 'section.subsection.subsubsection = value')
    keys = path.split('.')
    value = configs

    # move through the nested dicts to get the end value
    for key in keys:
        value = value[key]

    return value

def validate_configs(config):

    if config.dimens == 2 and config.strategy == 'lemni':
        raise ValueError("Lemniscate trajectories not supported in 2D. Please choose different strategy.")

    if config.dimens == 2 and config.strategy == 'cao':
        raise ValueError("Cao not adapted for 2D yet.")



# immutable class used to store high level configuration parameters
class Config:
    
    # define which top-level sections to auto-extract
    HYPER_CONFIGS = ['simulation', 'agents', 'orchestrator', 'data']
    
    def __init__(self, config_path):
        
        data = load_config(config_path)
        object.__setattr__(self, '_data', data)
        object.__setattr__(self, 'config_path', config_path)
        
        try:
            # auto-extract all params from specified sections
            for section in self.HYPER_CONFIGS:
                self._extract_section(data[section])
                
        except KeyError as e:
            raise ValueError(f"Missing required HYPER CONFIG section: {e}")
        
        validate_configs(self)
    
    def _extract_section(self, section_data, prefix=''):

        # recursively extract section(s) data
        for key, value in section_data.items():
            
            # to enforce immutability, 
            if isinstance(value, dict):

                # skip any nested sections (like, dicts within root json sections)
                continue
                # self._extract_section(value, f"{prefix}_{key}" if prefix else key)
            
            else:
                # set scalar value as attribute
                object.__setattr__(self, key, value)
    
    def get_technique(self, technique_name):
        return get_config(self._data, f'planner.techniques.{technique_name}')
    
    def get_learner(self, learner_name):
        return get_config(self._data, f'learner.{learner_name}')
    
    # don't allow setting of attributes
    def __setattr__(self, name, value):
        raise AttributeError(f"Config object is immutable. Cannot set {name}")
    
    # don't allow deletion of attributes
    def __delattr__(self, name):
        raise AttributeError(f"Config object is immutable. Cannot delete {name}")
    