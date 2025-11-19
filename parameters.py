import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np

@dataclass
class ModelParameters:
    """Model parameters loaded from config.yml"""
    
    def __init__(self, config_file: str = "config.yml"):
        """Initialize parameters from config file"""
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Flatten nested config for easy attribute access
        self._set_attributes_from_config(self.config)
    
    def _set_attributes_from_config(self, config: Dict, prefix: str = ""):
        """Recursively set attributes from nested config"""
        for key, value in config.items():
            if isinstance(value, dict):
                # For nested dicts, recurse
                self._set_attributes_from_config(value, f"{prefix}{key}_")
            else:
                # Set flattened attribute
                attr_name = f"{prefix}{key}"
                setattr(self, attr_name, value)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get nested config value using dot notation"""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key, default)
            else:
                return default
        return value