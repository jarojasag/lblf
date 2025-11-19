import pickle
import json
import yaml
import numpy as np
from typing import Any, Dict

def save_results(results: Dict, filepath: str):
    """Save simulation results to file"""
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {filepath}")

def load_results(filepath: str) -> Dict:
    """Load simulation results from file"""
    with open(filepath, 'rb') as f:
        results = pickle.load(f)
    print(f"Results loaded from {filepath}")
    return results

def export_metrics_to_csv(metrics: Dict, filepath: str):
    """Export metrics to CSV file"""
    import pandas as pd
    df = pd.DataFrame([metrics])
    df.to_csv(filepath, index=False)
    print(f"Metrics exported to {filepath}")

def validate_config(config_file: str) -> bool:
    """Validate configuration file structure"""
    required_sections = [
        'population', 'initial_conditions', 'radicalization',
        'economic', 'elite', 'inequality', 'climate',
        'political_stress', 'trust', 'policy', 'thresholds'
    ]
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        for section in required_sections:
            if section not in config:
                print(f"Missing required section: {section}")
                return False
        
        print("Configuration file is valid")
        return True
    except Exception as e:
        print(f"Error validating config: {e}")
        return False

def create_experiment_config(base_config: str, modifications: Dict, 
                            output_file: str):
    """Create modified config file for experiments"""
    with open(base_config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply modifications
    for path, value in modifications.items():
        keys = path.split('.')
        target = config
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value
    
    # Save modified config
    with open(output_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Experiment config saved to {output_file}")
    return output_file