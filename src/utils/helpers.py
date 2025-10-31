"""
Utility functions for the Murder Model project.
"""

import os
import json
import yaml
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_results(results: Dict[str, Any], output_path: str):
    """
    Save results to a JSON file.
    
    Args:
        results (dict): Results to save
        output_path (str): Path to save the results
    """
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

def ensure_dir(directory: str):
    """
    Create directory if it doesn't exist.
    
    Args:
        directory (str): Directory path to create
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_project_root() -> str:
    """
    Get the project root directory.
    
    Returns:
        str: Absolute path to project root
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))