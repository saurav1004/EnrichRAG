import yaml
import os
import shutil
from datetime import datetime

# This dict will hold our single, global config object
settings = {}

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def load_config(config_file):
    """
    Loads a base config, merges it with a dataset-specific config,
    and saves copies of the configs to the experiment directory.
    """
    global settings
    
    # --- Robustly find base.yaml ---
    config_dir = os.path.dirname(config_file)
    base_config_path = os.path.join(config_dir, "base.yaml")
    if not os.path.exists(base_config_path):
        # If not found, check the parent directory
        parent_dir = os.path.dirname(config_dir)
        base_config_path = os.path.join(parent_dir, "base.yaml")
    
    if not os.path.exists(base_config_path):
        raise FileNotFoundError(f"base.yaml not found in {config_dir} or its parent directory.")
        
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"{config_file} not found.")

    # 1. Load base config
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # 2. Load dataset-specific config
    with open(config_file, 'r') as f:
        dataset_config = yaml.safe_load(f)
        
    # 3. Merge them
    settings.update(base_config)
    settings.update(dataset_config)
    
    # 4. Create a unique experiment directory
    dataset_name = os.path.splitext(os.path.basename(config_file))[0]
    experiment_name = f"{dataset_name}_{get_timestamp()}"
    settings['experiment_path'] = os.path.join(settings['output_dir'], experiment_name)
    os.makedirs(settings['experiment_path'], exist_ok=True)
    
    # 5. Save the configs to the experiment directory
    try:
        shutil.copy(config_file, settings['experiment_path'])
        shutil.copy(base_config_path, settings['experiment_path'])
    except Exception as e:
        print(f"Warning: Could not save config files to experiment directory. Error: {e}")
    
    print(f"Config loaded for experiment: {experiment_name}")
    return settings

def get_config():
    """Returns the loaded global settings."""
    global settings
    if not settings:
        raise Exception("Config not loaded. Call load_config(path) first.")
    return settings