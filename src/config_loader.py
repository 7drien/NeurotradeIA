import json
from pathlib import Path

def load_config():
    """
    Loads the configuration from the config.json file at the project root.
    """
    try:
        config_path = Path(__file__).parent.parent / 'config.json'
        with open(config_path, 'r') as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error loading config.json: {e}")
        raise