"""Utility functions."""

import os

import yaml


def load_yaml_config():
    """Load the json configuration."""
    config_path = os.getenv("CONFIG_PATH", "./config.yaml")
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    return config
