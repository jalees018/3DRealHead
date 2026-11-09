import os
from types import SimpleNamespace

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_paths(config_file):
    """Load the path config, resolving relative entries against the repository root."""
    with open(config_file) as f:
        entries = yaml.safe_load(f) or {}
    resolved = {key: value if os.path.isabs(value) else os.path.join(REPO_ROOT, value)
                for key, value in entries.items()}
    return SimpleNamespace(**resolved)
