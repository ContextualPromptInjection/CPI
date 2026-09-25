"""Three-stage Contextual Prompt Injection pipeline."""

from .adapters import normalize_path
from .injector import inject_repositories
from .payloads import generate_payloads

__all__ = [
    "generate_payloads",
    "inject_repositories",
    "normalize_path",
]

__version__ = "0.1.0"
