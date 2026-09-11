"""TinyLM3: a transparent, CPU-first teaching model, not a production trainer."""
from .config import Config
from .model import TinyLM

__all__ = ["Config", "TinyLM"]
