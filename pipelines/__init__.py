# pipelines/__init__.py
# -----------------------------------------------------------
# Hace que puedas hacer:
#   from pipelines import load_and_clean_data, run_training, run_prediction

from .preprocessing import load_and_clean_data
from .training_pipeline import run_training
from .prediction_pipeline import run_prediction

__all__ = [
    "load_and_clean_data",
    "run_training",
    "run_prediction",
]
