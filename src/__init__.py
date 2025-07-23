"""
Population Health ML Analysis Package

This package contains modules for machine learning analysis of population health data,
including data processing, model training, evaluation, and visualization.
"""

__version__ = "1.0.0"
__author__ = "Sanjay Basu"
__email__ = "sanjay.basu@example.com"

from . import data_processing
from . import models
from . import evaluation
from . import visualization
from . import utils

__all__ = [
    "data_processing",
    "models", 
    "evaluation",
    "visualization",
    "utils"
]

