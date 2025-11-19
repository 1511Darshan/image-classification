"""
Initialization file for src module.
"""

from .data import load_mnist_raw, get_data_shapes, get_data_statistics, get_class_distribution
from .model import MNISTClassifier, DecisionTreeModel, LogisticRegressionModel
from .utils import save_results, get_classification_report, print_model_comparison, set_seed

__version__ = "1.0.0"
__author__ = "Darshan (1511Darshan)"

__all__ = [
    "load_mnist_raw",
    "get_data_shapes",
    "get_data_statistics",
    "get_class_distribution",
    "MNISTClassifier",
    "DecisionTreeModel",
    "LogisticRegressionModel",
    "save_results",
    "get_classification_report",
    "print_model_comparison",
    "set_seed",
]
