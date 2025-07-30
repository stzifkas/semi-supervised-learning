"""
FlexMatch Implementation for CIFAR-10

A complete implementation of FlexMatch for semi-supervised learning
with Curriculum Pseudo Labeling (CPL) and adaptive class-specific thresholds.
"""

from flexmatch.flexmatch import FlexMatch
from flexmatch.models import create_model, WideResNet, wrn_28_2, wrn_28_10, wrn_40_2, wrn_40_10
from flexmatch.data_utils import create_data_loaders, CIFAR10SSL, CIFAR10_CLASSES

__version__ = "1.0.0"
__author__ = "FlexMatch Implementation"

__all__ = [
    "FlexMatch",
    "create_model",
    "WideResNet",
    "wrn_28_2",
    "wrn_28_10", 
    "wrn_40_2",
    "wrn_40_10",
    "create_data_loaders",
    "CIFAR10SSL",
    "CIFAR10_CLASSES"
] 