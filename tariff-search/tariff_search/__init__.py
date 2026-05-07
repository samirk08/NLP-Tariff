"""
Tariff Search Module

A fast and efficient search tool for finding similar tariff descriptions
across historical US tariff data from 1789-2023.
"""

from .searcher import TariffSearch
from .data_preparation import prepare_data_files

__version__ = "0.1.0"
__all__ = ["TariffSearch", "prepare_data_files"]