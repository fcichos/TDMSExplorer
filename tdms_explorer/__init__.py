"""
TDMS Explorer Package

A Python package for working with TDMS (Technical Data Management Streaming) files.

Main modules:
- core: Core TDMS file exploration functionality
- cli: Command line interface
- utils: Utility functions
"""

from .core import TDMSFileExplorer, list_tdms_files, create_animation_from_tdms
from .cli.cli import main as cli_main

__version__ = "1.0.0"
__author__ = "TDMS Explorer Team"
__license__ = "MIT"

# Package-level functions for convenience
list_tdms_files = list_tdms_files
create_animation_from_tdms = create_animation_from_tdms