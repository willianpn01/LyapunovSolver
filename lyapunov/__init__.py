"""
LyapunovSolver-Hybrid v2.0
Optimized Architecture for Lyapunov Stability Analysis
"""

import logging

from .system_definition import SystemDefinition
from .symbolic_engine import SymbolicEngine
from .cache_manager import CacheManager
from .code_generator import CCodeGenerator
from .lyapunov_system import LyapunovSystem
from .visualization import LyapunovVisualizer
from .cli import LyapunovCLI
from .analysis import EquilibriumScanner, EquilibriumPoint, EquilibriumClassifier

__version__ = "2.0.0"
__author__ = "LyapunovSolver Team"

__all__ = [
    "SystemDefinition",
    "SymbolicEngine", 
    "CacheManager",
    "CCodeGenerator",
    "LyapunovSystem",
    "LyapunovVisualizer",
    "LyapunovCLI",
    "EquilibriumScanner",
    "EquilibriumPoint",
    "EquilibriumClassifier",
    "enable_debug_logging",
]


def enable_debug_logging(level: str = "DEBUG", log_file: str = None):
    """
    Enable debug logging for Lyapunov computations.
    
    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Optional file path to write logs (None = console only)
    
    Example:
        >>> import lyapunov
        >>> lyapunov.enable_debug_logging()  # Console output
        >>> lyapunov.enable_debug_logging(log_file="lyapunov.log")  # File output
    """
    log_level = getattr(logging, level.upper(), logging.DEBUG)
    
    logger = logging.getLogger("lyapunov")
    logger.setLevel(log_level)
    
    if not logger.handlers:
        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        if log_file:
            file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    logger.info(f"Debug logging enabled (level={level})")
    return logger
