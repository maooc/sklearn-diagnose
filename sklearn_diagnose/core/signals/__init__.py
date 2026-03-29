"""
Signal extraction module for sklearn-diagnose.

This module provides diagnostic signal extraction with a design inspired by
scikit-learn's inspection module. It supports:

- Automatic registration of signal extractors
- Standardized SignalResult containers
- Category-based signal organization
- Both class-based and function-based extractors

Main Entry Points
-----------------
- `extract_all_signals`: Extract all signals from evidence
- `extract_single_signal`: Extract a specific signal
- `extract_signals_by_category`: Extract signals by category
- `register_signal`: Decorator to register custom signals

Examples
--------
>>> from sklearn_diagnose.core.signals import extract_all_signals, register_signal
>>> from sklearn_diagnose.core.signals import SignalCategory, SignalResult
>>>
>>> # Extract all signals
>>> signals = extract_all_signals(evidence)
>>> print(signals["cv_mean"].value)
>>>
>>> # Extract by category
>>> cv_signals = extract_signals_by_category(evidence, SignalCategory.CV)
>>>
>>> # Register custom signal
>>> @register_signal("my_signal", category=SignalCategory.PERFORMANCE)
... def my_signal_extractor(evidence):
...     return SignalResult(name="my_signal", value=0.5, category=SignalCategory.PERFORMANCE)
"""

# Core classes
from ._base import (
    SignalExtractor,
    SignalResult,
    SignalRegistry,
    SignalCategory,
    SignalsCollection,
    get_global_registry,
)

# Registry utilities
from ._registry import (
    register_signal,
    create_extractor,
    get_registered_signals,
    unregister_signal,
    register_extractor_class,
    get_signal_categories,
    get_extractors_by_category,
)

# Factory functions
from ._factory import (
    extract_all_signals,
    extract_single_signal,
    extract_signals_by_category,
    extract_signals_to_legacy_format,
    get_available_signals,
    get_available_extractors,
    compute_score,
    analyze_cv_stability,
)

# Convenience aliases for sklearn-style API
from ._factory import get_available_signals as list_signals
from ._registry import get_signal

# Category-specific extractors (for advanced usage)
from .performance import (
    PerformanceSignalExtractor,
    ScoreBasedSignalExtractor,
)
from .cv_based import (
    CVSignalExtractor,
    CVFoldAnalyzer,
)
from .distribution import (
    DistributionSignalExtractor,
    ClassBalanceAnalyzer,
)
from .feature import (
    FeatureSignalExtractor,
    FeatureRedundancyAnalyzer,
)
from .leakage import (
    LeakageSignalExtractor,
    PreprocessingLeakageDetector,
)

__all__ = [
    # Core classes
    "SignalExtractor",
    "SignalResult",
    "SignalRegistry",
    "SignalCategory",
    "SignalsCollection",
    "get_global_registry",
    
    # Registry utilities
    "register_signal",
    "create_extractor",
    "get_registered_signals",
    "unregister_signal",
    "register_extractor_class",
    "get_signal_categories",
    "get_extractors_by_category",
    
    # Factory functions
    "extract_all_signals",
    "extract_single_signal",
    "extract_signals_by_category",
    "extract_signals_to_legacy_format",
    "get_available_signals",
    "get_available_extractors",
    "compute_score",
    "analyze_cv_stability",
    
    # Convenience aliases
    "list_signals",
    "get_signal",
    
    # Extractor classes (for advanced usage)
    "PerformanceSignalExtractor",
    "ScoreBasedSignalExtractor",
    "CVSignalExtractor",
    "CVFoldAnalyzer",
    "DistributionSignalExtractor",
    "ClassBalanceAnalyzer",
    "FeatureSignalExtractor",
    "FeatureRedundancyAnalyzer",
    "LeakageSignalExtractor",
    "PreprocessingLeakageDetector",
]
