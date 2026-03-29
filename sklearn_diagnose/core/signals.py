"""
Deterministic signal extraction for sklearn-diagnose.

This module computes quantitative statistics from the evidence.
All computations are deterministic and reproducible.

This module has been refactored to use a registry-based approach
with standardized SignalResult return format.

Signal extractors are organized by category:
- Performance signals (train/val scores, gaps)
- CV signals (mean, std, fold analysis)
- Distribution signals (class distribution, residual analysis)
- Feature signals (correlations, importance, redundancy)
- Leakage signals (suspicious patterns, risk assessment)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

# Import the signal registry and base classes
from .base import BaseSignalExtractor, SignalExtractorRegistry, signal_registry
from .schemas import Evidence, SignalCategory, SignalResult, Signals, TaskType

# Import all signal extractor modules to ensure registration
from . import cv_based_signals as _cv_based_signals
from . import distribution_signals as _distribution_signals
from . import feature_signals as _feature_signals
from . import leakage_signals as _leakage_signals
from . import score_based_signals as _score_based_signals

# Re-export utility functions for backward compatibility
from .cv_based_signals import analyze_cv_stability
from .score_based_signals import compute_score

__all__ = [
    "extract_all_signals",
    "extract_signals",
    "compute_score",
    "analyze_cv_stability",
    "signal_registry",
    "SignalResult",
    "SignalCategory",
    "BaseSignalExtractor",
    "SignalExtractorRegistry",
]


def extract_signals(
    evidence: Evidence,
    extractor_names: Optional[List[str]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> List[SignalResult]:
    """
    Extract signals using specified extractors.
    
    This is the new API that returns standardized SignalResult objects.
    
    Args:
        evidence: Evidence object with all diagnostic inputs
        extractor_names: Optional list of extractor names to use.
                       If None, use all registered extractors.
        params: Optional parameters to pass to extractors
        
    Returns:
        List of SignalResult objects with computed signals
        
    Examples:
        >>> evidence = Evidence(...)
        >>> signals = extract_signals(evidence)
        >>> for result in signals:
        ...     print(f"{result.name}: {result.value}")
        
        >>> # Extract specific signals
        >>> signals = extract_signals(evidence, ["train_score", "val_score", "cv_scores"])
    """
    return signal_registry.extract(evidence, extractor_names, params)


def extract_all_signals(evidence: Evidence) -> Signals:
    """
    Extract all signals from the provided evidence.
    
    This is the legacy API that returns a Signals object for backward compatibility.
    For new code, consider using extract_signals() which returns standardized SignalResult objects.
    
    Args:
        evidence: Evidence object with all diagnostic inputs
        
    Returns:
        Signals object with all computed statistics
    """
    signals = Signals()
    
    # Extract basic data characteristics
    signals.n_samples_train = evidence.n_samples_train
    signals.n_samples_val = evidence.n_samples_val
    signals.n_features = evidence.n_features
    
    if evidence.n_samples_train > 0 and evidence.n_features > 0:
        signals.feature_to_sample_ratio = evidence.n_features / evidence.n_samples_train
    
    # Use new registry-based extraction and populate legacy Signals object
    signal_results = extract_signals(evidence)
    
    # Map SignalResult objects to legacy Signals attributes
    _map_signal_results_to_legacy(signal_results, signals)
    
    return signals


def _map_signal_results_to_legacy(
    signal_results: List[SignalResult],
    signals: Signals,
) -> None:
    """
    Map SignalResult objects to legacy Signals object attributes.
    
    This is a compatibility layer to maintain backward compatibility
    with the existing API.
    """
    signal_dict = {result.name: result.value for result in signal_results}
    
    # Performance signals
    if "train_score" in signal_dict:
        signals.train_score = signal_dict["train_score"]
    if "val_score" in signal_dict:
        signals.val_score = signal_dict["val_score"]
    if "train_val_gap" in signal_dict:
        signals.train_val_gap = signal_dict["train_val_gap"]
    
    # CV signals
    if "cv_mean" in signal_dict:
        signals.cv_mean = signal_dict["cv_mean"]
    if "cv_std" in signal_dict:
        signals.cv_std = signal_dict["cv_std"]
    if "cv_min" in signal_dict:
        signals.cv_min = signal_dict["cv_min"]
    if "cv_max" in signal_dict:
        signals.cv_max = signal_dict["cv_max"]
    if "cv_range" in signal_dict:
        signals.cv_range = signal_dict["cv_range"]
    if "cv_fold_scores" in signal_dict:
        signals.cv_fold_scores = signal_dict["cv_fold_scores"]
    if "cv_train_mean" in signal_dict:
        signals.cv_train_mean = signal_dict["cv_train_mean"]
    if "cv_train_val_gap" in signal_dict:
        signals.cv_train_val_gap = signal_dict["cv_train_val_gap"]
    if "cv_holdout_gap" in signal_dict:
        signals.cv_holdout_gap = signal_dict["cv_holdout_gap"]
    
    # Distribution signals - Classification
    if "class_distribution" in signal_dict:
        signals.class_distribution = signal_dict["class_distribution"]
    if "minority_class_ratio" in signal_dict:
        signals.minority_class_ratio = signal_dict["minority_class_ratio"]
    if "confusion_matrix" in signal_dict:
        signals.confusion_matrix = signal_dict["confusion_matrix"]
    if "per_class_recall" in signal_dict:
        signals.per_class_recall = signal_dict["per_class_recall"]
    if "per_class_precision" in signal_dict:
        signals.per_class_precision = signal_dict["per_class_precision"]
    
    # Distribution signals - Regression
    if "residual_mean" in signal_dict:
        signals.residual_mean = signal_dict["residual_mean"]
    if "residual_std" in signal_dict:
        signals.residual_std = signal_dict["residual_std"]
    if "residual_skew" in signal_dict:
        signals.residual_skew = signal_dict["residual_skew"]
    if "residual_kurtosis" in signal_dict:
        signals.residual_kurtosis = signal_dict["residual_kurtosis"]
    
    # Feature signals
    if "feature_correlations" in signal_dict:
        signals.feature_correlations = signal_dict["feature_correlations"]
    if "high_correlation_pairs" in signal_dict:
        signals.high_correlation_pairs = signal_dict["high_correlation_pairs"]
    if "feature_importances" in signal_dict:
        signals.feature_importances = np.array(list(signal_dict["feature_importances"].values()))
    if "feature_target_correlations" in signal_dict:
        signals.feature_target_correlations = np.array(list(signal_dict["feature_target_correlations"].values()))
    
    # Leakage signals
    if "suspicious_feature_correlations" in signal_dict:
        signals.suspicious_feature_correlations = signal_dict["suspicious_feature_correlations"]


# Convenience functions for accessing the registry
def list_registered_extractors() -> Dict[str, Dict[str, Any]]:
    """
    List all registered signal extractors with their metadata.
    
    Returns:
        Dictionary mapping extractor names to their metadata
    """
    return signal_registry.list_all()


def get_extractor(name: str) -> BaseSignalExtractor:
    """
    Get a signal extractor by name.
    
    Args:
        name: Name of the extractor to retrieve
        
    Returns:
        The signal extractor instance
        
    Raises:
        ValueError: If extractor with given name is not registered
    """
    return signal_registry.get(name)


def register_extractor(
    name: str,
    extractor: BaseSignalExtractor,
    category: SignalCategory,
    tags: Optional[List[str]] = None,
    force: bool = False,
) -> None:
    """
    Register a custom signal extractor.
    
    This allows users to extend the signal extraction system with
    their own extractors.
    
    Args:
        name: Unique name for the extractor
        extractor: The signal extractor instance
        category: Category of the signal extractor
        tags: Optional tags for filtering
        force: If True, overwrite existing extractor with the same name
        
    Raises:
        ValueError: If extractor with same name already exists and force=False
        
    Examples:
        >>> from sklearn_diagnose.core import register_extractor, SignalCategory
        >>> from sklearn_diagnose.core import BaseSignalExtractor, SignalResult
        >>> 
        >>> class MyExtractor(BaseSignalExtractor):
        ...     category = SignalCategory.PERFORMANCE
        ...     def extract(self, evidence, params=None):
        ...         return [SignalResult(...)]
        >>> 
        >>> register_extractor("my_extractor", MyExtractor(), SignalCategory.PERFORMANCE)
    """
    signal_registry.register(name, extractor, category, tags, force)
