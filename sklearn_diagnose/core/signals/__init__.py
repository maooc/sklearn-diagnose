"""
Signal extraction module for sklearn-diagnose.

This module provides a modular, extensible signal extraction system inspired by
scikit-learn's inspection API design patterns.

Key components:
- SignalResult: Standardized return type for all signal extractors
- Signal: Abstract base class for signal extractors
- SignalRegistry: Registry for automatic signal discovery and registration
- extract_signals: Main entry point returning List[SignalResult]
- extract_all_signals: Backward-compatible entry returning Signals dataclass

Usage:
    from sklearn_diagnose.core.signals import extract_signals, SignalResult
    
    # Extract all signals (returns List[SignalResult])
    results = extract_signals(evidence)
    for result in results:
        print(f"{result.name}: {result.value}")
    
    # Extract signals by category
    from sklearn_diagnose.core.signals import extract_signals, SignalCategory
    results = extract_signals(evidence, categories=[SignalCategory.SCORE_BASED])
    
    # Register custom signals
    from sklearn_diagnose.core.signals import register_signal, Signal, SignalCategory
    
    @register_signal("my_custom_signal")
    class MyCustomSignal(Signal):
        name = "my_custom_signal"
        category = SignalCategory.SCORE_BASED
        
        def extract(self, evidence):
            return SignalResult(
                name=self.name,
                value=...,
                category=self.category
            )
"""

from typing import TYPE_CHECKING, Dict, List, Optional, Type

from .base import (
    Signal,
    SignalCategory,
    SignalFunction,
    SignalResult,
    signal,
)
from .registry import (
    DEFAULT_REGISTRY,
    SignalRegistry,
    clear_registry,
    get_signal,
    get_signals_by_category,
    list_signals,
    register_signal,
    register_signal_function,
)
from .score_based_signals import (
    SCORE_BASED_SIGNALS,
    RelativeGapSignal,
    TrainScoreSignal,
    TrainValGapSignal,
    ValScoreSignal,
    compute_score,
)
from .cv_based_signals import (
    CV_BASED_SIGNALS,
    CVFoldScoresSignal,
    CVHoldoutGapSignal,
    CVMeanSignal,
    CVOutlierFoldsSignal,
    CVRangeSignal,
    CVStabilitySignal,
    CVStdSignal,
    CVTrainTestGapSignal,
    analyze_cv_stability,
)
from .distribution_signals import (
    DISTRIBUTION_SIGNALS,
    ClassDistributionSignal,
    ClassImbalanceRatioSignal,
    ConfusionMatrixSignal,
    MinorityClassRatioSignal,
    PerClassPrecisionSignal,
    PerClassRecallSignal,
    ResidualKurtosisSignal,
    ResidualMeanSignal,
    ResidualSkewSignal,
    ResidualStdSignal,
)
from .feature_signals import (
    FEATURE_SIGNALS,
    FeatureCorrelationsSignal,
    FeatureImportanceAvailableSignal,
    FeatureTargetCorrelationsSignal,
    FeatureToSampleRatioSignal,
    HighCorrelationPairsSignal,
    NFeaturesSignal,
    NSamplesTrainSignal,
    ZeroVarianceFeaturesSignal,
)
from .leakage_signals import (
    LEAKAGE_SIGNALS,
    CVHoldoutDiscrepancySignal,
    PerfectTrainScoreSignal,
    SuspiciousFeatureCorrelationsSignal,
    TargetDistributionShiftSignal,
    TrainTestDistributionShiftSignal,
)

if TYPE_CHECKING:
    from ..schemas import Evidence, Signals


ALL_SIGNAL_CLASSES: List[Type[Signal]] = (
    SCORE_BASED_SIGNALS +
    CV_BASED_SIGNALS +
    DISTRIBUTION_SIGNALS +
    FEATURE_SIGNALS +
    LEAKAGE_SIGNALS
)


def extract_signals(
    evidence: "Evidence",
    categories: Optional[List[SignalCategory]] = None,
    registry: Optional[SignalRegistry] = None,
) -> List[SignalResult]:
    """
    Extract signals from evidence using the modular signal system.
    
    This is the main entry point for signal extraction, returning a list
    of SignalResult objects with standardized format.
    
    Args:
        evidence: Evidence object with all diagnostic inputs
        categories: Optional list of categories to limit extraction.
                   If None, extracts all categories.
        registry: Optional custom registry. If None, uses DEFAULT_REGISTRY.
        
    Returns:
        List of SignalResult objects, each containing:
        - name: Unique signal identifier
        - value: Computed signal value
        - category: Signal category enum
        - metadata: Additional context
        - confidence: Optional confidence score
        - is_anomaly: Whether this signal indicates an issue
        
    Example:
        results = extract_signals(evidence)
        for result in results:
            if result.is_anomaly:
                print(f"Anomaly detected: {result.name} = {result.value}")
    """
    reg = registry or DEFAULT_REGISTRY
    return reg.extract_all(evidence, categories)


def _signal_results_to_signals(results: List[SignalResult]) -> "Signals":
    """
    Convert List[SignalResult] to Signals dataclass for backward compatibility.
    
    This function maps the new SignalResult format to the legacy Signals
    dataclass fields.
    """
    from ..schemas import Signals
    import numpy as np
    
    signals = Signals()
    
    result_map: Dict[str, SignalResult] = {r.name: r for r in results}
    
    def get_value(name: str):
        result = result_map.get(name)
        return result.value if result else None
    
    signals.train_score = get_value("train_score")
    signals.val_score = get_value("val_score")
    signals.train_val_gap = get_value("train_val_gap")
    
    signals.cv_mean = get_value("cv_mean")
    signals.cv_std = get_value("cv_std")
    signals.cv_min = get_value("cv_min") if "cv_min" in result_map else None
    signals.cv_max = get_value("cv_max") if "cv_max" in result_map else None
    signals.cv_range = get_value("cv_range")
    signals.cv_fold_scores = get_value("cv_fold_scores")
    signals.cv_train_mean = get_value("cv_train_mean") if "cv_train_mean" in result_map else None
    signals.cv_train_val_gap = get_value("cv_train_test_gap")
    signals.cv_holdout_gap = get_value("cv_holdout_gap")
    
    signals.residual_mean = get_value("residual_mean")
    signals.residual_std = get_value("residual_std")
    signals.residual_skew = get_value("residual_skew")
    signals.residual_kurtosis = get_value("residual_kurtosis")
    
    signals.class_distribution = get_value("class_distribution")
    signals.minority_class_ratio = get_value("minority_class_ratio")
    signals.per_class_recall = get_value("per_class_recall")
    signals.per_class_precision = get_value("per_class_precision")
    cm = get_value("confusion_matrix")
    if cm is not None:
        signals.confusion_matrix = np.array(cm)
    
    fc = get_value("feature_correlations")
    if fc is not None:
        signals.feature_correlations = np.array(fc)
    hcp = get_value("high_correlation_pairs")
    if hcp:
        signals.high_correlation_pairs = [
            (item["feature_i"], item["feature_j"], item["correlation"]) for item in hcp
        ]
    ftc = get_value("feature_target_correlations")
    if ftc is not None:
        signals.feature_target_correlations = np.array(ftc)
    
    signals.n_samples_train = get_value("n_samples_train")
    signals.n_samples_val = get_value("n_samples_val") if "n_samples_val" in result_map else None
    signals.n_features = get_value("n_features")
    signals.feature_to_sample_ratio = get_value("feature_to_sample_ratio")
    
    signals.cv_holdout_gap = get_value("cv_holdout_discrepancy")
    sfc = get_value("suspicious_feature_correlations")
    if sfc:
        signals.suspicious_feature_correlations = [
            (item["feature_index"], item["correlation"]) for item in sfc
        ]
    
    return signals


def extract_all_signals(evidence: "Evidence") -> "Signals":
    """
    Extract all signals from the provided evidence.
    
    This function provides backward compatibility with the original API,
    returning a Signals dataclass. Internally, it uses the new modular
    signal extraction system.
    
    Args:
        evidence: Evidence object with all diagnostic inputs
        
    Returns:
        Signals dataclass with all computed statistics
    """
    results = extract_signals(evidence)
    return _signal_results_to_signals(results)


def get_available_signals() -> Dict[str, Dict[str, any]]:
    """
    Get information about all available signals.
    
    Returns:
        Dictionary mapping signal names to their metadata
    """
    return DEFAULT_REGISTRY.list()


def extract_signals_by_category(
    evidence: "Evidence",
    category: SignalCategory,
) -> List[SignalResult]:
    """
    Extract signals from a specific category.
    
    Args:
        evidence: Evidence object with all diagnostic inputs
        category: Signal category to extract
        
    Returns:
        List of SignalResult objects from the specified category
    """
    return extract_signals(evidence, categories=[category])


__all__ = [
    "SignalResult",
    "Signal",
    "SignalFunction",
    "SignalCategory",
    "signal",
    "SignalRegistry",
    "DEFAULT_REGISTRY",
    "register_signal",
    "register_signal_function",
    "get_signal",
    "get_signals_by_category",
    "list_signals",
    "clear_registry",
    "extract_signals",
    "extract_all_signals",
    "extract_signals_by_category",
    "get_available_signals",
    "compute_score",
    "analyze_cv_stability",
    "ALL_SIGNAL_CLASSES",
    "SCORE_BASED_SIGNALS",
    "CV_BASED_SIGNALS",
    "DISTRIBUTION_SIGNALS",
    "FEATURE_SIGNALS",
    "LEAKAGE_SIGNALS",
    "TrainScoreSignal",
    "ValScoreSignal",
    "TrainValGapSignal",
    "RelativeGapSignal",
    "CVMeanSignal",
    "CVStdSignal",
    "CVRangeSignal",
    "CVFoldScoresSignal",
    "CVTrainTestGapSignal",
    "CVHoldoutGapSignal",
    "CVStabilitySignal",
    "CVOutlierFoldsSignal",
    "ClassDistributionSignal",
    "MinorityClassRatioSignal",
    "ClassImbalanceRatioSignal",
    "PerClassRecallSignal",
    "PerClassPrecisionSignal",
    "ConfusionMatrixSignal",
    "ResidualMeanSignal",
    "ResidualStdSignal",
    "ResidualSkewSignal",
    "ResidualKurtosisSignal",
    "NFeaturesSignal",
    "NSamplesTrainSignal",
    "FeatureToSampleRatioSignal",
    "FeatureCorrelationsSignal",
    "HighCorrelationPairsSignal",
    "FeatureTargetCorrelationsSignal",
    "ZeroVarianceFeaturesSignal",
    "FeatureImportanceAvailableSignal",
    "SuspiciousFeatureCorrelationsSignal",
    "CVHoldoutDiscrepancySignal",
    "PerfectTrainScoreSignal",
    "TrainTestDistributionShiftSignal",
    "TargetDistributionShiftSignal",
]
