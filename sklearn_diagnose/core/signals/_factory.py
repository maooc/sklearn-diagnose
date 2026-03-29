"""
Factory functions for signal extraction.

Provides the main entry points for extracting signals from evidence,
similar to sklearn's inspection module patterns.
"""

from typing import Any, Dict, List, Optional, Union

from ._base import (
    SignalExtractor, 
    SignalResult, 
    SignalRegistry, 
    SignalCategory,
    SignalsCollection,
    get_global_registry
)
from ..schemas import Evidence, Signals, TaskType


def extract_all_signals(
    evidence: Evidence,
    extractors: Optional[List[str]] = None,
    categories: Optional[List[Union[SignalCategory, str]]] = None,
    exclude: Optional[List[str]] = None,
    return_collection: bool = True
) -> Union[SignalsCollection, Dict[str, SignalResult]]:
    """
    Extract all signals from evidence using registered extractors.
    
    Main entry point for signal extraction. Similar to
    sklearn.inspection.permutation_importance pattern.
    
    Parameters
    ----------
    evidence : Evidence
        Evidence object with all diagnostic inputs
    extractors : list of str, optional
        Specific extractor names to run. If None, runs all registered extractors.
    categories : list of SignalCategory or str, optional
        Filter extractors by categories
    exclude : list of str, optional
        Extractor names to exclude from extraction
    return_collection : bool, default=True
        If True, returns SignalsCollection. If False, returns dict.
    
    Returns
    -------
    SignalsCollection or dict
        Collection of SignalResult objects, keyed by signal name
    
    Examples
    --------
    >>> # Extract all signals
    >>> signals = extract_all_signals(evidence)
    >>> print(signals["cv_mean"].value)
    
    >>> # Extract only specific extractors
    >>> signals = extract_all_signals(evidence, extractors=["performance", "cv_based"])
    
    >>> # Extract by category
    >>> signals = extract_all_signals(evidence, categories=[SignalCategory.CV])
    
    >>> # Get all CV signals
    >>> cv_signals = signals.by_category(SignalCategory.CV)
    """
    registry = get_global_registry()
    collection = SignalsCollection()
    
    # Determine which extractors to run
    if extractors is not None:
        extractor_names = extractors
    elif categories is not None:
        extractor_names = []
        for cat in categories:
            extractor_names.extend(registry.list_signals(cat))
        extractor_names = list(dict.fromkeys(extractor_names))  # Remove duplicates
    else:
        extractor_names = registry.list_signals()
    
    if exclude:
        extractor_names = [n for n in extractor_names if n not in exclude]
    
    # Run each extractor
    for name in extractor_names:
        extractor = registry.get(name)
        if extractor is None:
            continue
        
        if not extractor.can_extract(evidence):
            continue
        
        try:
            results = extractor.extract(evidence)
            
            # Handle both single results and lists
            if isinstance(results, list):
                for result in results:
                    if isinstance(result, SignalResult):
                        collection.add(result)
            elif isinstance(results, SignalResult):
                collection.add(results)
        except Exception:
            # Skip failed extractions
            continue
    
    if return_collection:
        return collection
    else:
        return {name: collection[name] for name in collection}


def extract_signals_by_category(
    evidence: Evidence,
    category: Union[SignalCategory, str]
) -> SignalsCollection:
    """
    Extract signals of a specific category.
    
    Parameters
    ----------
    evidence : Evidence
        Evidence object with diagnostic inputs
    category : SignalCategory or str
        Category to extract
    
    Returns
    -------
    SignalsCollection
        Collection of signals of the specified category
    
    Examples
    --------
    >>> cv_signals = extract_signals_by_category(evidence, SignalCategory.CV)
    >>> for signal in cv_signals.values():
    ...     print(f"{signal.name}: {signal.value}")
    """
    return extract_all_signals(evidence, categories=[category])


def extract_single_signal(
    evidence: Evidence,
    signal_name: str
) -> Optional[SignalResult]:
    """
    Extract a single signal by name.
    
    Parameters
    ----------
    evidence : Evidence
        Evidence object with diagnostic inputs
    signal_name : str
        Name of the signal to extract
    
    Returns
    -------
    SignalResult or None
        The extracted signal, or None if not found or extraction failed
    
    Examples
    --------
    >>> result = extract_single_signal(evidence, "cv_mean")
    >>> if result:
    ...     print(result.value)
    """
    registry = get_global_registry()
    extractor = registry.get(signal_name)
    
    if extractor is None:
        return None
    
    if not extractor.can_extract(evidence):
        return SignalResult.failure(
            name=signal_name,
            error="Required evidence not available",
            category=getattr(extractor, 'category', SignalCategory.CUSTOM)
        )
    
    try:
        results = extractor.extract(evidence)
        
        # Handle both single results and lists
        if isinstance(results, list) and results:
            # Return first valid result
            for result in results:
                if isinstance(result, SignalResult) and result.is_valid:
                    return result
            return results[0] if results else None
        elif isinstance(results, SignalResult):
            return results
        
        return None
    except Exception as e:
        return SignalResult.failure(
            name=signal_name,
            error=str(e),
            category=getattr(extractor, 'category', SignalCategory.CUSTOM)
        )


def extract_signals_to_legacy_format(evidence: Evidence) -> Signals:
    """
    Extract signals and convert to legacy Signals dataclass format.
    
    Maintains backward compatibility with existing code that expects
    the original Signals dataclass.
    
    Parameters
    ----------
    evidence : Evidence
        Evidence object with diagnostic inputs
    
    Returns
    -------
    Signals
        Legacy Signals dataclass with all computed values
    
    Examples
    --------
    >>> signals = extract_signals_to_legacy_format(evidence)
    >>> print(signals.cv_mean)
    >>> print(signals.train_val_gap)
    """
    collection = extract_all_signals(evidence)
    return collection.to_legacy_signals()


def compute_score(
    y_true: Any,
    y_pred: Any,
    task: Union[TaskType, str],
    metric: str = "default"
) -> float:
    """
    Compute a score for predictions.
    
    Delegates to performance module for backward compatibility.
    
    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    task : TaskType or str
        Task type (classification or regression)
    metric : str, default="default"
        Metric name
    
    Returns
    -------
    float
        Computed score
    """
    from .performance import compute_score as _compute_score
    
    if isinstance(task, str):
        task = TaskType(task)
    
    return _compute_score(y_true, y_pred, task, metric)


def analyze_cv_stability(cv_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze CV stability.
    
    Delegates to cv_based module for backward compatibility.
    
    Parameters
    ----------
    cv_results : dict
        CV results from cross_validate
    
    Returns
    -------
    dict
        Stability analysis
    """
    from .cv_based import analyze_cv_stability as _analyze
    return _analyze(cv_results)


def get_available_signals(
    category: Optional[Union[SignalCategory, str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Get available signal extractors.
    
    Parameters
    ----------
    category : SignalCategory or str, optional
        Filter by category
    
    Returns
    -------
    dict
        Signal names mapped to their metadata
    """
    from ._registry import get_registered_signals
    return get_registered_signals(category)


def get_available_extractors() -> List[str]:
    """
    Get list of all registered extractor names.
    
    Returns
    -------
    list of str
        Names of all registered extractors
    """
    registry = get_global_registry()
    return registry.list_signals()


def register_builtin_extractors() -> None:
    """Register all built-in signal extractors."""
    from ._registry import register_extractor_class
    from .performance import PerformanceSignalExtractor, ScoreBasedSignalExtractor
    from .cv_based import CVSignalExtractor, CVFoldAnalyzer
    from .distribution import DistributionSignalExtractor, ClassBalanceAnalyzer
    from .feature import FeatureSignalExtractor, FeatureRedundancyAnalyzer
    from .leakage import LeakageSignalExtractor, PreprocessingLeakageDetector
    
    # Performance
    register_extractor_class(
        "performance",
        PerformanceSignalExtractor,
        category=SignalCategory.PERFORMANCE,
        description="Basic performance metrics and train/val gaps"
    )
    register_extractor_class(
        "score_based",
        ScoreBasedSignalExtractor,
        category=SignalCategory.PERFORMANCE,
        description="Score-based diagnostic signals"
    )
    
    # CV
    register_extractor_class(
        "cv_based",
        CVSignalExtractor,
        category=SignalCategory.CV,
        description="Cross-validation stability and performance signals"
    )
    register_extractor_class(
        "cv_fold_analysis",
        CVFoldAnalyzer,
        category=SignalCategory.CV,
        description="Detailed per-fold CV analysis"
    )
    
    # Distribution
    register_extractor_class(
        "distribution",
        DistributionSignalExtractor,
        category=SignalCategory.DISTRIBUTION,
        description="Class and label distribution signals"
    )
    register_extractor_class(
        "class_balance",
        ClassBalanceAnalyzer,
        category=SignalCategory.DISTRIBUTION,
        description="Detailed class imbalance analysis"
    )
    
    # Feature
    register_extractor_class(
        "feature",
        FeatureSignalExtractor,
        category=SignalCategory.FEATURE,
        description="Feature correlation and redundancy signals"
    )
    register_extractor_class(
        "feature_redundancy",
        FeatureRedundancyAnalyzer,
        category=SignalCategory.FEATURE,
        description="Feature redundancy analysis"
    )
    
    # Leakage
    register_extractor_class(
        "leakage",
        LeakageSignalExtractor,
        category=SignalCategory.LEAKAGE,
        description="Data leakage indicator signals"
    )
    register_extractor_class(
        "preprocessing_leakage",
        PreprocessingLeakageDetector,
        category=SignalCategory.LEAKAGE,
        description="Preprocessing-related leakage detection"
    )


# Register built-in extractors on module import
register_builtin_extractors()
