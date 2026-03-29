"""
Core module for sklearn-diagnose.

This module provides the foundational components:
- schemas: Data structures and type definitions
- evidence: Evidence collection and validation
- signals: Deterministic signal extraction (modular, extensible)
- hypotheses: Reference rule-based hypothesis generation (LLM is primary)
- recommendations: Example recommendation templates for LLM guidance
"""

from .evidence import (
    collect_evidence,
    get_estimator_type,
    is_pipeline,
    validate_cv_results,
    validate_datasets,
    validate_estimator,
)
from .hypotheses import generate_hypotheses
from .recommendations import (
    get_example_recommendations_for_failure_mode,
    get_all_failure_modes_with_examples,
    get_insufficient_evidence_message,
    RECOMMENDATION_TEMPLATES,
)
from .schemas import (
    ConfidenceLevel,
    DiagnosisReport,
    Evidence,
    FailureMode,
    Hypothesis,
    Recommendation,
    Signals,
    TaskType,
    ValidationResult,
)
from .signals import (
    extract_signals,
    extract_all_signals,
    extract_signals_by_category,
    get_available_signals,
    compute_score,
    analyze_cv_stability,
    Signal,
    SignalResult,
    SignalCategory,
    SignalRegistry,
    SignalFunction,
    DEFAULT_REGISTRY,
    register_signal,
    register_signal_function,
    get_signal,
    get_signals_by_category,
    list_signals,
    signal,
    ALL_SIGNAL_CLASSES,
)

__all__ = [
    "TaskType",
    "FailureMode",
    "ConfidenceLevel",
    "Evidence",
    "Signals",
    "Hypothesis",
    "Recommendation",
    "DiagnosisReport",
    "ValidationResult",
    "validate_estimator",
    "validate_datasets",
    "validate_cv_results",
    "collect_evidence",
    "get_estimator_type",
    "is_pipeline",
    "extract_signals",
    "extract_all_signals",
    "extract_signals_by_category",
    "get_available_signals",
    "compute_score",
    "analyze_cv_stability",
    "generate_hypotheses",
    "get_example_recommendations_for_failure_mode",
    "get_all_failure_modes_with_examples",
    "get_insufficient_evidence_message",
    "RECOMMENDATION_TEMPLATES",
    "Signal",
    "SignalResult",
    "SignalCategory",
    "SignalRegistry",
    "SignalFunction",
    "DEFAULT_REGISTRY",
    "register_signal",
    "register_signal_function",
    "get_signal",
    "get_signals_by_category",
    "list_signals",
    "signal",
    "ALL_SIGNAL_CLASSES",
]
