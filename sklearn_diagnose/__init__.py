"""
sklearn-diagnose: An intelligent diagnosis layer for scikit-learn.

LLM-powered model failure detection with evidence-based analysis.

This library uses LLM-powered analysis for model diagnosis. All hypotheses 
are probabilistic and evidence-based.

Cross-validation interpretation is a core signal extractor within 
sklearn-diagnose, used to detect instability, overfitting, and potential 
data leakage.

Quick Start:
    >>> from sklearn_diagnose import setup_llm, diagnose
    >>> 
    >>> setup_llm(provider="openai", model="gpt-4o", api_key="sk-...")
    >>> 
    >>> report = diagnose(
    ...     estimator=model,
    ...     datasets={
    ...         "train": (X_train, y_train),
    ...         "val": (X_val, y_val)
    ...     },
    ...     task="classification"
    ... )
    >>> 
    >>> print(report.summary())
    >>> print(report.recommendations)

The library follows an LLM-driven architecture:
1. Signal extraction: Compute deterministic statistics from the model
2. LLM hypothesis generation: Detect failure modes with confidence/severity
3. LLM recommendation generation: Generate actionable recommendations
4. LLM summary generation: Create human-readable summaries

Read-Only Guarantee:
    This library NEVER modifies your estimator, never calls fit(),
    and never mutates your input data.

Detected Failure Modes:
    - Overfitting
    - Underfitting
    - High variance
    - Label noise
    - Feature redundancy
    - Class imbalance
    - Data leakage (suspicious patterns)

LLM Setup (required):
    >>> from sklearn_diagnose import setup_llm
    >>> setup_llm(provider="openai", model="gpt-4o", api_key="sk-...")

New Refactored Signal Extraction API:
    >>> from sklearn_diagnose import (
    ...     SignalResult, SignalCategory,
    ...     BaseSignalExtractor, signal_registry,
    ...     extract_signals, list_registered_extractors,
    ...     register_extractor
    ... )
    >>> 
    >>> # List all available signal extractors
    >>> extractors = list_registered_extractors()
    >>> 
    >>> # Extract specific signals
    >>> signals = extract_signals(evidence, ["train_score", "cv_stability"])
    >>> 
    >>> # Register custom extractor
    >>> register_extractor("my_extractor", MyExtractor(), SignalCategory.PERFORMANCE)
"""

__version__ = "0.1.0"

# Main API
from .api import diagnose

# Core types (for advanced users)
from .core import (
    BaseSignalExtractor,
    ConfidenceLevel,
    DiagnosisReport,
    Evidence,
    FailureMode,
    Hypothesis,
    Recommendation,
    SignalCategory,
    SignalExtractorRegistry,
    SignalResult,
    Signals,
    TaskType,
    extract_signals,
    get_extractor,
    list_registered_extractors,
    register_extractor,
    signal_registry,
)

# LLM configuration
from .llm import setup_llm

# Chatbot
from .chatbot import launch_chatbot

__all__ = [
    # Version
    "__version__",
    # Main API
    "diagnose",
    "setup_llm",
    "launch_chatbot",
    # Types
    "DiagnosisReport",
    "Hypothesis",
    "Recommendation",
    "Signals",
    "Evidence",
    "TaskType",
    "FailureMode",
    "ConfidenceLevel",
    # New refactored signal components
    "SignalResult",
    "SignalCategory",
    "BaseSignalExtractor",
    "SignalExtractorRegistry",
    "signal_registry",
    # Signal extraction API
    "extract_signals",
    "list_registered_extractors",
    "get_extractor",
    "register_extractor",
]
