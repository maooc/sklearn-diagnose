"""
Base classes and interfaces for signal extraction.

This module defines the core abstractions for signals, inspired by
scikit-learn's inspection API design patterns.

Key concepts:
- SignalResult: Standardized return type for all signal extractors
- Signal: Abstract base class for signal extractors with registration support
- SignalCategory: Categories for organizing signals
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Callable
import numpy as np


class SignalCategory(str, Enum):
    """Categories for organizing signal extractors."""
    
    SCORE_BASED = "score_based"
    CV_BASED = "cv_based"
    DISTRIBUTION = "distribution"
    FEATURE = "feature"
    LEAKAGE = "leakage"


@dataclass
class SignalResult:
    """
    Standardized return type for all signal extractors.
    
    This provides a consistent interface for downstream consumers,
    replacing the previous mixed dict/list returns.
    
    Attributes:
        name: Unique identifier for this signal
        value: The computed signal value (can be scalar, array, or dict)
        category: Which category this signal belongs to
        metadata: Additional context about the signal computation
        confidence: Optional confidence score for the signal (0.0-1.0)
        is_anomaly: Whether this signal indicates an anomaly/issue
    """
    
    name: str
    value: Any
    category: SignalCategory
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: Optional[float] = None
    is_anomaly: Optional[bool] = None
    
    def __post_init__(self):
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "name": self.name,
            "value": self._serialize_value(self.value),
            "category": self.category.value,
            "metadata": self.metadata,
        }
        if self.confidence is not None:
            result["confidence"] = self.confidence
        if self.is_anomaly is not None:
            result["is_anomaly"] = self.is_anomaly
        return result
    
    @staticmethod
    def _serialize_value(value: Any) -> Any:
        """Serialize value to JSON-compatible format."""
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, dict):
            return {k: SignalResult._serialize_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [SignalResult._serialize_value(v) for v in value]
        return value


class Signal(ABC):
    """
    Abstract base class for signal extractors.
    
    Similar to scikit-learn's transformer interface, signal extractors
    follow a consistent pattern:
    
    1. Define signal metadata (name, category, description)
    2. Implement extract() method for computation
    3. Optionally define anomaly detection thresholds
    
    Example:
        class TrainValGapSignal(Signal):
            name = "train_val_gap"
            category = SignalCategory.SCORE_BASED
            
            def extract(self, evidence) -> SignalResult:
                gap = evidence.train_score - evidence.val_score
                return SignalResult(
                    name=self.name,
                    value=gap,
                    category=self.category,
                    is_anomaly=gap > 0.15
                )
    """
    
    name: str = ""
    category: SignalCategory = SignalCategory.SCORE_BASED
    description: str = ""
    requires_cv: bool = False
    requires_validation: bool = False
    
    @abstractmethod
    def extract(self, evidence: "Evidence") -> SignalResult:
        """
        Extract signal from evidence.
        
        Args:
            evidence: Evidence object containing model data and predictions
            
        Returns:
            SignalResult with computed signal value
        """
        pass
    
    def can_extract(self, evidence: "Evidence") -> bool:
        """
        Check if this signal can be extracted from the given evidence.
        
        Args:
            evidence: Evidence object to check
            
        Returns:
            True if all required data is available
        """
        if self.requires_cv and not evidence.has_cv_results:
            return False
        if self.requires_validation and not evidence.has_validation_set:
            return False
        return True
    
    def __call__(self, evidence: "Evidence") -> Optional[SignalResult]:
        """
        Callable interface for signal extraction.
        
        Returns None if signal cannot be extracted.
        """
        if not self.can_extract(evidence):
            return None
        return self.extract(evidence)


class SignalFunction:
    """
    Wrapper to convert a function into a Signal-like callable.
    
    This allows registering simple functions as signals without
    creating a full class.
    """
    
    def __init__(
        self,
        func: Callable[["Evidence"], SignalResult],
        name: str,
        category: SignalCategory,
        description: str = "",
        requires_cv: bool = False,
        requires_validation: bool = False,
    ):
        self.func = func
        self.name = name
        self.category = category
        self.description = description
        self.requires_cv = requires_cv
        self.requires_validation = requires_validation
    
    def can_extract(self, evidence: "Evidence") -> bool:
        if self.requires_cv and not evidence.has_cv_results:
            return False
        if self.requires_validation and not evidence.has_validation_set:
            return False
        return True
    
    def extract(self, evidence: "Evidence") -> SignalResult:
        return self.func(evidence)
    
    def __call__(self, evidence: "Evidence") -> Optional[SignalResult]:
        if not self.can_extract(evidence):
            return None
        return self.extract(evidence)


def signal(
    name: str,
    category: SignalCategory,
    description: str = "",
    requires_cv: bool = False,
    requires_validation: bool = False,
) -> Callable:
    """
    Decorator to register a function as a signal.
    
    Example:
        @signal("train_val_gap", SignalCategory.SCORE_BASED)
        def extract_train_val_gap(evidence):
            return SignalResult(
                name="train_val_gap",
                value=evidence.train_score - evidence.val_score,
                category=SignalCategory.SCORE_BASED
            )
    """
    def decorator(func: Callable) -> SignalFunction:
        return SignalFunction(
            func=func,
            name=name,
            category=category,
            description=description,
            requires_cv=requires_cv,
            requires_validation=requires_validation,
        )
    return decorator
