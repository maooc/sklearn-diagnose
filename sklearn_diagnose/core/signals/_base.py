"""
Base classes and data structures for signal extraction.

Follows scikit-learn design patterns with:
- Standardized result containers (similar to sklearn's Bunch)
- Abstract base classes for extensibility
- Clear separation between interface and implementation
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np


class SignalCategory(str, Enum):
    """
    Standardized signal categories.
    
    Similar to sklearn's metric categories, provides a type-safe
    classification system for organizing signals.
    """
    PERFORMANCE = "performance"
    CV = "cv"
    DISTRIBUTION = "distribution"
    FEATURE = "feature"
    LEAKAGE = "leakage"
    CUSTOM = "custom"
    
    @classmethod
    def from_string(cls, value: str) -> "SignalCategory":
        """Get category from string, defaulting to CUSTOM if unknown."""
        try:
            return cls(value.lower())
        except ValueError:
            return cls.CUSTOM


@dataclass
class SignalResult:
    """
    Standardized container for signal computation results.
    
    Similar to sklearn.utils.Bunch but with explicit schema for signal outputs.
    All signal extractors must return SignalResult instances.
    
    Parameters
    ----------
    name : str
        Identifier for this signal (e.g., "train_val_gap", "cv_stability")
    value : Any
        Primary computed value (scalar, array, or dict)
    category : SignalCategory or str
        Category classification for this signal
    description : str, optional
        Human-readable description of what this signal measures
    metadata : dict, optional
        Additional context (e.g., thresholds used, computation parameters)
    confidence : float, optional
        Confidence in the signal value (0.0 to 1.0), for probabilistic signals
    
    Attributes
    ----------
    is_valid : bool
        Whether the signal computation succeeded
    error : str, optional
        Error message if computation failed
    
    Examples
    --------
    >>> result = SignalResult(
    ...     name="cv_stability",
    ...     value=0.05,
    ...     category=SignalCategory.CV,
    ...     description="Coefficient of variation across CV folds",
    ...     metadata={"n_folds": 5, "threshold": 0.1}
    ... )
    >>> result.to_dict()
    {'name': 'cv_stability', 'value': 0.05, 'category': 'cv', ...}
    """
    
    name: str
    value: Any
    category: Union[SignalCategory, str] = SignalCategory.CUSTOM
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: Optional[float] = None
    
    # Internal state for error handling
    _error: Optional[str] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Normalize category to SignalCategory enum."""
        if isinstance(self.category, str):
            self.category = SignalCategory.from_string(self.category)
    
    @classmethod
    def failure(
        cls, 
        name: str, 
        error: str, 
        category: Union[SignalCategory, str] = SignalCategory.CUSTOM,
        description: str = ""
    ) -> "SignalResult":
        """Create a SignalResult indicating failed computation."""
        return cls(
            name=name,
            value=None,
            category=category,
            description=description,
            _error=error
        )
    
    @property
    def is_valid(self) -> bool:
        """Check if signal computation succeeded."""
        return self._error is None
    
    @property
    def error(self) -> Optional[str]:
        """Get error message if computation failed."""
        return self._error
    
    @property
    def category_name(self) -> str:
        """Get category as string."""
        return self.category.value if isinstance(self.category, SignalCategory) else str(self.category)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "name": self.name,
            "value": self._serialize_value(self.value),
            "category": self.category_name,
            "description": self.description,
            "metadata": self.metadata,
            "is_valid": self.is_valid,
        }
        if self.confidence is not None:
            result["confidence"] = self.confidence
        if self._error:
            result["error"] = self._error
        return result
    
    def _serialize_value(self, value: Any) -> Any:
        """Serialize value for JSON compatibility."""
        if value is None:
            return None
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            return float(value)
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]
        return value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from metadata or return default (dict-like interface)."""
        return self.metadata.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        """Allow dict-style access to metadata."""
        return self.metadata[key]
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in metadata."""
        return key in self.metadata


@dataclass
class SignalsCollection:
    """
    Container for multiple SignalResult objects.
    
    Similar to sklearn's Bunch or a structured DataFrame, provides
    unified access to all extracted signals with category-based grouping.
    
    Examples
    --------
    >>> collection = SignalsCollection([signal1, signal2, signal3])
    >>> collection.by_category(SignalCategory.CV)
    [SignalResult(...), SignalResult(...)]
    >>> collection["cv_mean"].value
    0.85
    >>> collection.to_dict()
    {"cv_mean": {...}, "train_score": {...}}
    """
    
    _signals: Dict[str, SignalResult] = field(default_factory=dict)
    
    def __init__(self, signals: Optional[List[SignalResult]] = None):
        """Initialize with list of SignalResult objects."""
        self._signals = {}
        if signals:
            for signal in signals:
                self.add(signal)
    
    def add(self, signal: SignalResult) -> None:
        """Add a signal to the collection."""
        self._signals[signal.name] = signal
    
    def get(self, name: str, default: Any = None) -> Optional[SignalResult]:
        """Get signal by name."""
        return self._signals.get(name, default)
    
    def __getitem__(self, name: str) -> SignalResult:
        """Get signal by name using bracket notation."""
        if name not in self._signals:
            raise KeyError(f"Signal '{name}' not found")
        return self._signals[name]
    
    def __contains__(self, name: str) -> bool:
        """Check if signal exists."""
        return name in self._signals
    
    def __iter__(self):
        """Iterate over signal names."""
        return iter(self._signals)
    
    def __len__(self) -> int:
        """Number of signals."""
        return len(self._signals)
    
    def keys(self):
        """Get all signal names."""
        return self._signals.keys()
    
    def values(self):
        """Get all SignalResult objects."""
        return self._signals.values()
    
    def items(self):
        """Get (name, SignalResult) pairs."""
        return self._signals.items()
    
    def by_category(self, category: Union[SignalCategory, str]) -> List[SignalResult]:
        """Get all signals of a specific category."""
        cat_str = category.value if isinstance(category, SignalCategory) else category
        return [
            sig for sig in self._signals.values()
            if sig.category_name == cat_str
        ]
    
    def categories(self) -> List[str]:
        """Get all unique category names."""
        return list(set(sig.category_name for sig in self._signals.values()))
    
    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Convert all signals to dictionary (full SignalResult format)."""
        return {name: sig.to_dict() for name, sig in self._signals.items()}
    
    def to_flat_dict(self) -> Dict[str, Any]:
        """
        Convert to flat dictionary with signal values only.
        
        Returns a simple mapping of signal names to their values,
        compatible with legacy code that expects the old format.
        
        Returns
        -------
        dict
            Mapping of signal names to their values
        """
        result = {}
        for name, sig in self._signals.items():
            result[name] = sig.value
        return result
    
    def to_legacy_signals(self) -> Any:
        """Convert to legacy Signals dataclass for backward compatibility."""
        from ..schemas import Signals
        
        signals = Signals()
        
        # Map SignalResult values to legacy Signals attributes
        for name, result in self._signals.items():
            if hasattr(signals, name):
                setattr(signals, name, result.value)
            # Handle special cases
            elif name == "cv_range" and result.metadata:
                signals.cv_min = result.metadata.get("min")
                signals.cv_max = result.metadata.get("max")
                signals.cv_range = result.value
        
        return signals


class SignalExtractor(ABC):
    """
    Abstract base class for signal extractors.
    
    Similar to sklearn's BaseEstimator pattern, signal extractors define
    a common interface for computing diagnostic signals from evidence.
    
    Subclasses must implement:
    - `extract()`: Main computation method
    - `get_required_evidence()`: Declare what evidence is needed
    
    Parameters
    ----------
    name : str
        Unique identifier for this extractor
    category : SignalCategory
        Category classification for signals from this extractor
    
    Examples
    --------
    >>> class MyExtractor(SignalExtractor):
    ...     def __init__(self):
    ...         super().__init__("my_extractor", SignalCategory.PERFORMANCE)
    ...     
    ...     def get_required_evidence(self):
    ...         return ["X_train", "y_train"]
    ...     
    ...     def extract(self, evidence):
    ...         value = compute_something(evidence.X_train)
    ...         return SignalResult(
    ...             name="my_signal", 
    ...             value=value,
    ...             category=self.category
    ...         )
    """
    
    def __init__(
        self, 
        name: Optional[str] = None,
        category: Union[SignalCategory, str] = SignalCategory.CUSTOM
    ):
        self.name = name or self.__class__.__name__
        self.category = (
            category if isinstance(category, SignalCategory)
            else SignalCategory.from_string(category)
        )
    
    @abstractmethod
    def get_required_evidence(self) -> List[str]:
        """
        Return list of required evidence attributes.
        
        Returns
        -------
        list of str
            Evidence attribute names needed by this extractor
        """
        pass
    
    @abstractmethod
    def extract(self, evidence: Any) -> Union[SignalResult, List[SignalResult]]:
        """
        Extract signals from evidence.
        
        Parameters
        ----------
        evidence : Evidence
            Evidence object containing diagnostic inputs
        
        Returns
        -------
        SignalResult or list of SignalResult
            Computed signal(s). Single signals can be returned directly,
            multiple signals should be returned as a list.
        """
        pass
    
    def can_extract(self, evidence: Any) -> bool:
        """
        Check if required evidence is available.
        
        Parameters
        ----------
        evidence : Evidence
            Evidence object to check
        
        Returns
        -------
        bool
            True if all required evidence is present
        """
        for attr in self.get_required_evidence():
            if not hasattr(evidence, attr):
                return False
            value = getattr(evidence, attr)
            if value is None:
                return False
        return True
    
    def __call__(self, evidence: Any) -> Union[SignalResult, List[SignalResult]]:
        """Allow extractor to be called directly."""
        if not self.can_extract(evidence):
            missing = [
                attr for attr in self.get_required_evidence()
                if not hasattr(evidence, attr) or getattr(evidence, attr) is None
            ]
            return SignalResult.failure(
                name=self.name,
                error=f"Missing required evidence: {missing}",
                category=self.category,
                description=f"Extractor {self.name} cannot run"
            )
        try:
            return self.extract(evidence)
        except Exception as e:
            return SignalResult.failure(
                name=self.name,
                error=str(e),
                category=self.category,
                description=f"Error in {self.name}"
            )


class SignalRegistry:
    """
    Registry for signal extractors.
    
    Similar to sklearn's metric registry pattern. Allows:
    - Registration of custom signal extractors
    - Discovery of available signals
    - Factory-style creation of extractors
    
    Examples
    --------
    >>> registry = SignalRegistry()
    >>> registry.register("cv_stability", CVSignalExtractor)
    >>> extractor = registry.get("cv_stability")
    >>> result = extractor(evidence)
    """
    
    def __init__(self):
        self._extractors: Dict[str, Type[SignalExtractor]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
    
    def register(
        self, 
        name: str, 
        extractor_class: Type[SignalExtractor],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Type[SignalExtractor]:
        """
        Register a signal extractor class.
        
        Parameters
        ----------
        name : str
            Unique identifier for this extractor
        extractor_class : type
            SignalExtractor subclass to register
        metadata : dict, optional
            Additional metadata (category, description, etc.)
        
        Returns
        -------
        type
            The registered class (for use as decorator)
        """
        self._extractors[name] = extractor_class
        self._metadata[name] = metadata or {}
        return extractor_class
    
    def get(self, name: str) -> Optional[SignalExtractor]:
        """
        Get an extractor instance by name.
        
        Parameters
        ----------
        name : str
            Registered name of the extractor
        
        Returns
        -------
        SignalExtractor or None
            Instance of the extractor, or None if not found
        """
        if name not in self._extractors:
            return None
        return self._extractors[name]()
    
    def list_signals(
        self, 
        category: Optional[Union[SignalCategory, str]] = None
    ) -> List[str]:
        """
        List registered signal names.
        
        Parameters
        ----------
        category : SignalCategory or str, optional
            Filter by category
        
        Returns
        -------
        list of str
            Names of registered signals
        """
        if category is None:
            return list(self._extractors.keys())
        
        cat_str = category.value if isinstance(category, SignalCategory) else category
        return [
            name for name, meta in self._metadata.items()
            if meta.get("category") == cat_str or meta.get("category") == SignalCategory(cat_str)
        ]
    
    def get_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a registered signal."""
        return self._metadata.get(name)
    
    def is_registered(self, name: str) -> bool:
        """Check if a signal is registered."""
        return name in self._extractors
    
    def get_by_category(
        self, 
        category: Union[SignalCategory, str]
    ) -> Dict[str, SignalExtractor]:
        """Get all extractors of a specific category."""
        names = self.list_signals(category)
        return {name: self.get(name) for name in names if self.get(name) is not None}


# Global registry instance
_GLOBAL_REGISTRY = SignalRegistry()


def get_global_registry() -> SignalRegistry:
    """Get the global signal registry."""
    return _GLOBAL_REGISTRY
