"""
Signal registry for automatic discovery and registration.

This module provides a registry pattern similar to scikit-learn's
scoring mechanism, allowing signals to be registered and retrieved
by name.

Example:
    # Register a custom signal
    @register_signal("my_custom_signal")
    class MyCustomSignal(Signal):
        ...
    
    # Retrieve and use
    signal = get_signal("my_custom_signal")
    result = signal(evidence)
"""

from typing import Callable, Dict, List, Optional, Type, Union, TYPE_CHECKING
from .base import Signal, SignalFunction, SignalCategory, SignalResult


_SIGNAL_REGISTRY: Dict[str, Union[Type[Signal], SignalFunction]] = {}
_CATEGORY_SIGNALS: Dict[SignalCategory, List[str]] = {
    cat: [] for cat in SignalCategory
}


class SignalRegistry:
    """
    A registry class that provides a scikit-learn-like interface for signal management.
    
    This class allows users to:
    - Register custom signals
    - Retrieve signals by name
    - Get all signals in a category
    - Extract all applicable signals from evidence
    
    Example:
        registry = SignalRegistry()
        
        # Register a custom signal
        registry.register("my_signal", MySignalClass)
        
        # Extract all signals from evidence
        results = registry.extract_all(evidence)
    """
    
    def __init__(self):
        self._signals: Dict[str, Union[Type[Signal], SignalFunction]] = {}
        self._by_category: Dict[SignalCategory, List[str]] = {
            cat: [] for cat in SignalCategory
        }
    
    def register(
        self,
        name: str,
        signal: Union[Type[Signal], SignalFunction, Callable],
        category: Optional[SignalCategory] = None,
        description: str = "",
    ) -> None:
        """
        Register a signal.
        
        Args:
            name: Unique name for the signal
            signal: Signal class, SignalFunction, or callable
            category: Optional category (inferred if not provided)
            description: Optional description
        """
        if name in self._signals:
            raise ValueError(f"Signal '{name}' is already registered")
        
        if callable(signal) and not isinstance(signal, (SignalFunction, type)):
            signal = SignalFunction(
                func=signal,
                name=name,
                category=category or SignalCategory.SCORE_BASED,
                description=description,
            )
        
        self._signals[name] = signal
        
        sig_category = category
        if sig_category is None:
            if hasattr(signal, 'category'):
                sig_category = signal.category
            else:
                sig_category = SignalCategory.SCORE_BASED
        
        self._by_category[sig_category].append(name)
    
    def get(self, name: str) -> Optional[Union[Type[Signal], SignalFunction]]:
        """Get a signal by name."""
        return self._signals.get(name)
    
    def get_category(self, category: SignalCategory) -> List[Union[Type[Signal], SignalFunction]]:
        """Get all signals in a category."""
        names = self._by_category.get(category, [])
        return [self._signals[n] for n in names if n in self._signals]
    
    def extract(
        self,
        name: str,
        evidence: "Evidence",
    ) -> Optional[SignalResult]:
        """
        Extract a single signal from evidence.
        
        Args:
            name: Signal name
            evidence: Evidence object
            
        Returns:
            SignalResult or None if signal cannot be extracted
        """
        signal = self.get(name)
        if signal is None:
            return None
        
        if isinstance(signal, SignalFunction):
            if not signal.can_extract(evidence):
                return None
            return signal.extract(evidence)
        elif isinstance(signal, type):
            instance = signal()
            if not instance.can_extract(evidence):
                return None
            return instance.extract(evidence)
        else:
            return signal(evidence)
    
    def extract_all(
        self,
        evidence: "Evidence",
        categories: Optional[List[SignalCategory]] = None,
    ) -> List[SignalResult]:
        """
        Extract all applicable signals from evidence.
        
        Args:
            evidence: Evidence object
            categories: Optional list of categories to limit extraction
            
        Returns:
            List of SignalResult objects
        """
        results = []
        
        if categories is None:
            categories = list(SignalCategory)
        
        for category in categories:
            for name in self._by_category.get(category, []):
                result = self.extract(name, evidence)
                if result is not None:
                    results.append(result)
        
        return results
    
    def list(self) -> Dict[str, Dict[str, any]]:
        """List all registered signals with metadata."""
        result = {}
        for name, obj in self._signals.items():
            if isinstance(obj, SignalFunction):
                result[name] = {
                    "category": obj.category.value,
                    "description": obj.description,
                    "requires_cv": obj.requires_cv,
                    "requires_validation": obj.requires_validation,
                }
            else:
                result[name] = {
                    "category": obj.category.value if hasattr(obj, 'category') else "unknown",
                    "description": obj.description if hasattr(obj, 'description') else "",
                    "requires_cv": obj.requires_cv if hasattr(obj, 'requires_cv') else False,
                    "requires_validation": obj.requires_validation if hasattr(obj, 'requires_validation') else False,
                }
        return result


DEFAULT_REGISTRY = SignalRegistry()


def register_signal(
    name: str,
    category: Optional[SignalCategory] = None,
    registry: Optional[SignalRegistry] = None,
) -> Callable:
    """
    Decorator to register a signal class or function.
    
    Args:
        name: Unique name for the signal
        category: Optional category override (uses class attribute if not provided)
        registry: Optional registry to register to (uses DEFAULT_REGISTRY if not provided)
        
    Returns:
        Decorator function
        
    Example:
        @register_signal("train_val_gap")
        class TrainValGapSignal(Signal):
            name = "train_val_gap"
            category = SignalCategory.SCORE_BASED
            ...
    """
    def decorator(obj: Union[Type[Signal], SignalFunction]) -> Union[Type[Signal], SignalFunction]:
        target_registry = registry if registry is not None else DEFAULT_REGISTRY
        
        if name in _SIGNAL_REGISTRY:
            raise ValueError(f"Signal '{name}' is already registered")
        
        _SIGNAL_REGISTRY[name] = obj
        
        sig_category = category
        if sig_category is None:
            if hasattr(obj, 'category'):
                sig_category = obj.category
            elif isinstance(obj, SignalFunction):
                sig_category = obj.category
            else:
                sig_category = SignalCategory.SCORE_BASED
        
        if sig_category not in _CATEGORY_SIGNALS:
            _CATEGORY_SIGNALS[sig_category] = []
        _CATEGORY_SIGNALS[sig_category].append(name)
        
        description = ""
        if hasattr(obj, 'description'):
            description = obj.description
        elif isinstance(obj, SignalFunction):
            description = obj.description
        
        target_registry.register(
            name=name,
            signal=obj,
            category=sig_category,
            description=description,
        )
        
        return obj
    
    return decorator


def register_signal_function(
    name: str,
    category: SignalCategory,
    description: str = "",
    requires_cv: bool = False,
    requires_validation: bool = False,
) -> Callable:
    """
    Decorator to register a simple function as a signal.
    
    This is a convenience decorator that wraps the function in a SignalFunction.
    
    Args:
        name: Unique name for the signal
        category: Signal category
        description: Human-readable description
        requires_cv: Whether CV results are required
        requires_validation: Whether validation set is required
        
    Returns:
        Decorator function
        
    Example:
        @register_signal_function("train_score", SignalCategory.SCORE_BASED)
        def extract_train_score(evidence):
            return SignalResult(
                name="train_score",
                value=evidence.train_score,
                category=SignalCategory.SCORE_BASED
            )
    """
    def decorator(func: Callable) -> SignalFunction:
        if name in _SIGNAL_REGISTRY:
            raise ValueError(f"Signal '{name}' is already registered")
        
        signal_func = SignalFunction(
            func=func,
            name=name,
            category=category,
            description=description,
            requires_cv=requires_cv,
            requires_validation=requires_validation,
        )
        
        _SIGNAL_REGISTRY[name] = signal_func
        _CATEGORY_SIGNALS[category].append(name)
        
        DEFAULT_REGISTRY.register(
            name=name,
            signal=signal_func,
            category=category,
            description=description,
        )
        
        return signal_func
    
    return decorator


def get_signal(name: str) -> Optional[Union[Type[Signal], SignalFunction]]:
    """
    Retrieve a registered signal by name.
    
    Args:
        name: Signal name
        
    Returns:
        Signal class or function, or None if not found
    """
    return _SIGNAL_REGISTRY.get(name)


def get_signals_by_category(category: SignalCategory) -> List[Union[Type[Signal], SignalFunction]]:
    """
    Get all registered signals in a category.
    
    Args:
        category: Signal category to filter by
        
    Returns:
        List of signal classes/functions in the category
    """
    names = _CATEGORY_SIGNALS.get(category, [])
    return [_SIGNAL_REGISTRY[name] for name in names if name in _SIGNAL_REGISTRY]


def list_signals() -> Dict[str, Dict[str, any]]:
    """
    List all registered signals with their metadata.
    
    Returns:
        Dictionary mapping signal names to their metadata
    """
    result = {}
    for name, obj in _SIGNAL_REGISTRY.items():
        if isinstance(obj, SignalFunction):
            result[name] = {
                "category": obj.category.value,
                "description": obj.description,
                "requires_cv": obj.requires_cv,
                "requires_validation": obj.requires_validation,
            }
        else:
            result[name] = {
                "category": obj.category.value if hasattr(obj, 'category') else "unknown",
                "description": obj.description if hasattr(obj, 'description') else "",
                "requires_cv": obj.requires_cv if hasattr(obj, 'requires_cv') else False,
                "requires_validation": obj.requires_validation if hasattr(obj, 'requires_validation') else False,
            }
    return result


def clear_registry() -> None:
    """Clear all registered signals (useful for testing)."""
    global _SIGNAL_REGISTRY, _CATEGORY_SIGNALS, DEFAULT_REGISTRY
    _SIGNAL_REGISTRY = {}
    _CATEGORY_SIGNALS = {cat: [] for cat in SignalCategory}
    DEFAULT_REGISTRY = SignalRegistry()
