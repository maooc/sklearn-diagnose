"""
Registry utilities for signal extractors.

Provides decorator-based registration similar to sklearn's scoring registry.
"""

from typing import Any, Callable, Dict, List, Optional, Type, Union

from ._base import SignalExtractor, SignalRegistry, SignalCategory, get_global_registry


def register_signal(
    name: Optional[str] = None,
    category: Union[SignalCategory, str] = SignalCategory.CUSTOM,
    description: str = "",
    required_evidence: Optional[List[str]] = None
):
    """
    Decorator to register a signal extractor or function.
    
    This is the primary way to add custom signals to sklearn-diagnose.
    Similar to sklearn.metrics.make_scorer pattern.
    
    Parameters
    ----------
    name : str, optional
        Identifier for this signal. If None, uses function/class name.
    category : SignalCategory or str, default=CUSTOM
        Category for grouping signals (e.g., PERFORMANCE, CV, DISTRIBUTION)
    description : str, default=""
        Human-readable description of the signal
    required_evidence : list of str, optional
        Required evidence attributes (for function-based signals)
    
    Returns
    -------
    decorator
        A decorator that registers the function or class
    
    Examples
    --------
    Register a function-based signal:
    
    >>> @register_signal("my_signal", category=SignalCategory.PERFORMANCE)
    ... def compute_my_signal(evidence):
    ...     return SignalResult(name="my_signal", value=42.0, category=SignalCategory.PERFORMANCE)
    
    Register a class-based extractor:
    
    >>> @register_signal("cv_stability", category=SignalCategory.CV)
    ... class CVStabilityExtractor(SignalExtractor):
    ...     def extract(self, evidence):
    ...         ...
    
    Access registered signals:
    
    >>> signals = get_registered_signals()
    >>> extractor = create_extractor("my_signal")
    """
    def decorator(obj: Union[Type[SignalExtractor], Callable]) -> Any:
        registry = get_global_registry()
        signal_name = name or obj.__name__

        # Normalize category - use different variable name to avoid conflict
        normalized_category = category
        if isinstance(normalized_category, str):
            normalized_category = SignalCategory.from_string(normalized_category)

        metadata = {
            "category": normalized_category,
            "description": description,
        }

        if isinstance(obj, type) and issubclass(obj, SignalExtractor):
            # Register class-based extractor
            registry.register(signal_name, obj, metadata)
            return obj
        else:
            # Register function-based signal
            # Create a wrapper class
            wrapper = _create_function_wrapper(
                obj, signal_name, normalized_category, required_evidence or []
            )
            registry.register(signal_name, wrapper, metadata)
            return obj  # Return original function

    return decorator


def _create_function_wrapper(
    func: Callable,
    name: str,
    category: SignalCategory,
    required_evidence: List[str]
) -> Type[SignalExtractor]:
    """Create a SignalExtractor subclass that wraps a function."""
    
    class FunctionExtractor(SignalExtractor):
        def __init__(self):
            super().__init__(name=name, category=category)
            self._func = func
        
        def get_required_evidence(self) -> List[str]:
            return required_evidence
        
        def extract(self, evidence):
            from ._base import SignalResult
            result = self._func(evidence)
            # Ensure result is SignalResult
            if not isinstance(result, SignalResult):
                result = SignalResult(name=name, value=result, category=category)
            return result
    
    FunctionExtractor.__name__ = f"{name}_extractor"
    FunctionExtractor.__doc__ = func.__doc__
    return FunctionExtractor


def get_registered_signals(
    category: Optional[Union[SignalCategory, str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Get all registered signals with metadata.
    
    Parameters
    ----------
    category : SignalCategory or str, optional
        Filter by category
    
    Returns
    -------
    dict
        Mapping of signal names to their metadata
    """
    registry = get_global_registry()
    signals = {}
    for name in registry.list_signals(category):
        meta = registry.get_metadata(name) or {}
        signals[name] = {
            "name": name,
            "category": meta.get("category", SignalCategory.CUSTOM).value 
                      if isinstance(meta.get("category"), SignalCategory) 
                      else meta.get("category", "custom"),
            "description": meta.get("description", ""),
        }
    return signals


def get_signal(name: str) -> Optional[SignalExtractor]:
    """
    Get a registered signal extractor by name.

    This is an alias for create_extractor() with a more intuitive name.

    Parameters
    ----------
    name : str
        Registered signal name

    Returns
    -------
    SignalExtractor or None
        Extractor instance if found

    Examples
    --------
    >>> extractor = get_signal("cv_stability")
    >>> result = extractor(evidence)
    """
    registry = get_global_registry()
    return registry.get(name)


def create_extractor(name: str) -> Optional[SignalExtractor]:
    """
    Create an extractor instance by name.

    Parameters
    ----------
    name : str
        Registered signal name

    Returns
    -------
    SignalExtractor or None
        Extractor instance if found

    Examples
    --------
    >>> extractor = create_extractor("cv_stability")
    >>> result = extractor(evidence)
    """
    return get_signal(name)


def unregister_signal(name: str) -> bool:
    """
    Unregister a signal.
    
    Parameters
    ----------
    name : str
        Signal name to unregister
    
    Returns
    -------
    bool
        True if signal was found and removed
    """
    registry = get_global_registry()
    if registry.is_registered(name):
        del registry._extractors[name]
        del registry._metadata[name]
        return True
    return False


def register_extractor_class(
    name: str,
    extractor_class: Type[SignalExtractor],
    category: Union[SignalCategory, str] = SignalCategory.CUSTOM,
    description: str = ""
) -> Type[SignalExtractor]:
    """
    Register a signal extractor class directly (non-decorator usage).
    
    Parameters
    ----------
    name : str
        Unique identifier for this extractor
    extractor_class : type
        SignalExtractor subclass to register
    category : SignalCategory or str, default=CUSTOM
        Category for grouping signals
    description : str, default=""
        Human-readable description
    
    Returns
    -------
    type
        The registered class
    """
    registry = get_global_registry()
    
    # Normalize category
    if isinstance(category, str):
        category = SignalCategory.from_string(category)
    
    metadata = {
        "category": category,
        "description": description,
    }
    registry.register(name, extractor_class, metadata)
    return extractor_class


def get_signal_categories() -> List[str]:
    """Get all available signal category names."""
    return [cat.value for cat in SignalCategory]


def get_extractors_by_category(
    category: Union[SignalCategory, str]
) -> Dict[str, SignalExtractor]:
    """
    Get all extractors of a specific category.
    
    Parameters
    ----------
    category : SignalCategory or str
        Category to filter by
    
    Returns
    -------
    dict
        Mapping of extractor names to instances
    """
    registry = get_global_registry()
    return registry.get_by_category(category)
