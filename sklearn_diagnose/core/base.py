"""
Base classes and registry for signal extractors.

This module provides the foundation for the signal extraction system,
including the registry mechanism and base classes for signal extractors.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar, Union

from .schemas import Evidence, SignalCategory, SignalResult, TaskType

# Type variables
SignalExtractorType = TypeVar("SignalExtractorType", bound="BaseSignalExtractor")
SignalFunction = Callable[[Evidence, Optional[Dict[str, Any]]], List[SignalResult]]


class SignalExtractorRegistry:
    """
    Registry for signal extractors.
    
    This class manages the registration and retrieval of signal extractors,
    similar to scikit-learn's scoring strategy. It supports both class-based
    and function-based extractors.
    
    Attributes:
        _registry: Dictionary mapping extractor names to extractor instances
        _categories: Dictionary mapping categories to extractor names
    """
    
    def __init__(self):
        self._registry: Dict[str, Union[BaseSignalExtractor, SignalFunction]] = {}
        self._categories: Dict[SignalCategory, List[str]] = defaultdict(list)
        self._tags: Dict[str, Set[str]] = defaultdict(set)
        
    def register(
        self,
        name: str,
        extractor: Union[BaseSignalExtractor, SignalFunction],
        category: SignalCategory,
        tags: Optional[List[str]] = None,
        force: bool = False,
    ) -> None:
        """
        Register a signal extractor.
        
        Args:
            name: Unique name for the extractor
            extractor: The signal extractor (class instance or function)
            category: Category of the signal extractor
            tags: Optional tags for filtering
            force: If True, overwrite existing extractor with the same name
            
        Raises:
            ValueError: If extractor with same name already exists and force=False
        """
        if name in self._registry and not force:
            raise ValueError(
                f"Signal extractor '{name}' already registered. "
                f"Use force=True to overwrite."
            )
        
        self._registry[name] = extractor
        self._categories[category].append(name)
        
        if tags:
            self._tags[name].update(tags)
            
    def register_class(
        self,
        name: Optional[str] = None,
        category: Optional[SignalCategory] = None,
        tags: Optional[List[str]] = None,
        force: bool = False,
    ) -> Callable[[Type[SignalExtractorType]], Type[SignalExtractorType]]:
        """
        Decorator to register a signal extractor class.
        
        Args:
            name: Optional name for the extractor (defaults to class.__name__)
            category: Optional category (defaults to class.category attribute)
            tags: Optional tags for filtering
            force: If True, overwrite existing extractor with the same name
            
        Returns:
            Decorator function that registers the class
            
        Examples:
            >>> @registry.register_class(name="my_extractor", category=SignalCategory.PERFORMANCE)
            ... class MyExtractor(BaseSignalExtractor):
            ...     def extract(self, evidence, params=None):
            ...         return [SignalResult(...)]
        """
        def decorator(cls: Type[SignalExtractorType]) -> Type[SignalExtractorType]:
            extractor_name = name or cls.__name__
            extractor_category = category or getattr(cls, "category", None)
            
            if extractor_category is None:
                raise ValueError(
                    f"Category must be provided either via decorator or "
                    f"as a class attribute for {cls.__name__}"
                )
                
            instance = cls()
            self.register(extractor_name, instance, extractor_category, tags, force)
            return cls
        
        return decorator
    
    def register_function(
        self,
        name: Optional[str] = None,
        category: Optional[SignalCategory] = None,
        tags: Optional[List[str]] = None,
        force: bool = False,
    ) -> Callable[[SignalFunction], SignalFunction]:
        """
        Decorator to register a signal extractor function.
        
        Args:
            name: Optional name for the extractor (defaults to function.__name__)
            category: Category of the signal extractor
            tags: Optional tags for filtering
            force: If True, overwrite existing extractor with the same name
            
        Returns:
            Decorator function that registers the function
            
        Examples:
            >>> @registry.register_function(category=SignalCategory.PERFORMANCE)
            ... def my_extractor(evidence, params=None):
            ...     return [SignalResult(...)]
        """
        def decorator(func: SignalFunction) -> SignalFunction:
            extractor_name = name or func.__name__
            
            if category is None:
                raise ValueError(
                    f"Category must be provided for function {func.__name__}"
                )
                
            self.register(extractor_name, func, category, tags, force)
            return func
        
        return decorator
    
    def get(self, name: str) -> Union[BaseSignalExtractor, SignalFunction]:
        """
        Get a signal extractor by name.
        
        Args:
            name: Name of the extractor to retrieve
            
        Returns:
            The signal extractor (class instance or function)
            
        Raises:
            ValueError: If extractor with given name is not registered
        """
        if name not in self._registry:
            raise ValueError(
                f"Signal extractor '{name}' not found. "
                f"Available extractors: {list(self._registry.keys())}"
            )
        return self._registry[name]
    
    def get_by_category(self, category: SignalCategory) -> List[str]:
        """
        Get all extractor names in a given category.
        
        Args:
            category: Category to filter by
            
        Returns:
            List of extractor names in the category
        """
        return self._categories.get(category, [])
    
    def get_by_tag(self, tag: str) -> List[str]:
        """
        Get all extractor names with a given tag.
        
        Args:
            tag: Tag to filter by
            
        Returns:
            List of extractor names with the tag
        """
        return [name for name, tags in self._tags.items() if tag in tags]
    
    def list_all(self) -> Dict[str, Dict[str, Any]]:
        """
        List all registered extractors with their metadata.
        
        Returns:
            Dictionary mapping extractor names to their metadata
        """
        result = {}
        for name, extractor in self._registry.items():
            if isinstance(extractor, BaseSignalExtractor):
                result[name] = {
                    "type": "class",
                    "category": extractor.category.value if hasattr(extractor, "category") else "unknown",
                    "description": inspect.getdoc(extractor) or "",
                }
            else:
                result[name] = {
                    "type": "function",
                    "description": inspect.getdoc(extractor) or "",
                }
        return result
    
    def extract(
        self,
        evidence: Evidence,
        extractor_names: Optional[List[str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[SignalResult]:
        """
        Extract signals using specified extractors.
        
        Args:
            evidence: Evidence object containing all input data
            extractor_names: Optional list of extractor names to use.
                           If None, use all registered extractors.
            params: Optional parameters to pass to extractors
            
        Returns:
            List of SignalResult objects from all extractors
        """
        if extractor_names is None:
            extractor_names = list(self._registry.keys())
            
        all_results = []
        params = params or {}
        
        for name in extractor_names:
            extractor = self._registry[name]
            
            try:
                if isinstance(extractor, BaseSignalExtractor):
                    results = extractor.extract(evidence, params.get(name))
                else:
                    results = extractor(evidence, params.get(name))
                    
                all_results.extend(results)
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Error in signal extractor '{name}': {e}")
                
        return all_results
    
    def __contains__(self, name: str) -> bool:
        """Check if an extractor is registered."""
        return name in self._registry
    
    def __len__(self) -> int:
        """Get the number of registered extractors."""
        return len(self._registry)


# Global registry instance
signal_registry = SignalExtractorRegistry()


class BaseSignalExtractor(ABC):
    """
    Abstract base class for signal extractors.
    
    This class defines the interface for all signal extractors.
    Subclasses must implement the extract() method.
    
    Attributes:
        category: The category of signals this extractor produces
        supported_tasks: Optional set of supported task types
        tags: Optional tags for filtering
    """
    
    category: SignalCategory
    supported_tasks: Optional[Set[TaskType]] = None
    tags: List[str] = []
    
    @abstractmethod
    def extract(
        self,
        evidence: Evidence,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[SignalResult]:
        """
        Extract signals from the given evidence.
        
        Args:
            evidence: Evidence object containing all input data
            params: Optional parameters for the extractor
            
        Returns:
            List of SignalResult objects
        """
        pass
    
    def supports_task(self, task: TaskType) -> bool:
        """
        Check if this extractor supports a given task type.
        
        Args:
            task: Task type to check
            
        Returns:
            True if the task is supported, False otherwise
        """
        if self.supported_tasks is None:
            return True
        return task in self.supported_tasks


class FunctionSignalExtractor(BaseSignalExtractor):
    """
    Wrapper class to convert a function to a signal extractor class.
    
    This class allows functions to be used as class-based extractors,
    providing consistency with the class-based API.
    """
    
    def __init__(
        self,
        func: SignalFunction,
        category: SignalCategory,
        supported_tasks: Optional[Set[TaskType]] = None,
        tags: Optional[List[str]] = None,
    ):
        self._func = func
        self.category = category
        self.supported_tasks = supported_tasks
        self.tags = tags or []
        
    def extract(
        self,
        evidence: Evidence,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[SignalResult]:
        """Extract signals by calling the wrapped function."""
        return self._func(evidence, params)
