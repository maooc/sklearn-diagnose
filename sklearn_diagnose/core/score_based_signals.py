"""
Score-based signal extractors.

This module contains signal extractors that compute performance scores
and related metrics like train/validation gaps.
"""

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from .base import BaseSignalExtractor, signal_registry
from .schemas import Evidence, SignalCategory, SignalResult, TaskType


@signal_registry.register_class(
    name="train_score",
    category=SignalCategory.PERFORMANCE,
    tags=["score", "training"],
)
class TrainScoreExtractor(BaseSignalExtractor):
    """
    Extract training score from the evidence.
    
    This extractor computes the model's performance on the training set
    using appropriate metrics for the task type.
    """
    
    category = SignalCategory.PERFORMANCE
    
    def extract(
        self,
        evidence: Evidence,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[SignalResult]:
        """Extract training score."""
        results = []
        
        if evidence.y_pred_train is None:
            return results
            
        params = params or {}
        metric = params.get("metric", "default")
        
        score = self._compute_score(
            evidence.y_train,
            evidence.y_pred_train,
            evidence.task,
            metric
        )
        
        results.append(SignalResult(
            name="train_score",
            value=score,
            category=SignalCategory.PERFORMANCE,
            description=f"Training {metric} score",
            metadata={"metric": metric, "task": evidence.task.value},
        ))
        
        return results
        
    def _compute_score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        task: TaskType,
        metric: str = "default",
    ) -> float:
        """Compute score based on task type and metric."""
        if task == TaskType.CLASSIFICATION:
            if metric == "default" or metric == "accuracy":
                return accuracy_score(y_true, y_pred)
            elif metric == "balanced_accuracy":
                return balanced_accuracy_score(y_true, y_pred)
            elif metric == "f1":
                return f1_score(y_true, y_pred, average="weighted")
            else:
                return accuracy_score(y_true, y_pred)
        else:  # Regression
            if metric == "default" or metric == "r2":
                return r2_score(y_true, y_pred)
            elif metric == "mse":
                return -mean_squared_error(y_true, y_pred)  # Negative for consistency
            elif metric == "mae":
                return -mean_absolute_error(y_true, y_pred)
            else:
                return r2_score(y_true, y_pred)


@signal_registry.register_class(
    name="val_score",
    category=SignalCategory.PERFORMANCE,
    tags=["score", "validation"],
)
class ValScoreExtractor(BaseSignalExtractor):
    """
    Extract validation score from the evidence.
    
    This extractor computes the model's performance on the validation set
    using appropriate metrics for the task type.
    """
    
    category = SignalCategory.PERFORMANCE
    
    def extract(
        self,
        evidence: Evidence,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[SignalResult]:
        """Extract validation score."""
        results = []
        
        if not evidence.has_validation_set or evidence.y_pred_val is None:
            return results
            
        params = params or {}
        metric = params.get("metric", "default")
        
        score = self._compute_score(
            evidence.y_val,
            evidence.y_pred_val,
            evidence.task,
            metric
        )
        
        results.append(SignalResult(
            name="val_score",
            value=score,
            category=SignalCategory.PERFORMANCE,
            description=f"Validation {metric} score",
            metadata={"metric": metric, "task": evidence.task.value},
        ))
        
        return results
        
    def _compute_score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        task: TaskType,
        metric: str = "default",
    ) -> float:
        """Compute score based on task type and metric."""
        if task == TaskType.CLASSIFICATION:
            if metric == "default" or metric == "accuracy":
                return accuracy_score(y_true, y_pred)
            elif metric == "balanced_accuracy":
                return balanced_accuracy_score(y_true, y_pred)
            elif metric == "f1":
                return f1_score(y_true, y_pred, average="weighted")
            else:
                return accuracy_score(y_true, y_pred)
        else:  # Regression
            if metric == "default" or metric == "r2":
                return r2_score(y_true, y_pred)
            elif metric == "mse":
                return -mean_squared_error(y_true, y_pred)  # Negative for consistency
            elif metric == "mae":
                return -mean_absolute_error(y_true, y_pred)
            else:
                return r2_score(y_true, y_pred)


@signal_registry.register_class(
    name="train_val_gap",
    category=SignalCategory.PERFORMANCE,
    tags=["score", "gap", "overfitting"],
)
class TrainValGapExtractor(BaseSignalExtractor):
    """
    Extract train-validation gap from the evidence.
    
    This extractor computes the difference between training and validation
    scores, which is a key indicator of overfitting.
    """
    
    category = SignalCategory.PERFORMANCE
    
    def extract(
        self,
        evidence: Evidence,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[SignalResult]:
        """Extract train-validation gap."""
        results = []
        
        if evidence.y_pred_train is None:
            return results
            
        if not evidence.has_validation_set or evidence.y_pred_val is None:
            return results
            
        params = params or {}
        metric = params.get("metric", "default")
        
        train_score = self._compute_score(
            evidence.y_train,
            evidence.y_pred_train,
            evidence.task,
            metric
        )
        
        val_score = self._compute_score(
            evidence.y_val,
            evidence.y_pred_val,
            evidence.task,
            metric
        )
        
        gap = train_score - val_score
        
        results.append(SignalResult(
            name="train_val_gap",
            value=gap,
            category=SignalCategory.PERFORMANCE,
            description="Train-validation score gap (overfitting indicator)",
            metadata={
                "metric": metric,
                "train_score": train_score,
                "val_score": val_score,
                "threshold": 0.05,  # Typical threshold for concern
            },
        ))
        
        return results
        
    def _compute_score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        task: TaskType,
        metric: str = "default",
    ) -> float:
        """Compute score based on task type and metric."""
        if task == TaskType.CLASSIFICATION:
            if metric == "default" or metric == "accuracy":
                return accuracy_score(y_true, y_pred)
            elif metric == "balanced_accuracy":
                return balanced_accuracy_score(y_true, y_pred)
            elif metric == "f1":
                return f1_score(y_true, y_pred, average="weighted")
            else:
                return accuracy_score(y_true, y_pred)
        else:  # Regression
            if metric == "default" or metric == "r2":
                return r2_score(y_true, y_pred)
            elif metric == "mse":
                return -mean_squared_error(y_true, y_pred)  # Negative for consistency
            elif metric == "mae":
                return -mean_absolute_error(y_true, y_pred)
            else:
                return r2_score(y_true, y_pred)


# Utility function (can be used independently)
def compute_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: TaskType,
    metric: str = "default",
) -> float:
    """
    Compute a single score for the given predictions.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        task: Classification or regression
        metric: Metric name or "default"
        
    Returns:
        Computed score
    """
    if task == TaskType.CLASSIFICATION:
        if metric == "default" or metric == "accuracy":
            return accuracy_score(y_true, y_pred)
        elif metric == "balanced_accuracy":
            return balanced_accuracy_score(y_true, y_pred)
        elif metric == "f1":
            return f1_score(y_true, y_pred, average="weighted")
        else:
            return accuracy_score(y_true, y_pred)
    else:  # Regression
        if metric == "default" or metric == "r2":
            return r2_score(y_true, y_pred)
        elif metric == "mse":
            return -mean_squared_error(y_true, y_pred)  # Negative for consistency
        elif metric == "mae":
            return -mean_absolute_error(y_true, y_pred)
        else:
            return r2_score(y_true, y_pred)
