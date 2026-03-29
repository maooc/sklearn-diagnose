"""
Performance-based signal extractors.

Signals related to model performance metrics, train/val gaps,
and score-based diagnostics.
"""

from typing import List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from ._base import SignalExtractor, SignalResult, SignalCategory
from ..schemas import Evidence, TaskType


class PerformanceSignalExtractor(SignalExtractor):
    """
    Extractor for basic performance metrics and train/val gaps.
    
    Computes training score, validation score, and the gap between them
    as indicators of overfitting/underfitting.
    """
    
    def __init__(self):
        super().__init__("performance", category=SignalCategory.PERFORMANCE)
    
    def get_required_evidence(self) -> List[str]:
        return ["y_train", "y_pred_train", "task"]
    
    def extract(self, evidence: Evidence) -> List[SignalResult]:
        """Extract performance signals."""
        results = []
        
        # Training score
        if evidence.y_pred_train is not None:
            train_score = self._compute_score(
                evidence.y_train, 
                evidence.y_pred_train, 
                evidence.task
            )
            results.append(SignalResult(
                name="train_score",
                value=train_score,
                category=SignalCategory.PERFORMANCE,
                description="Performance score on training set",
                metadata={"metric": self._get_default_metric(evidence.task)}
            ))
        
        # Validation score
        if evidence.has_validation_set and evidence.y_pred_val is not None:
            val_score = self._compute_score(
                evidence.y_val,
                evidence.y_pred_val,
                evidence.task
            )
            results.append(SignalResult(
                name="val_score",
                value=val_score,
                category=SignalCategory.PERFORMANCE,
                description="Performance score on validation set",
                metadata={"metric": self._get_default_metric(evidence.task)}
            ))
            
            # Train-val gap
            if evidence.y_pred_train is not None:
                train_score = results[0].value
                gap = train_score - val_score
                results.append(SignalResult(
                    name="train_val_gap",
                    value=gap,
                    category=SignalCategory.PERFORMANCE,
                    description="Difference between train and validation scores",
                    metadata={
                        "train_score": train_score,
                        "val_score": val_score,
                        "interpretation": self._interpret_gap(gap, evidence.task)
                    }
                ))
        
        return results
    
    def _compute_score(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        task: TaskType
    ) -> float:
        """Compute appropriate score based on task type."""
        if task == TaskType.CLASSIFICATION:
            return float(accuracy_score(y_true, y_pred))
        else:
            return float(r2_score(y_true, y_pred))
    
    def _get_default_metric(self, task: TaskType) -> str:
        """Get default metric name for task."""
        return "accuracy" if task == TaskType.CLASSIFICATION else "r2"
    
    def _interpret_gap(self, gap: float, task: TaskType) -> str:
        """Provide interpretation of train-val gap."""
        if task == TaskType.CLASSIFICATION:
            if gap > 0.1:
                return "significant_overfitting"
            elif gap > 0.05:
                return "moderate_overfitting"
            elif gap < 0:
                return "validation_higher_than_train"
            else:
                return "normal"
        else:  # Regression
            if gap > 0.2:
                return "significant_overfitting"
            elif gap > 0.1:
                return "moderate_overfitting"
            elif gap < 0:
                return "validation_higher_than_train"
            else:
                return "normal"


class ScoreBasedSignalExtractor(SignalExtractor):
    """
    Extractor for score-based diagnostic signals.
    
    Analyzes absolute score values to detect underfitting
    and performance quality.
    """
    
    def __init__(self):
        super().__init__("score_based", category=SignalCategory.PERFORMANCE)
    
    def get_required_evidence(self) -> List[str]:
        return ["y_train", "task"]
    
    def extract(self, evidence: Evidence) -> List[SignalResult]:
        """Extract score-based diagnostic signals."""
        results = []
        
        # Baseline comparison (random classifier/regressor performance)
        baseline = self._compute_baseline(evidence)
        results.append(SignalResult(
            name="baseline_score",
            value=baseline,
            category=SignalCategory.PERFORMANCE,
            description="Expected score from random predictions",
            metadata={"task": evidence.task.value}
        ))
        
        # If we have actual scores, compare to baseline
        if evidence.y_pred_train is not None:
            train_score = self._compute_actual_score(evidence, "train")
            baseline_ratio = (
                train_score / baseline if baseline != 0 else float('inf')
            )
            results.append(SignalResult(
                name="baseline_ratio",
                value=baseline_ratio,
                category=SignalCategory.PERFORMANCE,
                description="Ratio of actual score to baseline score",
                metadata={
                    "actual_score": train_score,
                    "baseline_score": baseline,
                    "interpretation": self._interpret_baseline_ratio(baseline_ratio)
                }
            ))
        
        return results
    
    def _compute_baseline(self, evidence: Evidence) -> float:
        """Compute baseline (random) performance."""
        if evidence.task == TaskType.CLASSIFICATION:
            # For classification, baseline is majority class frequency
            unique, counts = np.unique(evidence.y_train, return_counts=True)
            return float(np.max(counts) / len(evidence.y_train))
        else:
            # For regression, baseline R² is 0 (predict mean)
            return 0.0
    
    def _compute_actual_score(
        self, 
        evidence: Evidence, 
        split: str = "train"
    ) -> float:
        """Compute actual score for given split."""
        if split == "train":
            y_true = evidence.y_train
            y_pred = evidence.y_pred_train
        else:
            y_true = evidence.y_val
            y_pred = evidence.y_pred_val
        
        if evidence.task == TaskType.CLASSIFICATION:
            return float(accuracy_score(y_true, y_pred))
        else:
            return float(r2_score(y_true, y_pred))
    
    def _interpret_baseline_ratio(self, ratio: float) -> str:
        """Interpret the baseline ratio."""
        if ratio < 1.1:
            return "near_baseline_underfitting"
        elif ratio < 2.0:
            return "moderate_improvement"
        else:
            return "strong_improvement"


def compute_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: TaskType,
    metric: str = "default"
) -> float:
    """
    Compute a single score for the given predictions.
    
    Factory function for computing various metrics.
    Similar to sklearn.metrics.get_scorer pattern.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    task : TaskType
        Classification or regression
    metric : str, default="default"
        Metric name or "default" for task-appropriate metric
    
    Returns
    -------
    float
        Computed score
    
    Examples
    --------
    >>> score = compute_score(y_true, y_pred, TaskType.CLASSIFICATION, "f1")
    >>> score = compute_score(y_true, y_pred, TaskType.REGRESSION, "mse")
    """
    if task == TaskType.CLASSIFICATION:
        if metric == "default" or metric == "accuracy":
            return accuracy_score(y_true, y_pred)
        elif metric == "balanced_accuracy":
            return balanced_accuracy_score(y_true, y_pred)
        elif metric == "f1":
            return f1_score(y_true, y_pred, average="weighted", zero_division=0)
        elif metric == "f1_macro":
            return f1_score(y_true, y_pred, average="macro", zero_division=0)
        elif metric == "f1_micro":
            return f1_score(y_true, y_pred, average="micro", zero_division=0)
        else:
            return accuracy_score(y_true, y_pred)
    
    else:  # Regression
        if metric == "default" or metric == "r2":
            return r2_score(y_true, y_pred)
        elif metric == "mse":
            return -mean_squared_error(y_true, y_pred)  # Negative for consistency
        elif metric == "rmse":
            return -np.sqrt(mean_squared_error(y_true, y_pred))
        elif metric == "mae":
            return -mean_absolute_error(y_true, y_pred)
        else:
            return r2_score(y_true, y_pred)


def get_available_metrics(task: TaskType) -> List[str]:
    """
    Get list of available metrics for a task.
    
    Parameters
    ----------
    task : TaskType
        Task type
    
    Returns
    -------
    list of str
        Available metric names
    """
    if task == TaskType.CLASSIFICATION:
        return ["accuracy", "balanced_accuracy", "f1", "f1_macro", "f1_micro"]
    else:
        return ["r2", "mse", "rmse", "mae"]
