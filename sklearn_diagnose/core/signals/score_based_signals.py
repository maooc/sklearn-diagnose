"""
Score-based signal extractors.

This module contains signal extractors related to model performance scores,
including train/validation scores and their differences.
"""

from typing import TYPE_CHECKING

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from .base import Signal, SignalCategory, SignalResult
from .registry import register_signal

if TYPE_CHECKING:
    from ..schemas import Evidence, TaskType


@register_signal("train_score")
class TrainScoreSignal(Signal):
    """
    Training set performance score.
    
    For classification: accuracy
    For regression: R² score
    """
    
    name = "train_score"
    category = SignalCategory.SCORE_BASED
    description = "Training set performance score"
    
    def extract(self, evidence: "Evidence") -> SignalResult:
        if evidence.y_pred_train is None:
            return SignalResult(
                name=self.name,
                value=None,
                category=self.category,
                metadata={"reason": "No training predictions available"},
            )
        
        from ..schemas import TaskType
        
        if evidence.task == TaskType.CLASSIFICATION:
            score = accuracy_score(evidence.y_train, evidence.y_pred_train)
        else:
            score = r2_score(evidence.y_train, evidence.y_pred_train)
        
        return SignalResult(
            name=self.name,
            value=score,
            category=self.category,
            is_anomaly=score < 0.5,
        )


@register_signal("val_score")
class ValScoreSignal(Signal):
    """
    Validation set performance score.
    
    For classification: accuracy
    For regression: R² score
    """
    
    name = "val_score"
    category = SignalCategory.SCORE_BASED
    description = "Validation set performance score"
    requires_validation = True
    
    def extract(self, evidence: "Evidence") -> SignalResult:
        if evidence.y_pred_val is None:
            return SignalResult(
                name=self.name,
                value=None,
                category=self.category,
                metadata={"reason": "No validation predictions available"},
            )
        
        from ..schemas import TaskType
        
        if evidence.task == TaskType.CLASSIFICATION:
            score = accuracy_score(evidence.y_val, evidence.y_pred_val)
        else:
            score = r2_score(evidence.y_val, evidence.y_pred_val)
        
        return SignalResult(
            name=self.name,
            value=score,
            category=self.category,
            is_anomaly=score < 0.5,
        )


@register_signal("train_val_gap")
class TrainValGapSignal(Signal):
    """
    Gap between training and validation scores.
    
    A large gap may indicate overfitting.
    """
    
    name = "train_val_gap"
    category = SignalCategory.SCORE_BASED
    description = "Difference between training and validation scores"
    requires_validation = True
    
    def extract(self, evidence: "Evidence") -> SignalResult:
        if evidence.y_pred_train is None or evidence.y_pred_val is None:
            return SignalResult(
                name=self.name,
                value=None,
                category=self.category,
                metadata={"reason": "Missing predictions"},
            )
        
        from ..schemas import TaskType
        
        if evidence.task == TaskType.CLASSIFICATION:
            train_score = accuracy_score(evidence.y_train, evidence.y_pred_train)
            val_score = accuracy_score(evidence.y_val, evidence.y_pred_val)
        else:
            train_score = r2_score(evidence.y_train, evidence.y_pred_train)
            val_score = r2_score(evidence.y_val, evidence.y_pred_val)
        
        gap = train_score - val_score
        
        return SignalResult(
            name=self.name,
            value=gap,
            category=self.category,
            metadata={
                "train_score": train_score,
                "val_score": val_score,
            },
            is_anomaly=gap > 0.15,
        )


@register_signal("relative_gap")
class RelativeGapSignal(Signal):
    """
    Relative gap as a percentage of training score.
    
    This normalizes the gap to account for different score scales.
    """
    
    name = "relative_gap"
    category = SignalCategory.SCORE_BASED
    description = "Relative gap as percentage of training score"
    requires_validation = True
    
    def extract(self, evidence: "Evidence") -> SignalResult:
        if evidence.y_pred_train is None or evidence.y_pred_val is None:
            return SignalResult(
                name=self.name,
                value=None,
                category=self.category,
                metadata={"reason": "Missing predictions"},
            )
        
        from ..schemas import TaskType
        
        if evidence.task == TaskType.CLASSIFICATION:
            train_score = accuracy_score(evidence.y_train, evidence.y_pred_train)
            val_score = accuracy_score(evidence.y_val, evidence.y_pred_val)
        else:
            train_score = r2_score(evidence.y_train, evidence.y_pred_train)
            val_score = r2_score(evidence.y_val, evidence.y_pred_val)
        
        if train_score == 0:
            relative_gap = float('inf') if val_score != 0 else 0.0
        else:
            relative_gap = (train_score - val_score) / abs(train_score)
        
        return SignalResult(
            name=self.name,
            value=relative_gap,
            category=self.category,
            metadata={
                "train_score": train_score,
                "val_score": val_score,
            },
            is_anomaly=relative_gap > 0.2,
        )


def compute_score(
    y_true,
    y_pred,
    task: "TaskType",
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
    from ..schemas import TaskType
    
    if task == TaskType.CLASSIFICATION:
        if metric == "default" or metric == "accuracy":
            return accuracy_score(y_true, y_pred)
        elif metric == "balanced_accuracy":
            return balanced_accuracy_score(y_true, y_pred)
        elif metric == "f1":
            return f1_score(y_true, y_pred, average="weighted")
        else:
            return accuracy_score(y_true, y_pred)
    
    else:
        if metric == "default" or metric == "r2":
            return r2_score(y_true, y_pred)
        elif metric == "mse":
            return -mean_squared_error(y_true, y_pred)
        elif metric == "mae":
            return -mean_absolute_error(y_true, y_pred)
        else:
            return r2_score(y_true, y_pred)


SCORE_BASED_SIGNALS = [
    TrainScoreSignal,
    ValScoreSignal,
    TrainValGapSignal,
    RelativeGapSignal,
]
