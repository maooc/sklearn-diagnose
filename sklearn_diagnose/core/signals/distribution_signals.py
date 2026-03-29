"""
Distribution-based signal extractors.

This module contains signal extractors related to data distribution,
including class distribution for classification and residual analysis for regression.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from .base import Signal, SignalCategory, SignalResult
from .registry import register_signal

if TYPE_CHECKING:
    from ..schemas import Evidence, TaskType


@register_signal("class_distribution")
class ClassDistributionSignal(Signal):
    """Class distribution in training data (classification only)."""
    
    name = "class_distribution"
    category = SignalCategory.DISTRIBUTION
    description = "Class distribution in training data"
    
    def extract(self, evidence: "Evidence") -> SignalResult:
        from ..schemas import TaskType
        
        if evidence.task != TaskType.CLASSIFICATION:
            return SignalResult(
                name=self.name,
                value=None,
                category=self.category,
                metadata={"reason": "Only applicable to classification tasks"},
            )
        
        unique, counts = np.unique(evidence.y_train, return_counts=True)
        total = len(evidence.y_train)
        distribution = {str(cls): count / total for cls, count in zip(unique, counts)}
        
        return SignalResult(
            name=self.name,
            value=distribution,
            category=self.category,
            metadata={
                "n_classes": len(unique),
                "class_counts": {str(cls): int(count) for cls, count in zip(unique, counts)},
            },
        )


@register_signal("minority_class_ratio")
class MinorityClassRatioSignal(Signal):
    """Ratio of minority class in training data (classification only)."""
    
    name = "minority_class_ratio"
    category = SignalCategory.DISTRIBUTION
    description = "Ratio of minority class in training data"
    
    def extract(self, evidence: "Evidence") -> SignalResult:
        from ..schemas import TaskType
        
        if evidence.task != TaskType.CLASSIFICATION:
            return SignalResult(
                name=self.name,
                value=None,
                category=self.category,
                metadata={"reason": "Only applicable to classification tasks"},
            )
        
        unique, counts = np.unique(evidence.y_train, return_counts=True)
        
        if len(counts) < 2:
            return SignalResult(
                name=self.name,
                value=1.0,
                category=self.category,
                metadata={"reason": "Only one class present"},
            )
        
        ratio = float(np.min(counts) / np.sum(counts))
        
        return SignalResult(
            name=self.name,
            value=ratio,
            category=self.category,
            is_anomaly=ratio < 0.1,
        )


@register_signal("class_imbalance_ratio")
class ClassImbalanceRatioSignal(Signal):
    """Ratio of majority to minority class (classification only)."""
    
    name = "class_imbalance_ratio"
    category = SignalCategory.DISTRIBUTION
    description = "Ratio of majority to minority class"
    
    def extract(self, evidence: "Evidence") -> SignalResult:
        from ..schemas import TaskType
        
        if evidence.task != TaskType.CLASSIFICATION:
            return SignalResult(
                name=self.name,
                value=None,
                category=self.category,
                metadata={"reason": "Only applicable to classification tasks"},
            )
        
        unique, counts = np.unique(evidence.y_train, return_counts=True)
        
        if len(counts) < 2:
            return SignalResult(
                name=self.name,
                value=1.0,
                category=self.category,
                metadata={"reason": "Only one class present"},
            )
        
        ratio = float(np.max(counts) / np.min(counts))
        
        return SignalResult(
            name=self.name,
            value=ratio,
            category=self.category,
            is_anomaly=ratio > 10,
        )


@register_signal("per_class_recall")
class PerClassRecallSignal(Signal):
    """Per-class recall scores (classification only)."""
    
    name = "per_class_recall"
    category = SignalCategory.DISTRIBUTION
    description = "Per-class recall scores on validation set"
    requires_validation = True
    
    def extract(self, evidence: "Evidence") -> SignalResult:
        from ..schemas import TaskType
        
        if evidence.task != TaskType.CLASSIFICATION:
            return SignalResult(
                name=self.name,
                value=None,
                category=self.category,
                metadata={"reason": "Only applicable to classification tasks"},
            )
        
        if evidence.y_pred_val is None:
            return SignalResult(
                name=self.name,
                value=None,
                category=self.category,
                metadata={"reason": "No validation predictions available"},
            )
        
        try:
            unique = np.unique(evidence.y_train)
            recalls = recall_score(
                evidence.y_val,
                evidence.y_pred_val,
                average=None,
                zero_division=0,
            )
            
            recall_dict = {str(cls): float(rec) for cls, rec in zip(unique, recalls)}
            
            min_recall = float(np.min(recalls))
            max_recall = float(np.max(recalls))
            
            return SignalResult(
                name=self.name,
                value=recall_dict,
                category=self.category,
                metadata={
                    "min_recall": min_recall,
                    "max_recall": max_recall,
                    "recall_range": max_recall - min_recall,
                },
                is_anomaly=min_recall < 0.5,
            )
        except Exception as e:
            return SignalResult(
                name=self.name,
                value=None,
                category=self.category,
                metadata={"error": str(e)},
            )


@register_signal("per_class_precision")
class PerClassPrecisionSignal(Signal):
    """Per-class precision scores (classification only)."""
    
    name = "per_class_precision"
    category = SignalCategory.DISTRIBUTION
    description = "Per-class precision scores on validation set"
    requires_validation = True
    
    def extract(self, evidence: "Evidence") -> SignalResult:
        from ..schemas import TaskType
        
        if evidence.task != TaskType.CLASSIFICATION:
            return SignalResult(
                name=self.name,
                value=None,
                category=self.category,
                metadata={"reason": "Only applicable to classification tasks"},
            )
        
        if evidence.y_pred_val is None:
            return SignalResult(
                name=self.name,
                value=None,
                category=self.category,
                metadata={"reason": "No validation predictions available"},
            )
        
        try:
            unique = np.unique(evidence.y_train)
            precisions = precision_score(
                evidence.y_val,
                evidence.y_pred_val,
                average=None,
                zero_division=0,
            )
            
            precision_dict = {str(cls): float(prec) for cls, prec in zip(unique, precisions)}
            
            min_precision = float(np.min(precisions))
            max_precision = float(np.max(precisions))
            
            return SignalResult(
                name=self.name,
                value=precision_dict,
                category=self.category,
                metadata={
                    "min_precision": min_precision,
                    "max_precision": max_precision,
                    "precision_range": max_precision - min_precision,
                },
                is_anomaly=min_precision < 0.5,
            )
        except Exception as e:
            return SignalResult(
                name=self.name,
                value=None,
                category=self.category,
                metadata={"error": str(e)},
            )


@register_signal("confusion_matrix")
class ConfusionMatrixSignal(Signal):
    """Confusion matrix (classification only)."""
    
    name = "confusion_matrix"
    category = SignalCategory.DISTRIBUTION
    description = "Confusion matrix on validation set"
    requires_validation = True
    
    def extract(self, evidence: "Evidence") -> SignalResult:
        from ..schemas import TaskType
        
        if evidence.task != TaskType.CLASSIFICATION:
            return SignalResult(
                name=self.name,
                value=None,
                category=self.category,
                metadata={"reason": "Only applicable to classification tasks"},
            )
        
        if evidence.y_pred_val is None:
            return SignalResult(
                name=self.name,
                value=None,
                category=self.category,
                metadata={"reason": "No validation predictions available"},
            )
        
        try:
            cm = confusion_matrix(evidence.y_val, evidence.y_pred_val)
            
            return SignalResult(
                name=self.name,
                value=cm.tolist(),
                category=self.category,
                metadata={
                    "shape": cm.shape,
                },
            )
        except Exception as e:
            return SignalResult(
                name=self.name,
                value=None,
                category=self.category,
                metadata={"error": str(e)},
            )


@register_signal("residual_mean")
class ResidualMeanSignal(Signal):
    """Mean of residuals (regression only)."""
    
    name = "residual_mean"
    category = SignalCategory.DISTRIBUTION
    description = "Mean of training residuals"
    
    def extract(self, evidence: "Evidence") -> SignalResult:
        from ..schemas import TaskType
        
        if evidence.task != TaskType.REGRESSION:
            return SignalResult(
                name=self.name,
                value=None,
                category=self.category,
                metadata={"reason": "Only applicable to regression tasks"},
            )
        
        if evidence.y_pred_train is None:
            return SignalResult(
                name=self.name,
                value=None,
                category=self.category,
                metadata={"reason": "No training predictions available"},
            )
        
        residuals = evidence.y_train - evidence.y_pred_train
        mean_residual = float(np.mean(residuals))
        
        return SignalResult(
            name=self.name,
            value=mean_residual,
            category=self.category,
            is_anomaly=abs(mean_residual) > 0.1 * np.std(residuals),
        )


@register_signal("residual_std")
class ResidualStdSignal(Signal):
    """Standard deviation of residuals (regression only)."""
    
    name = "residual_std"
    category = SignalCategory.DISTRIBUTION
    description = "Standard deviation of training residuals"
    
    def extract(self, evidence: "Evidence") -> SignalResult:
        from ..schemas import TaskType
        
        if evidence.task != TaskType.REGRESSION:
            return SignalResult(
                name=self.name,
                value=None,
                category=self.category,
                metadata={"reason": "Only applicable to regression tasks"},
            )
        
        if evidence.y_pred_train is None:
            return SignalResult(
                name=self.name,
                value=None,
                category=self.category,
                metadata={"reason": "No training predictions available"},
            )
        
        residuals = evidence.y_train - evidence.y_pred_train
        std_residual = float(np.std(residuals))
        
        return SignalResult(
            name=self.name,
            value=std_residual,
            category=self.category,
        )


@register_signal("residual_skew")
class ResidualSkewSignal(Signal):
    """Skewness of residual distribution (regression only)."""
    
    name = "residual_skew"
    category = SignalCategory.DISTRIBUTION
    description = "Skewness of training residual distribution"
    
    def extract(self, evidence: "Evidence") -> SignalResult:
        from ..schemas import TaskType
        
        if evidence.task != TaskType.REGRESSION:
            return SignalResult(
                name=self.name,
                value=None,
                category=self.category,
                metadata={"reason": "Only applicable to regression tasks"},
            )
        
        if evidence.y_pred_train is None:
            return SignalResult(
                name=self.name,
                value=None,
                category=self.category,
                metadata={"reason": "No training predictions available"},
            )
        
        residuals = evidence.y_train - evidence.y_pred_train
        
        if len(residuals) < 3:
            return SignalResult(
                name=self.name,
                value=None,
                category=self.category,
                metadata={"reason": "Not enough samples for skewness calculation"},
            )
        
        try:
            skew = float(stats.skew(residuals))
            
            return SignalResult(
                name=self.name,
                value=skew,
                category=self.category,
                is_anomaly=abs(skew) > 1.0,
            )
        except Exception as e:
            return SignalResult(
                name=self.name,
                value=None,
                category=self.category,
                metadata={"error": str(e)},
            )


@register_signal("residual_kurtosis")
class ResidualKurtosisSignal(Signal):
    """Kurtosis of residual distribution (regression only)."""
    
    name = "residual_kurtosis"
    category = SignalCategory.DISTRIBUTION
    description = "Kurtosis of training residual distribution"
    
    def extract(self, evidence: "Evidence") -> SignalResult:
        from ..schemas import TaskType
        
        if evidence.task != TaskType.REGRESSION:
            return SignalResult(
                name=self.name,
                value=None,
                category=self.category,
                metadata={"reason": "Only applicable to regression tasks"},
            )
        
        if evidence.y_pred_train is None:
            return SignalResult(
                name=self.name,
                value=None,
                category=self.category,
                metadata={"reason": "No training predictions available"},
            )
        
        residuals = evidence.y_train - evidence.y_pred_train
        
        if len(residuals) < 4:
            return SignalResult(
                name=self.name,
                value=None,
                category=self.category,
                metadata={"reason": "Not enough samples for kurtosis calculation"},
            )
        
        try:
            kurtosis = float(stats.kurtosis(residuals))
            
            return SignalResult(
                name=self.name,
                value=kurtosis,
                category=self.category,
                is_anomaly=kurtosis > 7.0,
            )
        except Exception as e:
            return SignalResult(
                name=self.name,
                value=None,
                category=self.category,
                metadata={"error": str(e)},
            )


DISTRIBUTION_SIGNALS = [
    ClassDistributionSignal,
    MinorityClassRatioSignal,
    ClassImbalanceRatioSignal,
    PerClassRecallSignal,
    PerClassPrecisionSignal,
    ConfusionMatrixSignal,
    ResidualMeanSignal,
    ResidualStdSignal,
    ResidualSkewSignal,
    ResidualKurtosisSignal,
]
