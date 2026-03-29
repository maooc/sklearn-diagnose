"""
Leakage detection signal extractors.

This module contains signal extractors that detect potential data leakage,
including suspicious feature-target correlations and CV-holdout discrepancies.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from .base import Signal, SignalCategory, SignalResult
from .registry import register_signal

if TYPE_CHECKING:
    from ..schemas import Evidence


@register_signal("suspicious_feature_correlations")
class SuspiciousFeatureCorrelationsSignal(Signal):
    """
    Features with suspiciously high correlation to target.
    
    A correlation > 0.95 may indicate data leakage (target encoding,
    future information, etc.)
    """
    
    name = "suspicious_feature_correlations"
    category = SignalCategory.LEAKAGE
    description = "Features with suspiciously high target correlation (> 0.95)"
    
    def extract(self, evidence: "Evidence") -> SignalResult:
        X = evidence.X_train
        
        if len(X.shape) != 2:
            return SignalResult(
                name=self.name,
                value=[],
                category=self.category,
                metadata={"reason": "Invalid feature matrix shape"},
            )
        
        try:
            y = evidence.y_train.astype(float)
            suspicious = []
            
            for i in range(X.shape[1]):
                try:
                    corr = np.corrcoef(X[:, i], y)[0, 1]
                    if not np.isnan(corr) and abs(corr) > 0.95:
                        suspicious.append({
                            "feature_index": i,
                            "correlation": float(corr),
                        })
                except Exception:
                    pass
            
            suspicious.sort(key=lambda x: abs(x["correlation"]), reverse=True)
            
            return SignalResult(
                name=self.name,
                value=suspicious,
                category=self.category,
                metadata={
                    "n_suspicious": len(suspicious),
                    "threshold": 0.95,
                },
                is_anomaly=len(suspicious) > 0,
            )
        except Exception as e:
            return SignalResult(
                name=self.name,
                value=[],
                category=self.category,
                metadata={"error": str(e)},
            )


@register_signal("cv_holdout_discrepancy")
class CVHoldoutDiscrepancySignal(Signal):
    """
    Discrepancy between CV score and holdout validation score.
    
    A large discrepancy may indicate:
    - Data leakage in CV (if CV >> holdout)
    - Distribution shift between train and validation
    """
    
    name = "cv_holdout_discrepancy"
    category = SignalCategory.LEAKAGE
    description = "Discrepancy between CV and holdout scores"
    requires_cv = True
    requires_validation = True
    
    def extract(self, evidence: "Evidence") -> SignalResult:
        cv = evidence.cv_results
        
        if cv is None or "test_score" not in cv:
            return SignalResult(
                name=self.name,
                value=None,
                category=self.category,
                metadata={"reason": "No CV results available"},
            )
        
        if evidence.y_pred_val is None:
            return SignalResult(
                name=self.name,
                value=None,
                category=self.category,
                metadata={"reason": "No validation predictions"},
            )
        
        from sklearn.metrics import accuracy_score, r2_score
        from ..schemas import TaskType
        
        test_scores = np.asarray(cv["test_score"])
        cv_mean = float(np.mean(test_scores))
        cv_std = float(np.std(test_scores))
        
        if evidence.task == TaskType.CLASSIFICATION:
            val_score = accuracy_score(evidence.y_val, evidence.y_pred_val)
        else:
            val_score = r2_score(evidence.y_val, evidence.y_pred_val)
        
        discrepancy = cv_mean - val_score
        
        is_anomaly = False
        anomaly_reason = None
        
        if discrepancy > 0.1:
            is_anomaly = True
            anomaly_reason = "CV score much higher than holdout - possible CV leakage"
        elif discrepancy < -0.1:
            is_anomaly = True
            anomaly_reason = "Holdout score much higher than CV - possible distribution shift"
        
        return SignalResult(
            name=self.name,
            value=discrepancy,
            category=self.category,
            metadata={
                "cv_mean": cv_mean,
                "cv_std": cv_std,
                "val_score": val_score,
                "anomaly_reason": anomaly_reason,
            },
            is_anomaly=is_anomaly,
        )


@register_signal("perfect_train_score")
class PerfectTrainScoreSignal(Signal):
    """
    Check for suspiciously perfect training score.
    
    A perfect or near-perfect training score may indicate overfitting
    or data leakage.
    """
    
    name = "perfect_train_score"
    category = SignalCategory.LEAKAGE
    description = "Check for suspiciously perfect training score"
    
    def extract(self, evidence: "Evidence") -> SignalResult:
        if evidence.y_pred_train is None:
            return SignalResult(
                name=self.name,
                value=None,
                category=self.category,
                metadata={"reason": "No training predictions available"},
            )
        
        from sklearn.metrics import accuracy_score, r2_score
        from ..schemas import TaskType
        
        if evidence.task == TaskType.CLASSIFICATION:
            train_score = accuracy_score(evidence.y_train, evidence.y_pred_train)
        else:
            train_score = r2_score(evidence.y_train, evidence.y_pred_train)
        
        is_perfect = train_score >= 0.999
        
        return SignalResult(
            name=self.name,
            value=train_score,
            category=self.category,
            metadata={
                "is_perfect": is_perfect,
                "threshold": 0.999,
            },
            is_anomaly=is_perfect,
        )


@register_signal("train_test_distribution_shift")
class TrainTestDistributionShiftSignal(Signal):
    """
    Detect potential distribution shift between train and validation.
    
    Uses simple statistical tests to compare distributions.
    """
    
    name = "train_test_distribution_shift"
    category = SignalCategory.LEAKAGE
    description = "Detect distribution shift between train and validation"
    requires_validation = True
    
    def extract(self, evidence: "Evidence") -> SignalResult:
        if evidence.X_val is None:
            return SignalResult(
                name=self.name,
                value=None,
                category=self.category,
                metadata={"reason": "No validation features available"},
            )
        
        X_train = evidence.X_train
        X_val = evidence.X_val
        
        if len(X_train.shape) != 2 or len(X_val.shape) != 2:
            return SignalResult(
                name=self.name,
                value=None,
                category=self.category,
                metadata={"reason": "Invalid feature matrix shapes"},
            )
        
        try:
            train_means = np.mean(X_train, axis=0)
            val_means = np.mean(X_val, axis=0)
            train_stds = np.std(X_train, axis=0)
            
            mean_shifts = []
            for i in range(len(train_means)):
                if train_stds[i] > 1e-10:
                    shift = abs(train_means[i] - val_means[i]) / train_stds[i]
                    mean_shifts.append({
                        "feature_index": i,
                        "shift": float(shift),
                    })
            
            mean_shifts.sort(key=lambda x: x["shift"], reverse=True)
            
            max_shift = mean_shifts[0]["shift"] if mean_shifts else 0.0
            
            return SignalResult(
                name=self.name,
                value={
                    "max_shift": max_shift,
                    "feature_shifts": mean_shifts[:10],
                },
                category=self.category,
                metadata={
                    "n_features_checked": len(train_means),
                },
                is_anomaly=max_shift > 2.0,
            )
        except Exception as e:
            return SignalResult(
                name=self.name,
                value=None,
                category=self.category,
                metadata={"error": str(e)},
            )


@register_signal("target_distribution_shift")
class TargetDistributionShiftSignal(Signal):
    """
    Detect distribution shift in target variable.
    
    Compares target distributions between train and validation sets.
    """
    
    name = "target_distribution_shift"
    category = SignalCategory.LEAKAGE
    description = "Detect target distribution shift between train and validation"
    requires_validation = True
    
    def extract(self, evidence: "Evidence") -> SignalResult:
        from ..schemas import TaskType
        
        if evidence.y_val is None:
            return SignalResult(
                name=self.name,
                value=None,
                category=self.category,
                metadata={"reason": "No validation target available"},
            )
        
        y_train = evidence.y_train
        y_val = evidence.y_val
        
        try:
            if evidence.task == TaskType.CLASSIFICATION:
                train_unique, train_counts = np.unique(y_train, return_counts=True)
                val_unique, val_counts = np.unique(y_val, return_counts=True)
                
                train_dist = {str(k): v / len(y_train) for k, v in zip(train_unique, train_counts)}
                val_dist = {str(k): v / len(y_val) for k, v in zip(val_unique, val_counts)}
                
                all_classes = set(train_dist.keys()) | set(val_dist.keys())
                
                distribution_diff = {}
                for cls in all_classes:
                    train_p = train_dist.get(cls, 0.0)
                    val_p = val_dist.get(cls, 0.0)
                    distribution_diff[cls] = abs(train_p - val_p)
                
                max_diff = max(distribution_diff.values()) if distribution_diff else 0.0
                
                return SignalResult(
                    name=self.name,
                    value={
                        "train_distribution": train_dist,
                        "val_distribution": val_dist,
                        "distribution_diff": distribution_diff,
                        "max_diff": max_diff,
                    },
                    category=self.category,
                    is_anomaly=max_diff > 0.1,
                )
            else:
                train_mean = float(np.mean(y_train))
                val_mean = float(np.mean(y_val))
                train_std = float(np.std(y_train))
                
                if train_std > 1e-10:
                    mean_shift = abs(train_mean - val_mean) / train_std
                else:
                    mean_shift = 0.0
                
                return SignalResult(
                    name=self.name,
                    value={
                        "train_mean": train_mean,
                        "val_mean": val_mean,
                        "train_std": train_std,
                        "mean_shift": mean_shift,
                    },
                    category=self.category,
                    is_anomaly=mean_shift > 0.5,
                )
        except Exception as e:
            return SignalResult(
                name=self.name,
                value=None,
                category=self.category,
                metadata={"error": str(e)},
            )


LEAKAGE_SIGNALS = [
    SuspiciousFeatureCorrelationsSignal,
    CVHoldoutDiscrepancySignal,
    PerfectTrainScoreSignal,
    TrainTestDistributionShiftSignal,
    TargetDistributionShiftSignal,
]
