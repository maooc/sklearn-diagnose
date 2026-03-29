"""
Cross-validation based signal extractors.

This module contains signal extractors related to cross-validation results,
including fold analysis, stability metrics, and train-test gap analysis.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from .base import Signal, SignalCategory, SignalResult
from .registry import register_signal

if TYPE_CHECKING:
    from ..schemas import Evidence


@register_signal("cv_mean")
class CVMeanSignal(Signal):
    """Mean cross-validation test score."""
    
    name = "cv_mean"
    category = SignalCategory.CV_BASED
    description = "Mean cross-validation test score"
    requires_cv = True
    
    def extract(self, evidence: "Evidence") -> SignalResult:
        cv = evidence.cv_results
        if cv is None or "test_score" not in cv:
            return SignalResult(
                name=self.name,
                value=None,
                category=self.category,
                metadata={"reason": "No CV test scores available"},
            )
        
        test_scores = np.asarray(cv["test_score"])
        mean_score = float(np.mean(test_scores))
        
        return SignalResult(
            name=self.name,
            value=mean_score,
            category=self.category,
            is_anomaly=mean_score < 0.5,
        )


@register_signal("cv_std")
class CVStdSignal(Signal):
    """Standard deviation of cross-validation test scores."""
    
    name = "cv_std"
    category = SignalCategory.CV_BASED
    description = "Standard deviation of CV test scores"
    requires_cv = True
    
    def extract(self, evidence: "Evidence") -> SignalResult:
        cv = evidence.cv_results
        if cv is None or "test_score" not in cv:
            return SignalResult(
                name=self.name,
                value=None,
                category=self.category,
                metadata={"reason": "No CV test scores available"},
            )
        
        test_scores = np.asarray(cv["test_score"])
        std_score = float(np.std(test_scores))
        
        return SignalResult(
            name=self.name,
            value=std_score,
            category=self.category,
            is_anomaly=std_score > 0.1,
        )


@register_signal("cv_range")
class CVRangeSignal(Signal):
    """Range (max - min) of cross-validation test scores."""
    
    name = "cv_range"
    category = SignalCategory.CV_BASED
    description = "Range of CV test scores"
    requires_cv = True
    
    def extract(self, evidence: "Evidence") -> SignalResult:
        cv = evidence.cv_results
        if cv is None or "test_score" not in cv:
            return SignalResult(
                name=self.name,
                value=None,
                category=self.category,
                metadata={"reason": "No CV test scores available"},
            )
        
        test_scores = np.asarray(cv["test_score"])
        score_range = float(np.max(test_scores) - np.min(test_scores))
        
        return SignalResult(
            name=self.name,
            value=score_range,
            category=self.category,
            metadata={
                "min": float(np.min(test_scores)),
                "max": float(np.max(test_scores)),
            },
            is_anomaly=score_range > 0.15,
        )


@register_signal("cv_fold_scores")
class CVFoldScoresSignal(Signal):
    """Individual fold scores from cross-validation."""
    
    name = "cv_fold_scores"
    category = SignalCategory.CV_BASED
    description = "Individual CV fold scores"
    requires_cv = True
    
    def extract(self, evidence: "Evidence") -> SignalResult:
        cv = evidence.cv_results
        if cv is None or "test_score" not in cv:
            return SignalResult(
                name=self.name,
                value=None,
                category=self.category,
                metadata={"reason": "No CV test scores available"},
            )
        
        test_scores = np.asarray(cv["test_score"])
        
        return SignalResult(
            name=self.name,
            value=test_scores.tolist(),
            category=self.category,
            metadata={"n_folds": len(test_scores)},
        )


@register_signal("cv_train_test_gap")
class CVTrainTestGapSignal(Signal):
    """Gap between CV train and test scores (overfitting indicator)."""
    
    name = "cv_train_test_gap"
    category = SignalCategory.CV_BASED
    description = "Gap between CV train and test scores"
    requires_cv = True
    
    def extract(self, evidence: "Evidence") -> SignalResult:
        cv = evidence.cv_results
        if cv is None:
            return SignalResult(
                name=self.name,
                value=None,
                category=self.category,
                metadata={"reason": "No CV results available"},
            )
        
        if "test_score" not in cv or "train_score" not in cv:
            return SignalResult(
                name=self.name,
                value=None,
                category=self.category,
                metadata={"reason": "Missing train or test scores"},
            )
        
        test_scores = np.asarray(cv["test_score"])
        train_scores = np.asarray(cv["train_score"])
        
        test_mean = float(np.mean(test_scores))
        train_mean = float(np.mean(train_scores))
        gap = train_mean - test_mean
        
        return SignalResult(
            name=self.name,
            value=gap,
            category=self.category,
            metadata={
                "train_mean": train_mean,
                "test_mean": test_mean,
            },
            is_anomaly=gap > 0.15,
        )


@register_signal("cv_holdout_gap")
class CVHoldoutGapSignal(Signal):
    """Gap between CV mean and holdout validation score (potential leakage indicator)."""
    
    name = "cv_holdout_gap"
    category = SignalCategory.CV_BASED
    description = "Gap between CV mean and holdout validation score"
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
        
        if evidence.task == TaskType.CLASSIFICATION:
            val_score = accuracy_score(evidence.y_val, evidence.y_pred_val)
        else:
            val_score = r2_score(evidence.y_val, evidence.y_pred_val)
        
        gap = cv_mean - val_score
        
        return SignalResult(
            name=self.name,
            value=gap,
            category=self.category,
            metadata={
                "cv_mean": cv_mean,
                "val_score": val_score,
            },
            is_anomaly=abs(gap) > 0.1,
        )


@register_signal("cv_stability")
class CVStabilitySignal(Signal):
    """
    Cross-validation stability analysis.
    
    Provides coefficient of variation and stability assessment.
    """
    
    name = "cv_stability"
    category = SignalCategory.CV_BASED
    description = "CV stability analysis including coefficient of variation"
    requires_cv = True
    
    def extract(self, evidence: "Evidence") -> SignalResult:
        cv = evidence.cv_results
        if cv is None or "test_score" not in cv:
            return SignalResult(
                name=self.name,
                value=None,
                category=self.category,
                metadata={"reason": "No CV results available"},
            )
        
        test_scores = np.asarray(cv["test_score"])
        mean_score = float(np.mean(test_scores))
        std_score = float(np.std(test_scores))
        
        if mean_score > 0:
            cv_coef = std_score / mean_score
        else:
            cv_coef = None
        
        if cv_coef is not None:
            if cv_coef < 0.05:
                stability = "high"
            elif cv_coef < 0.10:
                stability = "medium"
            elif cv_coef < 0.20:
                stability = "low"
            else:
                stability = "very_low"
        else:
            stability = "unknown"
        
        return SignalResult(
            name=self.name,
            value={
                "coefficient_of_variation": cv_coef,
                "stability": stability,
                "n_folds": len(test_scores),
            },
            category=self.category,
            metadata={
                "mean": mean_score,
                "std": std_score,
            },
            is_anomaly=stability in ("low", "very_low"),
        )


@register_signal("cv_outlier_folds")
class CVOutlierFoldsSignal(Signal):
    """Identify outlier folds that deviate significantly from mean."""
    
    name = "cv_outlier_folds"
    category = SignalCategory.CV_BASED
    description = "Identify CV folds with outlier performance"
    requires_cv = True
    
    def extract(self, evidence: "Evidence") -> SignalResult:
        cv = evidence.cv_results
        if cv is None or "test_score" not in cv:
            return SignalResult(
                name=self.name,
                value=None,
                category=self.category,
                metadata={"reason": "No CV results available"},
            )
        
        test_scores = np.asarray(cv["test_score"])
        mean_score = float(np.mean(test_scores))
        std_score = float(np.std(test_scores))
        
        outliers = []
        if std_score > 0:
            for i, score in enumerate(test_scores):
                deviation = (score - mean_score) / std_score
                if abs(deviation) > 2:
                    outliers.append({
                        "fold": int(i),
                        "score": float(score),
                        "deviation": float(deviation),
                    })
        
        return SignalResult(
            name=self.name,
            value=outliers,
            category=self.category,
            metadata={
                "n_outliers": len(outliers),
                "threshold": "2 std deviations",
            },
            is_anomaly=len(outliers) > 0,
        )


def analyze_cv_stability(cv_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze cross-validation result stability.
    
    This provides additional detail for CV interpretation.
    
    Args:
        cv_results: Dictionary from cross_validate()
        
    Returns:
        Dictionary with stability analysis
    """
    if "test_score" not in cv_results:
        return {"error": "No test_score in cv_results"}
    
    test_scores = np.asarray(cv_results["test_score"])
    n_folds = len(test_scores)
    
    analysis = {
        "n_folds": n_folds,
        "mean": float(np.mean(test_scores)),
        "std": float(np.std(test_scores)),
        "cv": float(np.std(test_scores) / np.mean(test_scores)) if np.mean(test_scores) > 0 else None,
        "range": float(np.max(test_scores) - np.min(test_scores)),
        "min_fold": int(np.argmin(test_scores)),
        "max_fold": int(np.argmax(test_scores)),
    }
    
    cv_coef = analysis["cv"]
    if cv_coef is not None:
        if cv_coef < 0.05:
            analysis["stability"] = "high"
        elif cv_coef < 0.10:
            analysis["stability"] = "medium"
        elif cv_coef < 0.20:
            analysis["stability"] = "low"
        else:
            analysis["stability"] = "very_low"
    
    mean = analysis["mean"]
    std = analysis["std"]
    outliers = []
    for i, score in enumerate(test_scores):
        if std > 0 and abs(score - mean) > 2 * std:
            outliers.append({
                "fold": i,
                "score": float(score),
                "deviation": float((score - mean) / std),
            })
    analysis["outlier_folds"] = outliers
    
    if "train_score" in cv_results:
        train_scores = np.asarray(cv_results["train_score"])
        gaps = train_scores - test_scores
        analysis["train_test_gaps"] = {
            "mean": float(np.mean(gaps)),
            "std": float(np.std(gaps)),
            "max": float(np.max(gaps)),
        }
    
    return analysis


CV_BASED_SIGNALS = [
    CVMeanSignal,
    CVStdSignal,
    CVRangeSignal,
    CVFoldScoresSignal,
    CVTrainTestGapSignal,
    CVHoldoutGapSignal,
    CVStabilitySignal,
    CVOutlierFoldsSignal,
]
