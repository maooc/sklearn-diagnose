"""
Cross-validation based signal extractors.

Signals related to CV stability, fold analysis, and cross-validation
performance patterns.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from ._base import SignalExtractor, SignalResult, SignalCategory
from ..schemas import Evidence


class CVSignalExtractor(SignalExtractor):
    """
    Extractor for cross-validation based signals.
    
    Analyzes CV results to detect instability, overfitting,
    and potential data leakage patterns.
    """
    
    def __init__(self):
        super().__init__("cv_based", category=SignalCategory.CV)
    
    def get_required_evidence(self) -> List[str]:
        return ["cv_results"]
    
    def extract(self, evidence: Evidence) -> List[SignalResult]:
        """Extract CV-based signals."""
        results = []
        cv = evidence.cv_results
        
        if cv is None or "test_score" not in cv:
            return results
        
        test_scores = np.asarray(cv["test_score"])
        n_folds = len(test_scores)
        
        # Basic statistics
        cv_mean = float(np.mean(test_scores))
        cv_std = float(np.std(test_scores))
        cv_min = float(np.min(test_scores))
        cv_max = float(np.max(test_scores))
        cv_range = cv_max - cv_min
        
        results.append(SignalResult(
            name="cv_mean",
            value=cv_mean,
            category=SignalCategory.CV,
            description="Mean cross-validation score",
            metadata={"n_folds": n_folds}
        ))
        
        results.append(SignalResult(
            name="cv_std",
            value=cv_std,
            category=SignalCategory.CV,
            description="Standard deviation of CV scores",
            metadata={"n_folds": n_folds}
        ))
        
        results.append(SignalResult(
            name="cv_range",
            value=cv_range,
            category=SignalCategory.CV,
            description="Range between best and worst CV fold",
            metadata={"min": cv_min, "max": cv_max}
        ))
        
        results.append(SignalResult(
            name="cv_fold_scores",
            value=test_scores.tolist(),
            category=SignalCategory.CV,
            description="Individual fold scores",
            metadata={"n_folds": n_folds}
        ))
        
        # Coefficient of variation (stability metric)
        cv_coef = cv_std / cv_mean if cv_mean != 0 else float('inf')
        results.append(SignalResult(
            name="cv_coefficient_of_variation",
            value=cv_coef,
            category=SignalCategory.CV,
            description="Coefficient of variation (std/mean)",
            metadata={
                "stability": self._interpret_cv(cv_coef),
                "threshold_low": 0.05,
                "threshold_medium": 0.10,
                "threshold_high": 0.20
            }
        ))
        
        # Train scores analysis (if available)
        if "train_score" in cv:
            train_scores = np.asarray(cv["train_score"])
            train_mean = float(np.mean(train_scores))
            
            results.append(SignalResult(
                name="cv_train_mean",
                value=train_mean,
                category=SignalCategory.CV,
                description="Mean training score across CV folds",
                metadata={"n_folds": n_folds}
            ))
            
            # Train-test gap
            train_val_gap = train_mean - cv_mean
            results.append(SignalResult(
                name="cv_train_val_gap",
                value=train_val_gap,
                category=SignalCategory.CV,
                description="Average gap between train and validation scores",
                metadata={
                    "train_mean": train_mean,
                    "val_mean": cv_mean,
                    "interpretation": self._interpret_gap(train_val_gap)
                }
            ))
            
            # Per-fold gaps
            gaps = train_scores - test_scores
            results.append(SignalResult(
                name="cv_fold_gaps",
                value=gaps.tolist(),
                category=SignalCategory.CV,
                description="Train-val gap for each fold",
                metadata={
                    "mean_gap": float(np.mean(gaps)),
                    "max_gap": float(np.max(gaps))
                }
            ))
        
        # Outlier fold detection
        outliers = self._detect_outliers(test_scores)
        if outliers:
            results.append(SignalResult(
                name="cv_outlier_folds",
                value=outliers,
                category=SignalCategory.CV,
                description="Folds with scores significantly different from mean",
                metadata={
                    "n_outliers": len(outliers),
                    "threshold_std": 2.0
                }
            ))
        
        # CV vs holdout comparison (if validation set available)
        if evidence.has_validation_set and evidence.y_pred_val is not None:
            from .performance import compute_score
            holdout_score = compute_score(
                evidence.y_val,
                evidence.y_pred_val,
                evidence.task
            )
            cv_holdout_gap = cv_mean - holdout_score
            results.append(SignalResult(
                name="cv_holdout_gap",
                value=cv_holdout_gap,
                category=SignalCategory.CV,
                description="Difference between CV mean and holdout score",
                metadata={
                    "cv_mean": cv_mean,
                    "holdout_score": holdout_score,
                    "interpretation": self._interpret_cv_holdout_gap(cv_holdout_gap)
                }
            ))
        
        return results
    
    def _interpret_cv(self, cv_coef: float) -> str:
        """Interpret coefficient of variation."""
        if cv_coef < 0.05:
            return "high"
        elif cv_coef < 0.10:
            return "medium"
        elif cv_coef < 0.20:
            return "low"
        else:
            return "very_low"
    
    def _interpret_gap(self, gap: float) -> str:
        """Interpret train-val gap."""
        if gap > 0.1:
            return "significant_overfitting"
        elif gap > 0.05:
            return "moderate_overfitting"
        elif gap < 0:
            return "validation_higher"
        else:
            return "normal"
    
    def _interpret_cv_holdout_gap(self, gap: float) -> str:
        """Interpret CV vs holdout gap."""
        if abs(gap) > 0.1:
            return "potential_leakage_or_distribution_shift"
        elif abs(gap) > 0.05:
            return "moderate_discrepancy"
        else:
            return "consistent"
    
    def _detect_outliers(self, scores: np.ndarray) -> List[Dict[str, Any]]:
        """Detect outlier folds (>2 std from mean)."""
        mean = np.mean(scores)
        std = np.std(scores)
        
        if std == 0:
            return []
        
        outliers = []
        for i, score in enumerate(scores):
            z_score = abs(score - mean) / std
            if z_score > 2.0:
                outliers.append({
                    "fold": i,
                    "score": float(score),
                    "z_score": float(z_score),
                    "deviation": float(score - mean)
                })
        
        return outliers


def analyze_cv_stability(cv_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze cross-validation result stability.
    
    Provides detailed stability analysis for CV interpretation.
    Similar to sklearn's model inspection utilities.
    
    Parameters
    ----------
    cv_results : dict
        Dictionary from sklearn.model_selection.cross_validate()
    
    Returns
    -------
    dict
        Stability analysis with statistics and interpretations
    
    Examples
    --------
    >>> from sklearn.model_selection import cross_validate
    >>> cv_results = cross_validate(model, X, y, cv=5, return_train_score=True)
    >>> analysis = analyze_cv_stability(cv_results)
    >>> print(analysis["stability"])  # "high", "medium", "low", "very_low"
    """
    if "test_score" not in cv_results:
        return {"error": "No test_score in cv_results"}
    
    test_scores = np.asarray(cv_results["test_score"])
    n_folds = len(test_scores)
    
    mean = float(np.mean(test_scores))
    std = float(np.std(test_scores))
    cv_coef = std / mean if mean != 0 else float('inf')
    
    analysis = {
        "n_folds": n_folds,
        "mean": mean,
        "std": std,
        "cv": cv_coef,
        "range": float(np.max(test_scores) - np.min(test_scores)),
        "min_fold": int(np.argmin(test_scores)),
        "max_fold": int(np.argmax(test_scores)),
    }
    
    # Stability assessment
    if cv_coef < 0.05:
        analysis["stability"] = "high"
    elif cv_coef < 0.10:
        analysis["stability"] = "medium"
    elif cv_coef < 0.20:
        analysis["stability"] = "low"
    else:
        analysis["stability"] = "very_low"
    
    # Outlier folds
    outliers = []
    for i, score in enumerate(test_scores):
        if std > 0 and abs(score - mean) > 2 * std:
            outliers.append({
                "fold": i,
                "score": float(score),
                "deviation": float((score - mean) / std)
            })
    analysis["outlier_folds"] = outliers
    
    # Train-test gap analysis
    if "train_score" in cv_results:
        train_scores = np.asarray(cv_results["train_score"])
        gaps = train_scores - test_scores
        analysis["train_test_gaps"] = {
            "mean": float(np.mean(gaps)),
            "std": float(np.std(gaps)),
            "max": float(np.max(gaps)),
            "min": float(np.min(gaps)),
        }
        
        # Overfitting assessment
        mean_gap = np.mean(gaps)
        if mean_gap > 0.1:
            analysis["overfitting_risk"] = "high"
        elif mean_gap > 0.05:
            analysis["overfitting_risk"] = "moderate"
        else:
            analysis["overfitting_risk"] = "low"
    
    return analysis


class CVFoldAnalyzer(SignalExtractor):
    """
    Detailed per-fold analysis extractor.
    
    Provides granular insights into individual fold performance.
    """
    
    def __init__(self):
        super().__init__("cv_fold_analysis", category=SignalCategory.CV)
    
    def get_required_evidence(self) -> List[str]:
        return ["cv_results"]
    
    def extract(self, evidence: Evidence) -> SignalResult:
        """Extract detailed per-fold analysis."""
        cv = evidence.cv_results
        
        if cv is None or "test_score" not in cv:
            return SignalResult.failure(
                name="cv_fold_analysis",
                error="No CV results available",
                category=SignalCategory.CV,
                description="Detailed per-fold analysis"
            )
        
        test_scores = np.asarray(cv["test_score"])
        n_folds = len(test_scores)
        
        fold_analysis = []
        for i in range(n_folds):
            fold_info = {
                "fold": i,
                "test_score": float(test_scores[i]),
            }
            
            if "train_score" in cv:
                fold_info["train_score"] = float(cv["train_score"][i])
                fold_info["gap"] = float(cv["train_score"][i] - test_scores[i])
            
            if "fit_time" in cv:
                fold_info["fit_time"] = float(cv["fit_time"][i])
            
            if "score_time" in cv:
                fold_info["score_time"] = float(cv["score_time"][i])
            
            fold_analysis.append(fold_info)
        
        return SignalResult(
            name="cv_fold_analysis",
            value=fold_analysis,
            category=SignalCategory.CV,
            description="Detailed per-fold performance analysis",
            metadata={"n_folds": n_folds}
        )
