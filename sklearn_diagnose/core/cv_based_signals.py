"""
Cross-validation based signal extractors.

This module contains signal extractors that analyze cross-validation
results, including stability, fold variation, and train-test gaps.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from .base import BaseSignalExtractor, signal_registry
from .schemas import Evidence, SignalCategory, SignalResult


@signal_registry.register_class(
    name="cv_scores",
    category=SignalCategory.CV,
    tags=["cv", "score", "stability"],
)
class CVScoresExtractor(BaseSignalExtractor):
    """
    Extract cross-validation scores and basic statistics.
    
    This extractor computes mean, standard deviation, min, max, and range
    from cross-validation test scores.
    """
    
    category = SignalCategory.CV
    
    def extract(
        self,
        evidence: Evidence,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[SignalResult]:
        """Extract cross-validation score statistics."""
        results = []
        
        if not evidence.has_cv_results:
            return results
            
        cv = evidence.cv_results
        if "test_score" not in cv:
            return results
            
        test_scores = np.asarray(cv["test_score"])
        n_folds = len(test_scores)
        
        # Mean score
        cv_mean = float(np.mean(test_scores))
        results.append(SignalResult(
            name="cv_mean",
            value=cv_mean,
            category=SignalCategory.CV,
            description="Mean cross-validation test score",
            metadata={"n_folds": n_folds},
        ))
        
        # Standard deviation
        cv_std = float(np.std(test_scores))
        results.append(SignalResult(
            name="cv_std",
            value=cv_std,
            category=SignalCategory.CV,
            description="Standard deviation of cross-validation test scores",
            metadata={"n_folds": n_folds, "cv_std": cv_std},
        ))
        
        # Minimum score
        cv_min = float(np.min(test_scores))
        results.append(SignalResult(
            name="cv_min",
            value=cv_min,
            category=SignalCategory.CV,
            description="Minimum cross-validation test score across folds",
            metadata={"n_folds": n_folds, "min_fold": int(np.argmin(test_scores))},
        ))
        
        # Maximum score
        cv_max = float(np.max(test_scores))
        results.append(SignalResult(
            name="cv_max",
            value=cv_max,
            category=SignalCategory.CV,
            description="Maximum cross-validation test score across folds",
            metadata={"n_folds": n_folds, "max_fold": int(np.argmax(test_scores))},
        ))
        
        # Score range
        cv_range = cv_max - cv_min
        results.append(SignalResult(
            name="cv_range",
            value=cv_range,
            category=SignalCategory.CV,
            description="Range (max - min) of cross-validation test scores",
            metadata={"n_folds": n_folds},
        ))
        
        # Individual fold scores
        results.append(SignalResult(
            name="cv_fold_scores",
            value=test_scores.tolist(),
            category=SignalCategory.CV,
            description="Individual cross-validation fold scores",
            metadata={"n_folds": n_folds},
        ))
        
        return results


@signal_registry.register_class(
    name="cv_train_scores",
    category=SignalCategory.CV,
    tags=["cv", "score", "training"],
)
class CVTrainScoresExtractor(BaseSignalExtractor):
    """
    Extract cross-validation training scores and train-test gap.
    
    This extractor analyzes training scores from CV and computes the
    train-test gap which indicates overfitting.
    """
    
    category = SignalCategory.CV
    
    def extract(
        self,
        evidence: Evidence,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[SignalResult]:
        """Extract cross-validation training scores and train-test gap."""
        results = []
        
        if not evidence.has_cv_results:
            return results
            
        cv = evidence.cv_results
        
        # Training scores (if available)
        if "train_score" in cv:
            train_scores = np.asarray(cv["train_score"])
            cv_train_mean = float(np.mean(train_scores))
            
            results.append(SignalResult(
                name="cv_train_mean",
                value=cv_train_mean,
                category=SignalCategory.CV,
                description="Mean cross-validation training score",
                metadata={"n_folds": len(train_scores)},
            ))
            
            # CV train-test gap (overfitting signal)
            if "test_score" in cv:
                test_scores = np.asarray(cv["test_score"])
                cv_test_mean = float(np.mean(test_scores))
                cv_train_val_gap = cv_train_mean - cv_test_mean
                
                results.append(SignalResult(
                    name="cv_train_val_gap",
                    value=cv_train_val_gap,
                    category=SignalCategory.CV,
                    description="Mean train-test gap across CV folds (overfitting indicator)",
                    metadata={
                        "n_folds": len(train_scores),
                        "train_mean": cv_train_mean,
                        "test_mean": cv_test_mean,
                    },
                ))
        
        return results


@signal_registry.register_class(
    name="cv_holdout_gap",
    category=SignalCategory.CV,
    tags=["cv", "gap", "leakage"],
)
class CVHoldoutGapExtractor(BaseSignalExtractor):
    """
    Extract gap between CV mean and holdout validation score.
    
    This extractor computes the difference between cross-validation
    mean score and holdout validation score, which can indicate
    data leakage or concept drift.
    """
    
    category = SignalCategory.CV
    
    def extract(
        self,
        evidence: Evidence,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[SignalResult]:
        """Extract CV-holdout gap."""
        results = []
        
        if not evidence.has_cv_results or not evidence.has_validation_set:
            return results
            
        cv = evidence.cv_results
        if "test_score" not in cv:
            return results
            
        if evidence.y_pred_val is None:
            return results
            
        # Compute mean CV score
        test_scores = np.asarray(cv["test_score"])
        cv_mean = float(np.mean(test_scores))
        
        # Get validation score from existing extractor or compute directly
        # We'll compute it here to be self-contained
        from .score_based_signals import compute_score
        val_score = compute_score(
            evidence.y_val,
            evidence.y_pred_val,
            evidence.task,
            "default"
        )
        
        cv_holdout_gap = cv_mean - val_score
        
        results.append(SignalResult(
            name="cv_holdout_gap",
            value=cv_holdout_gap,
            category=SignalCategory.CV,
            description="Gap between CV mean and holdout validation score (leakage indicator)",
            metadata={
                "cv_mean": cv_mean,
                "val_score": val_score,
                "threshold": 0.1,  # Typical threshold for leakage suspicion
            },
        ))
        
        return results


@signal_registry.register_class(
    name="cv_stability",
    category=SignalCategory.CV,
    tags=["cv", "stability", "variance"],
)
class CVStabilityExtractor(BaseSignalExtractor):
    """
    Analyze cross-validation stability and detect outlier folds.
    
    This extractor provides a detailed stability analysis including:
    - Coefficient of variation (CV)
    - Stability classification (high/medium/low/very_low)
    - Outlier fold detection
    """
    
    category = SignalCategory.CV
    
    def extract(
        self,
        evidence: Evidence,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[SignalResult]:
        """Extract cross-validation stability analysis."""
        results = []
        
        if not evidence.has_cv_results:
            return results
            
        cv = evidence.cv_results
        if "test_score" not in cv:
            return results
            
        test_scores = np.asarray(cv["test_score"])
        n_folds = len(test_scores)
        mean = float(np.mean(test_scores))
        std = float(np.std(test_scores))
        
        # Coefficient of variation
        cv_coeff = float(std / mean) if mean > 1e-10 else None
        
        # Stability assessment
        stability = "unknown"
        if cv_coeff is not None:
            if cv_coeff < 0.05:
                stability = "high"
            elif cv_coeff < 0.10:
                stability = "medium"
            elif cv_coeff < 0.20:
                stability = "low"
            else:
                stability = "very_low"
        
        results.append(SignalResult(
            name="cv_stability",
            value=stability,
            category=SignalCategory.CV,
            description="Cross-validation stability assessment",
            metadata={
                "n_folds": n_folds,
                "mean": mean,
                "std": std,
                "cv_coefficient": cv_coeff,
                "stability_levels": ["high", "medium", "low", "very_low"],
            },
        ))
        
        # Detect outlier folds (more than 2 std from mean)
        outliers = []
        for i, score in enumerate(test_scores):
            if abs(score - mean) > 2 * std:
                outliers.append({
                    "fold": i,
                    "score": float(score),
                    "deviation": float((score - mean) / std) if std > 0 else 0
                })
        
        if outliers:
            results.append(SignalResult(
                name="cv_outlier_folds",
                value=outliers,
                category=SignalCategory.CV,
                description="Outlier folds detected in cross-validation (>= 2 std from mean)",
                metadata={"n_folds": n_folds, "n_outliers": len(outliers)},
            ))
        
        return results


# Utility function (can be used independently)
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
    
    # Stability assessment
    cv = analysis["cv"]
    if cv is not None:
        if cv < 0.05:
            analysis["stability"] = "high"
        elif cv < 0.10:
            analysis["stability"] = "medium"
        elif cv < 0.20:
            analysis["stability"] = "low"
        else:
            analysis["stability"] = "very_low"
    
    # Detect outlier folds (more than 2 std from mean)
    mean = analysis["mean"]
    std = analysis["std"]
    outliers = []
    for i, score in enumerate(test_scores):
        if abs(score - mean) > 2 * std:
            outliers.append({
                "fold": i,
                "score": float(score),
                "deviation": float((score - mean) / std) if std > 0 else 0
            })
    analysis["outlier_folds"] = outliers
    
    # Train-test gap analysis if train scores available
    if "train_score" in cv_results:
        train_scores = np.asarray(cv_results["train_score"])
        gaps = train_scores - test_scores
        analysis["train_test_gaps"] = {
            "mean": float(np.mean(gaps)),
            "std": float(np.std(gaps)),
            "max": float(np.max(gaps)),
        }
    
    return analysis
