"""
Deterministic signal extraction for sklearn-diagnose.

This module computes quantitative statistics from the evidence.
All computations are deterministic and reproducible.

Signal extractors are organized by category:
- Performance signals (train/val scores, gaps)
- CV signals (mean, std, fold analysis)
- Residual signals (for regression)
- Classification signals (class distribution, per-class metrics)
- Feature signals (correlations, redundancy)
- Leakage signals (suspicious patterns)
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from .schemas import Evidence, Signals, TaskType


def extract_all_signals(evidence: Evidence) -> Signals:
    """
    Extract all signals from the provided evidence.
    
    This is the main entry point for signal extraction.
    
    Args:
        evidence: Evidence object with all diagnostic inputs
        
    Returns:
        Signals object with all computed statistics
    """
    signals = Signals()
    
    # Basic data characteristics
    signals.n_samples_train = evidence.n_samples_train
    signals.n_samples_val = evidence.n_samples_val
    signals.n_features = evidence.n_features
    
    if evidence.n_samples_train > 0 and evidence.n_features > 0:
        signals.feature_to_sample_ratio = evidence.n_features / evidence.n_samples_train
    
    # Extract performance signals
    _extract_performance_signals(evidence, signals)
    
    # Extract CV signals if available
    if evidence.has_cv_results:
        _extract_cv_signals(evidence, signals)
    
    # Extract task-specific signals
    if evidence.task == TaskType.CLASSIFICATION:
        _extract_classification_signals(evidence, signals)
    else:
        _extract_regression_signals(evidence, signals)
    
    # Extract feature signals
    _extract_feature_signals(evidence, signals)
    
    # Extract leakage indicators
    _extract_leakage_signals(evidence, signals)
    
    return signals


def _extract_performance_signals(evidence: Evidence, signals: Signals) -> None:
    """Extract basic performance metrics."""
    
    if evidence.task == TaskType.CLASSIFICATION:
        # Training score
        if evidence.y_pred_train is not None:
            signals.train_score = accuracy_score(evidence.y_train, evidence.y_pred_train)
        
        # Validation score
        if evidence.has_validation_set and evidence.y_pred_val is not None:
            signals.val_score = accuracy_score(evidence.y_val, evidence.y_pred_val)
    
    else:  # Regression
        # Training score (R²)
        if evidence.y_pred_train is not None:
            signals.train_score = r2_score(evidence.y_train, evidence.y_pred_train)
        
        # Validation score
        if evidence.has_validation_set and evidence.y_pred_val is not None:
            signals.val_score = r2_score(evidence.y_val, evidence.y_pred_val)
    
    # Train-val gap
    if signals.train_score is not None and signals.val_score is not None:
        signals.train_val_gap = signals.train_score - signals.val_score


def _extract_cv_signals(evidence: Evidence, signals: Signals) -> None:
    """
    Extract cross-validation signals.
    
    CV interpretation is a core signal extractor within sklearn-diagnose,
    used to detect instability, overfitting, and potential data leakage.
    """
    cv = evidence.cv_results
    
    # Test scores
    if "test_score" in cv:
        test_scores = np.asarray(cv["test_score"])
        signals.cv_fold_scores = test_scores.tolist()
        signals.cv_mean = float(np.mean(test_scores))
        signals.cv_std = float(np.std(test_scores))
        signals.cv_min = float(np.min(test_scores))
        signals.cv_max = float(np.max(test_scores))
        signals.cv_range = signals.cv_max - signals.cv_min
    
    # Train scores (if available)
    if "train_score" in cv:
        train_scores = np.asarray(cv["train_score"])
        signals.cv_train_mean = float(np.mean(train_scores))
        
        # CV train-test gap (overfitting signal)
        if signals.cv_mean is not None:
            signals.cv_train_val_gap = signals.cv_train_mean - signals.cv_mean
    
    # CV vs holdout comparison (leakage signal)
    if signals.cv_mean is not None and signals.val_score is not None:
        signals.cv_holdout_gap = signals.cv_mean - signals.val_score


def _extract_classification_signals(evidence: Evidence, signals: Signals) -> None:
    """Extract classification-specific signals."""
    
    # Class distribution
    unique, counts = np.unique(evidence.y_train, return_counts=True)
    total = len(evidence.y_train)
    signals.class_distribution = {
        str(cls): count / total 
        for cls, count in zip(unique, counts)
    }
    
    # Minority class ratio
    if len(counts) > 1:
        signals.minority_class_ratio = float(np.min(counts) / total)
    
    # Confusion matrix and per-class metrics (if predictions available)
    if evidence.y_pred_val is not None and evidence.y_val is not None:
        try:
            signals.confusion_matrix = confusion_matrix(evidence.y_val, evidence.y_pred_val)
            
            # Per-class recall
            recalls = recall_score(
                evidence.y_val, evidence.y_pred_val, 
                average=None, zero_division=0
            )
            signals.per_class_recall = {
                str(cls): float(rec) 
                for cls, rec in zip(unique, recalls)
            }
            
            # Per-class precision
            precisions = precision_score(
                evidence.y_val, evidence.y_pred_val,
                average=None, zero_division=0
            )
            signals.per_class_precision = {
                str(cls): float(prec)
                for cls, prec in zip(unique, precisions)
            }
        except Exception:
            pass  # Handle edge cases gracefully
    
    # Extract probability prediction signals
    _extract_probability_signals(evidence, signals, unique)


def _extract_probability_signals(
    evidence: Evidence, 
    signals: Signals, 
    unique_classes: np.ndarray
) -> None:
    """
    Extract probability prediction diagnostics for classification.
    
    This function analyzes the quality and characteristics of probability
    predictions, including:
    - Confidence levels and calibration
    - Class separation ability
    - Threshold optimization
    - Brier score for probability quality
    
    Args:
        evidence: Evidence object with probability predictions
        signals: Signals object to populate
        unique_classes: Array of unique class labels
    """
    if evidence.y_proba_val is None or evidence.y_val is None:
        signals.has_probability_predictions = False
        return
    
    signals.has_probability_predictions = True
    y_proba = evidence.y_proba_val
    y_true = evidence.y_val
    y_pred = evidence.y_pred_val
    
    n_classes = y_proba.shape[1] if len(y_proba.shape) > 1 else 2
    is_binary = n_classes == 2
    
    try:
        # Get predicted probabilities for the predicted class
        if is_binary:
            if len(y_proba.shape) == 1:
                pred_proba = y_proba
            else:
                pred_proba = y_proba[:, 1]
            max_proba = np.maximum(y_proba[:, 0] if len(y_proba.shape) > 1 else 1 - y_proba, pred_proba)
        else:
            max_proba = np.max(y_proba, axis=1)
        
        # Average predicted probability (confidence)
        signals.avg_predicted_probability = float(np.mean(max_proba))
        
        # Probability entropy (measure of uncertainty)
        if is_binary:
            if len(y_proba.shape) == 1:
                p = np.column_stack([1 - y_proba, y_proba])
            else:
                p = y_proba
        else:
            p = y_proba
        
        p_clipped = np.clip(p, 1e-10, 1 - 1e-10)
        entropy = -np.sum(p_clipped * np.log(p_clipped), axis=1)
        signals.probability_entropy = float(np.mean(entropy))
        
        # Brier score (probability calibration quality)
        if is_binary:
            if len(y_proba.shape) == 1:
                prob_pos = y_proba
            else:
                prob_pos = y_proba[:, 1]
            signals.brier_score = float(brier_score_loss(y_true, prob_pos))
        else:
            y_true_bin = np.zeros((len(y_true), n_classes))
            for i, cls in enumerate(unique_classes):
                y_true_bin[:, i] = (y_true == cls).astype(int)
            signals.brier_score = float(np.mean((y_proba - y_true_bin) ** 2))
        
        # High and low confidence ratios
        signals.high_confidence_ratio = float(np.mean(max_proba >= 0.9))
        signals.low_confidence_ratio = float(np.mean(max_proba < 0.6))
        
        # Confidence for correct vs incorrect predictions
        if y_pred is not None:
            correct_mask = y_true == y_pred
            if np.any(correct_mask):
                signals.avg_confidence_correct = float(np.mean(max_proba[correct_mask]))
            if np.any(~correct_mask):
                signals.avg_confidence_incorrect = float(np.mean(max_proba[~correct_mask]))
            
            if signals.avg_confidence_correct is not None and signals.avg_confidence_incorrect is not None:
                signals.confidence_gap = signals.avg_confidence_correct - signals.avg_confidence_incorrect
        
        # Class separation score (how well probabilities separate classes)
        if is_binary:
            if len(y_proba.shape) == 1:
                prob_pos = y_proba
            else:
                prob_pos = y_proba[:, 1]
            
            try:
                signals.class_separation_score = float(roc_auc_score(y_true, prob_pos))
            except Exception:
                pass
        else:
            try:
                signals.class_separation_score = float(roc_auc_score(
                    y_true, y_proba, multi_class='ovr', average='weighted'
                ))
            except Exception:
                pass
        
        # Per-class average probability and std
        per_class_avg = {}
        per_class_std = {}
        for i, cls in enumerate(unique_classes):
            cls_mask = y_true == cls
            if np.any(cls_mask):
                if is_binary:
                    if len(y_proba.shape) == 1:
                        prob_for_cls = np.where(y_true[cls_mask] == unique_classes[1], 
                                                y_proba[cls_mask], 
                                                1 - y_proba[cls_mask])
                    else:
                        cls_idx = list(unique_classes).index(cls)
                        prob_for_cls = y_proba[cls_mask, cls_idx]
                else:
                    cls_idx = list(unique_classes).index(cls)
                    prob_for_cls = y_proba[cls_mask, cls_idx]
                
                per_class_avg[str(cls)] = float(np.mean(prob_for_cls))
                per_class_std[str(cls)] = float(np.std(prob_for_cls))
        
        signals.per_class_avg_probability = per_class_avg
        signals.per_class_probability_std = per_class_std
        
        # Threshold analysis for binary classification
        if is_binary and len(y_proba.shape) > 1:
            signals.threshold_metrics = _compute_threshold_metrics(y_true, y_proba[:, 1], unique_classes)
            optimal = _find_optimal_threshold(y_true, y_proba[:, 1])
            if optimal:
                signals.optimal_threshold = optimal['threshold']
                signals.optimal_threshold_f1 = optimal['f1']
        
        # Calibration error (expected calibration error approximation)
        signals.calibration_error = _compute_calibration_error(y_true, max_proba, y_pred)
        
    except Exception as e:
        pass


def _compute_threshold_metrics(
    y_true: np.ndarray, 
    y_proba_pos: np.ndarray,
    unique_classes: np.ndarray
) -> List[Dict[str, float]]:
    """
    Compute metrics at different probability thresholds.
    
    Analyzes how precision, recall, and F1 change as the decision
    threshold varies from 0.1 to 0.9.
    """
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    metrics = []
    
    pos_class = unique_classes[1] if len(unique_classes) > 1 else 1
    
    for thresh in thresholds:
        y_pred_thresh = np.where(y_proba_pos >= thresh, pos_class, unique_classes[0])
        
        try:
            prec = precision_score(y_true, y_pred_thresh, pos_label=pos_class, zero_division=0)
            rec = recall_score(y_true, y_pred_thresh, pos_label=pos_class, zero_division=0)
            f1 = f1_score(y_true, y_pred_thresh, pos_label=pos_class, zero_division=0)
            
            metrics.append({
                'threshold': float(thresh),
                'precision': float(prec),
                'recall': float(rec),
                'f1': float(f1)
            })
        except Exception:
            pass
    
    return metrics


def _find_optimal_threshold(
    y_true: np.ndarray, 
    y_proba_pos: np.ndarray
) -> Optional[Dict[str, float]]:
    """
    Find the optimal probability threshold that maximizes F1 score.
    
    Uses a fine-grained search over thresholds to find the best
    balance between precision and recall.
    """
    thresholds = np.arange(0.05, 0.96, 0.05)
    best_f1 = 0
    best_threshold = 0.5
    
    for thresh in thresholds:
        y_pred = (y_proba_pos >= thresh).astype(int)
        
        try:
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh
        except Exception:
            pass
    
    return {'threshold': float(best_threshold), 'f1': float(best_f1)}


def _compute_calibration_error(
    y_true: np.ndarray, 
    max_proba: np.ndarray,
    y_pred: np.ndarray
) -> Optional[float]:
    """
    Compute Expected Calibration Error (ECE).
    
    ECE measures how well the predicted probabilities match the
    actual accuracy. Lower values indicate better calibration.
    """
    try:
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        ece = 0.0
        for i in range(n_bins):
            in_bin = (max_proba > bin_boundaries[i]) & (max_proba <= bin_boundaries[i + 1])
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(y_true[in_bin] == y_pred[in_bin])
                avg_confidence_in_bin = np.mean(max_proba[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return float(ece)
    except Exception:
        return None


def _extract_regression_signals(evidence: Evidence, signals: Signals) -> None:
    """Extract regression-specific signals (residual analysis)."""
    
    if evidence.y_pred_train is None:
        return
    
    # Training residuals
    residuals = evidence.y_train - evidence.y_pred_train
    
    signals.residual_mean = float(np.mean(residuals))
    signals.residual_std = float(np.std(residuals))
    
    # Skewness and kurtosis for residual distribution analysis
    if len(residuals) > 3:
        try:
            signals.residual_skew = float(stats.skew(residuals))
            signals.residual_kurtosis = float(stats.kurtosis(residuals))
        except Exception:
            pass


def _extract_feature_signals(evidence: Evidence, signals: Signals) -> None:
    """Extract feature-level signals."""
    
    X = evidence.X_train
    
    if len(X.shape) != 2 or X.shape[1] < 2:
        return  # Need at least 2 features
    
    try:
        # Feature correlation matrix
        # Handle potential NaN/inf values
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Only compute if we have enough variance
        variances = np.var(X_clean, axis=0)
        if np.all(variances > 1e-10):
            corr_matrix = np.corrcoef(X_clean, rowvar=False)
            signals.feature_correlations = corr_matrix
            
            # Find highly correlated feature pairs
            high_corr_pairs = []
            n_features = corr_matrix.shape[0]
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    corr = abs(corr_matrix[i, j])
                    if corr > 0.9:  # Threshold for high correlation
                        high_corr_pairs.append((i, j, float(corr)))
            
            if high_corr_pairs:
                signals.high_correlation_pairs = sorted(
                    high_corr_pairs, 
                    key=lambda x: x[2], 
                    reverse=True
                )
    except Exception:
        pass  # Handle numerical issues gracefully
    
    # Feature-target correlations
    try:
        y = evidence.y_train.astype(float)
        feature_target_corr = []
        for i in range(X.shape[1]):
            corr = np.corrcoef(X[:, i], y)[0, 1]
            if not np.isnan(corr):
                feature_target_corr.append(corr)
            else:
                feature_target_corr.append(0.0)
        signals.feature_target_correlations = np.array(feature_target_corr)
    except Exception:
        pass


def _extract_leakage_signals(evidence: Evidence, signals: Signals) -> None:
    """Extract signals that may indicate data leakage."""
    
    # Already computed CV vs holdout gap in CV signals
    
    # Look for suspiciously high feature-target correlations
    if signals.feature_target_correlations is not None:
        suspicious = []
        for i, corr in enumerate(signals.feature_target_correlations):
            if abs(corr) > 0.95:  # Threshold for suspicious correlation
                suspicious.append((i, float(corr)))
        
        if suspicious:
            signals.suspicious_feature_correlations = sorted(
                suspicious,
                key=lambda x: abs(x[1]),
                reverse=True
            )


def compute_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: TaskType,
    metric: str = "default"
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
