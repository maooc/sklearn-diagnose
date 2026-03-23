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
    
    # Extract probability quality signals if available (use val first, then train)
    if evidence.y_proba_val is not None or evidence.y_proba_train is not None:
        _extract_probability_quality_signals(evidence, signals)


def _extract_probability_quality_signals(evidence: Evidence, signals: Signals) -> None:
    """
    Extract probability prediction quality signals for classification models.
    
    This includes:
    - Confidence distribution statistics
    - Class separation quality
    - Calibration metrics
    """
    signals.has_probability_predictions = True
    
    # Use validation data first, fall back to training data if needed
    if evidence.y_proba_val is not None:
        y_proba = evidence.y_proba_val
        y_true = evidence.y_val
    else:
        y_proba = evidence.y_proba_train
        y_true = evidence.y_train
        
    n_samples = y_proba.shape[0]
    
    # Get prediction confidence (max probability for each sample)
    pred_confidences = np.max(y_proba, axis=1)
    
    # Basic confidence statistics
    signals.mean_prediction_confidence = float(np.mean(pred_confidences))
    signals.median_prediction_confidence = float(np.median(pred_confidences))
    signals.min_prediction_confidence = float(np.min(pred_confidences))
    signals.max_prediction_confidence = float(np.max(pred_confidences))
    signals.prediction_confidence_std = float(np.std(pred_confidences))
    
    # Low confidence ratios
    signals.low_confidence_ratio = float(np.mean(pred_confidences < 0.7))
    signals.very_low_confidence_ratio = float(np.mean(pred_confidences < 0.5))
    
    # Class separation metrics
    sorted_proba = np.sort(y_proba, axis=1)
    max_proba = sorted_proba[:, -1]
    second_max_proba = sorted_proba[:, -2] if sorted_proba.shape[1] > 1 else np.zeros_like(max_proba)
    
    signals.mean_max_class_probability = float(np.mean(max_proba))
    signals.mean_second_max_class_probability = float(np.mean(second_max_proba))
    signals.mean_confidence_margin = float(np.mean(max_proba - second_max_proba))
    
    # Class separation score (normalized margin)
    if signals.mean_confidence_margin is not None:
        signals.class_separation_score = float(
            np.clip(signals.mean_confidence_margin * 2, 0, 1)
        )
    
    # Calibration analysis (simplified ECE - Expected Calibration Error)
    if y_true is not None:
        _extract_calibration_signals(y_true, y_proba, pred_confidences, signals)
    
    # Threshold analysis (for binary classification with true labels)
    if y_true is not None and y_proba.shape[1] == 2:
        _extract_threshold_analysis_signals(y_true, y_proba, signals)


def _extract_calibration_signals(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    pred_confidences: np.ndarray,
    signals: Signals
) -> None:
    """Extract probability calibration signals."""
    try:
        n_classes = y_proba.shape[1]
        n_samples = len(y_true)
        
        # Convert true labels to one-hot for multi-class
        if n_classes > 2:
            y_true_one_hot = np.zeros((n_samples, n_classes))
            y_true_one_hot[np.arange(n_samples), y_true] = 1
        else:
            # Binary classification
            y_true_one_hot = np.column_stack([1 - y_true, y_true])
        
        # Calculate calibration error (simplified ECE)
        # Bin predictions by confidence level
        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)
        digitized = np.digitize(pred_confidences, bins[1:-1])
        
        ece = 0.0
        max_error = 0.0
        overconfident_count = 0
        underconfident_count = 0
        
        for bin_idx in range(n_bins):
            bin_mask = digitized == bin_idx
            if np.any(bin_mask):
                bin_conf = np.mean(pred_confidences[bin_mask])
                bin_acc = np.mean(y_true_one_hot[bin_mask, y_true[bin_mask]] if n_classes > 2 else y_true[bin_mask])
                bin_error = abs(bin_conf - bin_acc)
                bin_weight = np.sum(bin_mask) / n_samples
                
                ece += bin_error * bin_weight
                max_error = max(max_error, bin_error)
                
                if bin_conf > bin_acc + 0.05:
                    overconfident_count += np.sum(bin_mask)
                elif bin_conf < bin_acc - 0.05:
                    underconfident_count += np.sum(bin_mask)
        
        signals.expected_calibration_error = float(ece)
        signals.max_calibration_error = float(max_error)
        signals.overconfidence_ratio = float(overconfident_count / n_samples)
        signals.underconfidence_ratio = float(underconfident_count / n_samples)
        
    except Exception as e:
        # Calibration computation may fail in edge cases
        pass


def _extract_threshold_analysis_signals(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    signals: Signals
) -> None:
    """
    Extract threshold analysis signals for binary classification models.
    
    This includes:
    - Optimal thresholds for precision, recall, and F1
    - Threshold sensitivity analysis
    - Score stability index
    - Performance metrics at different thresholds
    """
    try:
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        n_samples = len(y_true)
        pos_proba = y_proba[:, 1]  # Probability of positive class
        
        # Calculate default predictions
        default_preds = (pos_proba >= 0.5).astype(int)
        default_precision = precision_score(y_true, default_preds, zero_division=0)
        default_recall = recall_score(y_true, default_preds, zero_division=0)
        default_f1 = f1_score(y_true, default_preds, zero_division=0)
        
        # Find optimal threshold for F1 score (grid search)
        thresholds = np.linspace(0.01, 0.99, 99)
        f1_scores = []
        precision_scores = []
        recall_scores = []
        
        for thresh in thresholds:
            preds = (pos_proba >= thresh).astype(int)
            f1_scores.append(f1_score(y_true, preds, zero_division=0))
            precision_scores.append(precision_score(y_true, preds, zero_division=0))
            recall_scores.append(recall_score(y_true, preds, zero_division=0))
        
        # Optimal thresholds
        best_f1_idx = np.argmax(f1_scores)
        best_precision_idx = np.argmax(precision_scores)
        best_recall_idx = np.argmax(recall_scores)
        
        signals.optimal_threshold_f1 = float(thresholds[best_f1_idx])
        signals.optimal_threshold_precision = float(thresholds[best_precision_idx])
        signals.optimal_threshold_recall = float(thresholds[best_recall_idx])
        
        # Threshold sensitivity analysis
        # Calculate how predictions change with threshold variations
        preds_05 = (pos_proba >= 0.5).astype(int)
        preds_04 = (pos_proba >= 0.4).astype(int)
        preds_06 = (pos_proba >= 0.6).astype(int)
        preds_045 = (pos_proba >= 0.45).astype(int)
        preds_055 = (pos_proba >= 0.55).astype(int)
        
        # Percentage of predictions that change with ±0.1 threshold
        change_high = np.mean(preds_04 != preds_06)
        signals.threshold_sensitivity_high = float(change_high)
        
        # Percentage of predictions that change with ±0.05 threshold
        change_low = np.mean(preds_045 != preds_055)
        signals.threshold_sensitivity_low = float(change_low)
        
        # Score stability index
        # Measures how close probabilities are to the default threshold (0.5)
        # Lower values mean probabilities are clustered around the decision boundary
        distance_to_threshold = np.abs(pos_proba - 0.5)
        signals.score_stability_index = float(np.mean(distance_to_threshold))
        
        # Detailed threshold analysis dictionary
        signals.threshold_analysis = {
            "default_threshold": 0.5,
            "default_metrics": {
                "precision": float(default_precision),
                "recall": float(default_recall),
                "f1": float(default_f1)
            },
            "optimal_metrics": {
                "best_f1": float(f1_scores[best_f1_idx]),
                "best_precision": float(precision_scores[best_precision_idx]),
                "best_recall": float(recall_scores[best_recall_idx])
            },
            "threshold_range_analysis": {
                "thresholds": thresholds.tolist(),
                "f1_scores": [float(s) for s in f1_scores],
                "precision_scores": [float(s) for s in precision_scores],
                "recall_scores": [float(s) for s in recall_scores]
            },
            "stability_metrics": {
                "score_stability_index": signals.score_stability_index,
                "threshold_sensitivity_01": signals.threshold_sensitivity_high,
                "threshold_sensitivity_005": signals.threshold_sensitivity_low
            }
        }
        
    except Exception as e:
        # Threshold analysis may fail in edge cases
        pass


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
