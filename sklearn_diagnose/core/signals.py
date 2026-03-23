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
    average_precision_score,
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

    # Extract probability-based signals if available
    _extract_probability_signals(evidence, signals)


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


def _extract_probability_signals(evidence: Evidence, signals: Signals) -> None:
    """
    Extract probability prediction quality signals.

    This function analyzes probability outputs to assess:
    - Probability distribution quality
    - Prediction confidence patterns
    - Class separation capability
    - Threshold sensitivity (binary)
    - Calibration quality

    Only processes validation set probabilities if available.
    """
    # Check if probability predictions are available
    if evidence.y_proba_val is None or evidence.y_val is None:
        return

    y_proba = evidence.y_proba_val
    y_true = evidence.y_val

    # Validate probability array shape
    if len(y_proba.shape) != 2 or y_proba.shape[1] < 2:
        return

    n_samples, n_classes = y_proba.shape
    if n_samples == 0 or n_samples != len(y_true):
        return

    signals.has_probability_outputs = True

    try:
        # 1. Probability distribution analysis
        _analyze_probability_distribution(y_proba, signals)

        # 2. Confidence analysis
        _analyze_confidence_patterns(y_proba, y_true, signals)

        # 3. Class separation analysis
        _analyze_class_separation(y_proba, signals)

        # 4. Per-class probability analysis
        _analyze_per_class_probabilities(y_proba, y_true, signals)

        # 5. Binary-specific threshold analysis
        if n_classes == 2:
            _analyze_threshold_sensitivity(y_proba, y_true, signals)

    except Exception:
        # Graceful degradation - don't break diagnosis if probability analysis fails
        pass


def _analyze_probability_distribution(y_proba: np.ndarray, signals: Signals) -> None:
    """Analyze the distribution of predicted probabilities."""
    # Get max probability for each sample (confidence in predicted class)
    max_proba = np.max(y_proba, axis=1)

    signals.proba_mean = float(np.mean(max_proba))
    signals.proba_std = float(np.std(max_proba))

    # Compute prediction entropy (uncertainty measure)
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    proba_clipped = np.clip(y_proba, epsilon, 1.0)
    entropy = -np.sum(proba_clipped * np.log(proba_clipped), axis=1)
    signals.proba_entropy = float(np.mean(entropy))


def _analyze_confidence_patterns(
    y_proba: np.ndarray, y_true: np.ndarray, signals: Signals
) -> None:
    """Analyze prediction confidence patterns and their relationship to accuracy."""
    max_proba = np.max(y_proba, axis=1)
    y_pred = np.argmax(y_proba, axis=1)
    correct = (y_pred == y_true).astype(int)

    # High confidence ratio (> 0.9)
    signals.high_confidence_ratio = float(np.mean(max_proba > 0.9))

    # Low confidence ratio (< 0.6)
    signals.low_confidence_ratio = float(np.mean(max_proba < 0.6))

    # Correlation between confidence and correctness
    if len(max_proba) > 1 and np.std(max_proba) > 0 and np.std(correct) > 0:
        signals.confidence_accuracy_correlation = float(
            np.corrcoef(max_proba, correct)[0, 1]
        )


def _analyze_class_separation(y_proba: np.ndarray, signals: Signals) -> None:
    """Analyze how well-separated the classes are based on probability margins."""
    # Sort probabilities in descending order for each sample
    sorted_proba = np.sort(y_proba, axis=1)[:, ::-1]

    # Margin between top-2 class probabilities
    if sorted_proba.shape[1] >= 2:
        margin = sorted_proba[:, 0] - sorted_proba[:, 1]
        signals.proba_margin_mean = float(np.mean(margin))
        signals.proba_margin_std = float(np.std(margin))

        # Ambiguous predictions (margin < 0.2)
        signals.ambiguous_predictions_ratio = float(np.mean(margin < 0.2))


def _analyze_per_class_probabilities(
    y_proba: np.ndarray, y_true: np.ndarray, signals: Signals
) -> None:
    """Analyze probability distributions for each true class."""
    n_classes = y_proba.shape[1]
    unique_classes = np.unique(y_true)

    per_class_mean = {}
    per_class_std = {}

    for cls in unique_classes:
        cls_mask = y_true == cls
        if np.sum(cls_mask) > 0:
            # Probability assigned to the true class
            true_class_proba = y_proba[cls_mask, cls]
            per_class_mean[str(cls)] = float(np.mean(true_class_proba))
            per_class_std[str(cls)] = float(np.std(true_class_proba))

    signals.per_class_proba_mean = per_class_mean
    signals.per_class_proba_std = per_class_std


def _analyze_threshold_sensitivity(
    y_proba: np.ndarray, y_true: np.ndarray, signals: Signals
) -> None:
    """
    Analyze threshold sensitivity for binary classification.

    Computes optimal threshold and performance across different thresholds.
    """
    # Positive class probabilities
    pos_proba = y_proba[:, 1]

    # Compute AUC-ROC and AUC-PR
    try:
        signals.auc_roc = float(roc_auc_score(y_true, pos_proba))
        signals.auc_pr = float(average_precision_score(y_true, pos_proba))
    except Exception:
        pass

    # Find optimal threshold by F1 score
    thresholds = np.linspace(0.1, 0.9, 81)
    f1_scores = []

    for thresh in thresholds:
        y_pred_thresh = (pos_proba >= thresh).astype(int)
        try:
            f1 = f1_score(y_true, y_pred_thresh, zero_division=0)
            f1_scores.append(f1)
        except Exception:
            f1_scores.append(0.0)

    if f1_scores:
        best_idx = np.argmax(f1_scores)
        signals.optimal_threshold = float(thresholds[best_idx])

        # Threshold sensitivity: range of F1 scores across thresholds
        signals.threshold_sensitivity = float(np.max(f1_scores) - np.min(f1_scores))

    # Calibration error (simplified - mean absolute error between proba and empirical accuracy)
    try:
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        calibration_errors = []

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            # Find samples in this bin
            if i == n_bins - 1:
                in_bin = (pos_proba >= bin_lower) & (pos_proba <= bin_upper)
            else:
                in_bin = (pos_proba >= bin_lower) & (pos_proba < bin_upper)

            if np.sum(in_bin) > 0:
                avg_confidence = np.mean(pos_proba[in_bin])
                avg_accuracy = np.mean(y_true[in_bin])
                calibration_errors.append(abs(avg_confidence - avg_accuracy))

        if calibration_errors:
            signals.proba_calibration_error = float(np.mean(calibration_errors))
    except Exception:
        pass


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
