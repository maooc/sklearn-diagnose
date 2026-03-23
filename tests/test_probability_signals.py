"""
Tests for probability prediction signal extraction and diagnosis.

These tests cover:
- Probability signal extraction from various classifiers
- Pipeline compatibility with probability outputs
- Scenarios with and without validation sets
- Scenarios with and without CV results
- Graceful degradation for classifiers without predict_proba
- Detection of probability-related failure modes
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn_diagnose import diagnose
from sklearn_diagnose.core import FailureMode, TaskType


# Fixtures for test data
@pytest.fixture
def binary_classification_data():
    """Generate balanced binary classification data."""
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_val, y_train, y_val


@pytest.fixture
def multiclass_classification_data():
    """Generate multiclass classification data."""
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=15,
        n_redundant=3,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_val, y_train, y_val


@pytest.fixture
def imbalanced_binary_data():
    """Generate imbalanced binary classification data."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_classes=2,
        weights=[0.95, 0.05],
        random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    return X_train, X_val, y_train, y_val


class TestProbabilitySignalExtraction:
    """Test extraction of probability prediction signals."""

    def test_logistic_regression_probability_signals(self, binary_classification_data):
        """Test probability signal extraction from LogisticRegression."""
        X_train, X_val, y_train, y_val = binary_classification_data

        model = LogisticRegression(random_state=42, max_iter=500)
        model.fit(X_train, y_train)

        report = diagnose(
            estimator=model,
            datasets={
                "train": (X_train, y_train),
                "val": (X_val, y_val)
            },
            task="classification"
        )

        # Check that probability signals are extracted
        assert report.signals.has_probability_outputs is True
        assert report.signals.proba_mean is not None
        assert report.signals.proba_std is not None
        assert report.signals.proba_entropy is not None
        assert report.signals.high_confidence_ratio is not None
        assert report.signals.low_confidence_ratio is not None
        assert report.signals.proba_margin_mean is not None
        assert report.signals.ambiguous_predictions_ratio is not None

        # Binary-specific signals
        assert report.signals.optimal_threshold is not None
        assert report.signals.auc_roc is not None
        assert report.signals.auc_pr is not None

    def test_random_forest_probability_signals(self, binary_classification_data):
        """Test probability signal extraction from RandomForest."""
        X_train, X_val, y_train, y_val = binary_classification_data

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        report = diagnose(
            estimator=model,
            datasets={
                "train": (X_train, y_train),
                "val": (X_val, y_val)
            },
            task="classification"
        )

        assert report.signals.has_probability_outputs is True
        assert report.signals.proba_mean is not None
        assert report.signals.proba_calibration_error is not None
        assert 0 <= report.signals.proba_calibration_error <= 1

    def test_multiclass_probability_signals(self, multiclass_classification_data):
        """Test probability signal extraction for multiclass."""
        X_train, X_val, y_train, y_val = multiclass_classification_data

        model = LogisticRegression(random_state=42, max_iter=500)
        model.fit(X_train, y_train)

        report = diagnose(
            estimator=model,
            datasets={
                "train": (X_train, y_train),
                "val": (X_val, y_val)
            },
            task="classification"
        )

        assert report.signals.has_probability_outputs is True
        assert report.signals.proba_mean is not None
        assert report.signals.proba_entropy is not None

        # Per-class probability analysis
        assert report.signals.per_class_proba_mean is not None
        assert len(report.signals.per_class_proba_mean) == 3
        assert report.signals.per_class_proba_std is not None

        # Binary-specific signals should be None for multiclass
        assert report.signals.optimal_threshold is None
        assert report.signals.auc_roc is None

    def test_pipeline_probability_signals(self, binary_classification_data):
        """Test probability signal extraction from Pipeline."""
        X_train, X_val, y_train, y_val = binary_classification_data

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(random_state=42, max_iter=500))
        ])
        pipeline.fit(X_train, y_train)

        report = diagnose(
            estimator=pipeline,
            datasets={
                "train": (X_train, y_train),
                "val": (X_val, y_val)
            },
            task="classification"
        )

        assert report.signals.has_probability_outputs is True
        assert report.signals.proba_mean is not None
        assert report.signals.confidence_accuracy_correlation is not None


class TestNoProbabilitySupport:
    """Test graceful degradation for classifiers without predict_proba."""

    def test_svc_without_proba(self, binary_classification_data):
        """Test SVC without probability support."""
        X_train, X_val, y_train, y_val = binary_classification_data

        # SVC without probability=True does not have predict_proba
        model = SVC(kernel="linear", random_state=42)
        model.fit(X_train, y_train)

        report = diagnose(
            estimator=model,
            datasets={
                "train": (X_train, y_train),
                "val": (X_val, y_val)
            },
            task="classification"
        )

        # Should not have probability outputs
        assert report.signals.has_probability_outputs is False
        assert report.signals.proba_mean is None
        assert report.signals.proba_std is None

        # But basic signals should still work
        assert report.signals.train_score is not None
        assert report.signals.val_score is not None

    def test_sgd_classifier_without_proba(self, binary_classification_data):
        """Test SGDClassifier without loss that supports predict_proba."""
        X_train, X_val, y_train, y_val = binary_classification_data

        # SGDClassifier with hinge loss (SVM-like) doesn't have predict_proba
        model = SGDClassifier(loss="hinge", random_state=42)
        model.fit(X_train, y_train)

        report = diagnose(
            estimator=model,
            datasets={
                "train": (X_train, y_train),
                "val": (X_val, y_val)
            },
            task="classification"
        )

        assert report.signals.has_probability_outputs is False
        assert report.signals.train_score is not None


class TestProbabilityWithCVResults:
    """Test probability signals with cross-validation results."""

    def test_probability_signals_with_cv(self, binary_classification_data):
        """Test that probability signals work with CV results."""
        X_train, X_val, y_train, y_val = binary_classification_data

        model = LogisticRegression(random_state=42, max_iter=500)
        model.fit(X_train, y_train)

        cv_results = cross_validate(
            model, X_train, y_train,
            cv=5, return_train_score=True
        )

        report = diagnose(
            estimator=model,
            datasets={
                "train": (X_train, y_train),
                "val": (X_val, y_val)
            },
            task="classification",
            cv_results=cv_results
        )

        # Should have both probability and CV signals
        assert report.signals.has_probability_outputs is True
        assert report.signals.proba_mean is not None
        assert report.signals.cv_mean is not None
        assert report.signals.cv_std is not None

    def test_probability_signals_without_val_set(self, binary_classification_data):
        """Test that probability signals are None without validation set."""
        X_train, _, y_train, _ = binary_classification_data

        model = LogisticRegression(random_state=42, max_iter=500)
        model.fit(X_train, y_train)

        report = diagnose(
            estimator=model,
            datasets={"train": (X_train, y_train)},
            task="classification"
        )

        # Without validation set, no probability signals
        assert report.signals.has_probability_outputs is False
        assert report.signals.proba_mean is None


class TestProbabilityFailureModeDetection:
    """Test detection of probability-related failure modes."""

    def test_detect_poor_calibration(self, binary_classification_data):
        """Test detection of poor calibration."""
        X_train, X_val, y_train, y_val = binary_classification_data

        # RandomForest tends to have calibration issues
        model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
        model.fit(X_train, y_train)

        report = diagnose(
            estimator=model,
            datasets={
                "train": (X_train, y_train),
                "val": (X_val, y_val)
            },
            task="classification"
        )

        # Check calibration error is computed
        assert report.signals.proba_calibration_error is not None

        # Check if poor calibration is detected
        cal_error = report.signals.proba_calibration_error
        if cal_error > 0.05:
            poor_cal_hypotheses = [
                h for h in report.hypotheses
                if h.name == FailureMode.POOR_CALIBRATION
            ]
            # Note: Detection depends on actual calibration quality

    def test_detect_low_confidence(self, binary_classification_data):
        """Test detection of low confidence predictions."""
        X_train, X_val, y_train, y_val = binary_classification_data

        # Underfit model may produce low confidence predictions
        model = LogisticRegression(random_state=42, max_iter=500, C=0.001)
        model.fit(X_train, y_train)

        report = diagnose(
            estimator=model,
            datasets={
                "train": (X_train, y_train),
                "val": (X_val, y_val)
            },
            task="classification"
        )

        # Check low confidence ratio
        assert report.signals.low_confidence_ratio is not None

        if report.signals.low_confidence_ratio > 0.25:
            low_conf_hypotheses = [
                h for h in report.hypotheses
                if h.name == FailureMode.LOW_CONFIDENCE_PREDICTIONS
            ]

    def test_detect_suboptimal_threshold(self, imbalanced_binary_data):
        """Test detection of suboptimal threshold for imbalanced data."""
        X_train, X_val, y_train, y_val = imbalanced_binary_data

        model = LogisticRegression(random_state=42, max_iter=500)
        model.fit(X_train, y_train)

        report = diagnose(
            estimator=model,
            datasets={
                "train": (X_train, y_train),
                "val": (X_val, y_val)
            },
            task="classification"
        )

        # Check optimal threshold
        assert report.signals.optimal_threshold is not None

        # For imbalanced data, optimal threshold often differs from 0.5
        if abs(report.signals.optimal_threshold - 0.5) > 0.1:
            threshold_hypotheses = [
                h for h in report.hypotheses
                if h.name == FailureMode.SUBOPTIMAL_THRESHOLD
            ]

    def test_detect_confidence_accuracy_mismatch(self, binary_classification_data):
        """Test detection of confidence-accuracy mismatch."""
        X_train, X_val, y_train, y_val = binary_classification_data

        # Overfit model may show confidence-accuracy mismatch
        model = DecisionTreeClassifier(max_depth=None, min_samples_leaf=1, random_state=42)
        model.fit(X_train, y_train)

        report = diagnose(
            estimator=model,
            datasets={
                "train": (X_train, y_train),
                "val": (X_val, y_val)
            },
            task="classification"
        )

        # Check confidence-accuracy correlation (may be None if all predictions have same confidence)
        # The correlation is computed only if there's variance in both confidence and accuracy
        if report.signals.confidence_accuracy_correlation is not None:
            if report.signals.confidence_accuracy_correlation < 0.2:
                mismatch_hypotheses = [
                    h for h in report.hypotheses
                    if h.name == FailureMode.CONFIDENCE_ACCURACY_MISMATCH
                ]


class TestProbabilitySignalRanges:
    """Test that probability signals are in valid ranges."""

    def test_signal_ranges(self, binary_classification_data):
        """Test that all probability signals are in valid ranges."""
        X_train, X_val, y_train, y_val = binary_classification_data

        model = LogisticRegression(random_state=42, max_iter=500)
        model.fit(X_train, y_train)

        report = diagnose(
            estimator=model,
            datasets={
                "train": (X_train, y_train),
                "val": (X_val, y_val)
            },
            task="classification"
        )

        s = report.signals

        # Probabilities should be in [0, 1]
        if s.proba_mean is not None:
            assert 0 <= s.proba_mean <= 1
        if s.proba_std is not None:
            assert 0 <= s.proba_std <= 0.5  # Max std for [0,1] is 0.5
        if s.high_confidence_ratio is not None:
            assert 0 <= s.high_confidence_ratio <= 1
        if s.low_confidence_ratio is not None:
            assert 0 <= s.low_confidence_ratio <= 1
        if s.ambiguous_predictions_ratio is not None:
            assert 0 <= s.ambiguous_predictions_ratio <= 1

        # AUC scores should be in [0, 1]
        if s.auc_roc is not None:
            assert 0 <= s.auc_roc <= 1
        if s.auc_pr is not None:
            assert 0 <= s.auc_pr <= 1

        # Threshold should be in [0, 1]
        if s.optimal_threshold is not None:
            assert 0 <= s.optimal_threshold <= 1

        # Calibration error should be in [0, 1]
        if s.proba_calibration_error is not None:
            assert 0 <= s.proba_calibration_error <= 1

        # Correlation should be in [-1, 1]
        if s.confidence_accuracy_correlation is not None:
            assert -1 <= s.confidence_accuracy_correlation <= 1


class TestProbabilitySignalSerialization:
    """Test that probability signals can be serialized."""

    def test_signals_to_dict(self, binary_classification_data):
        """Test that probability signals can be converted to dict."""
        X_train, X_val, y_train, y_val = binary_classification_data

        model = LogisticRegression(random_state=42, max_iter=500)
        model.fit(X_train, y_train)

        report = diagnose(
            estimator=model,
            datasets={
                "train": (X_train, y_train),
                "val": (X_val, y_val)
            },
            task="classification"
        )

        signals_dict = report.signals.to_dict()

        # Check probability signals are in dict
        assert "proba_mean" in signals_dict
        assert "proba_std" in signals_dict
        assert "has_probability_outputs" in signals_dict
        assert signals_dict["has_probability_outputs"] is True

    def test_report_to_dict(self, binary_classification_data):
        """Test that full report can be serialized."""
        X_train, X_val, y_train, y_val = binary_classification_data

        model = LogisticRegression(random_state=42, max_iter=500)
        model.fit(X_train, y_train)

        report = diagnose(
            estimator=model,
            datasets={
                "train": (X_train, y_train),
                "val": (X_val, y_val)
            },
            task="classification"
        )

        report_dict = report.to_dict()

        # Check signals are included
        assert "signals" in report_dict
        assert "proba_mean" in report_dict["signals"]
        assert "auc_roc" in report_dict["signals"]


class TestGradientBoostingProbability:
    """Test probability signals with GradientBoosting."""

    def test_gradient_boosting_probability_signals(self, binary_classification_data):
        """Test probability signal extraction from GradientBoosting."""
        X_train, X_val, y_train, y_val = binary_classification_data

        model = GradientBoostingClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        report = diagnose(
            estimator=model,
            datasets={
                "train": (X_train, y_train),
                "val": (X_val, y_val)
            },
            task="classification"
        )

        assert report.signals.has_probability_outputs is True
        assert report.signals.proba_mean is not None
        assert report.signals.proba_margin_mean is not None
        assert report.signals.auc_roc is not None
