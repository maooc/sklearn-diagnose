"""
Unit tests for probability prediction diagnosis in sklearn-diagnose.

This module tests the probability quality analysis capabilities:
- Confidence distribution analysis
- Class separation metrics
- Calibration error (ECE) calculation
- Probability-based hypothesis generation
- Graceful degradation for non-probability models
"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from sklearn_diagnose import diagnose
from sklearn_diagnose.core.schemas import FailureMode, Signals
from sklearn_diagnose.core.signals import extract_all_signals as extract_signals
from sklearn_diagnose.core.evidence import collect_evidence as gather_evidence
from sklearn_diagnose.core.hypotheses import generate_hypotheses as generate_rule_based_hypotheses


# =============================================================================
# Test fixtures
# =============================================================================

@pytest.fixture(scope="module")
def classification_data():
    """Generate synthetic classification data for testing."""
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


@pytest.fixture(scope="module")
def calibrated_classifier(classification_data):
    """A well-calibrated classifier (Logistic Regression)."""
    X_train, _, y_train, _ = classification_data
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_train, y_train)
    return clf


@pytest.fixture(scope="module")
def uncalibrated_classifier(classification_data):
    """A potentially less calibrated classifier (SVM with probabilities)."""
    X_train, _, y_train, _ = classification_data
    clf = SVC(probability=True, random_state=42)
    clf.fit(X_train, y_train)
    return clf


@pytest.fixture(scope="module")
def non_probability_classifier(classification_data):
    """A classifier without predict_proba support."""
    X_train, _, y_train, _ = classification_data
    clf = SVC(probability=False, random_state=42)
    clf.fit(X_train, y_train)
    return clf


@pytest.fixture(scope="module")
def pipeline_classifier(classification_data):
    """A pipeline with a probability-enabled classifier."""
    X_train, _, y_train, _ = classification_data
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    pipe.fit(X_train, y_train)
    return pipe


# =============================================================================
# Tests for probability signal extraction
# =============================================================================

class TestProbabilitySignalExtraction:
    """Tests for extracting probability quality signals."""

    def test_probability_signals_extracted(self, calibrated_classifier, classification_data):
        """Test that probability-related signals are correctly extracted."""
        X_train, X_val, y_train, y_val = classification_data
        
        evidence = gather_evidence(
            estimator=calibrated_classifier,
            datasets={
                "train": (X_train, y_train),
                "val": (X_val, y_val)
            },
            task="classification"
        )
        signals = extract_signals(evidence)
        
        # Verify probability signals exist
        assert signals.has_probability_predictions is True
        assert signals.mean_prediction_confidence is not None
        assert signals.median_prediction_confidence is not None
        assert signals.min_prediction_confidence is not None
        assert signals.max_prediction_confidence is not None
        assert signals.prediction_confidence_std is not None
        
        # Verify confidence values are within valid range [0, 1]
        assert 0 <= signals.mean_prediction_confidence <= 1
        assert 0 <= signals.median_prediction_confidence <= 1
        assert 0 <= signals.min_prediction_confidence <= 1
        assert 0 <= signals.max_prediction_confidence <= 1
        assert 0 <= signals.prediction_confidence_std <= 1

    def test_class_separation_signals(self, calibrated_classifier, classification_data):
        """Test that class separation metrics are computed."""
        X_train, X_val, y_train, y_val = classification_data
        
        evidence = gather_evidence(
            estimator=calibrated_classifier,
            datasets={
                "train": (X_train, y_train),
                "val": (X_val, y_val)
            },
            task="classification"
        )
        signals = extract_signals(evidence)
        
        # Verify class separation signals
        assert signals.class_separation_score is not None
        assert signals.mean_max_class_probability is not None
        assert signals.mean_second_max_class_probability is not None
        assert signals.mean_confidence_margin is not None
        
        # Validate ranges
        assert 0 <= signals.class_separation_score <= 1
        assert 0 <= signals.mean_max_class_probability <= 1
        assert 0 <= signals.mean_second_max_class_probability <= 1
        assert 0 <= signals.mean_confidence_margin <= 1

    def test_calibration_signals(self, calibrated_classifier, classification_data):
        """Test that calibration metrics (ECE) are computed."""
        X_train, X_val, y_train, y_val = classification_data
        
        evidence = gather_evidence(
            estimator=calibrated_classifier,
            datasets={
                "train": (X_train, y_train),
                "val": (X_val, y_val)
            },
            task="classification"
        )
        signals = extract_signals(evidence)
        
        # Verify calibration signals
        assert signals.expected_calibration_error is not None
        assert signals.max_calibration_error is not None
        assert signals.overconfidence_ratio is not None
        assert signals.underconfidence_ratio is not None
        
        # ECE should be a reasonable value
        assert 0 <= signals.expected_calibration_error <= 1
        assert 0 <= signals.max_calibration_error <= 1
        assert 0 <= signals.overconfidence_ratio <= 1
        assert 0 <= signals.underconfidence_ratio <= 1

    def test_low_confidence_ratios(self, calibrated_classifier, classification_data):
        """Test that low confidence ratio signals are computed."""
        X_train, X_val, y_train, y_val = classification_data
        
        evidence = gather_evidence(
            estimator=calibrated_classifier,
            datasets={
                "train": (X_train, y_train),
                "val": (X_val, y_val)
            },
            task="classification"
        )
        signals = extract_signals(evidence)
        
        # Verify confidence ratio signals
        assert signals.low_confidence_ratio is not None
        assert signals.very_low_confidence_ratio is not None
        
        # These should be ratios between 0 and 1
        assert 0 <= signals.low_confidence_ratio <= 1
        assert 0 <= signals.very_low_confidence_ratio <= 1


# =============================================================================
# Tests for graceful degradation (non-probability models)
# =============================================================================

class TestGracefulDegradation:
    """Tests for graceful degradation with non-probability models."""

    def test_non_probability_model_no_crash(self, non_probability_classifier, classification_data):
        """Test that non-probability models don't crash the diagnosis."""
        X_train, X_val, y_train, y_val = classification_data
        
        # This should not raise an exception
        evidence = gather_evidence(
            estimator=non_probability_classifier,
            datasets={
                "train": (X_train, y_train),
                "val": (X_val, y_val)
            },
            task="classification"
        )
        signals = extract_signals(evidence)
        
        # Probability signals should be False/None
        assert signals.has_probability_predictions is False
        assert signals.mean_prediction_confidence is None
        assert signals.expected_calibration_error is None
        assert signals.class_separation_score is None

    def test_diagnosis_with_non_probability_model(self, non_probability_classifier, classification_data):
        """Test full diagnosis workflow with non-probability models."""
        X_train, X_val, y_train, y_val = classification_data
        
        # This should complete successfully
        report = diagnose(
            estimator=non_probability_classifier,
            datasets={
                "train": (X_train, y_train),
                "val": (X_val, y_val)
            },
            task="classification"
        )
        
        # The report should be valid
        assert report is not None
        assert report.signals is not None
        assert report.signals.has_probability_predictions is False
        
        # Other diagnostics should still work
        assert report.signals.train_score is not None
        assert report.signals.val_score is not None


# =============================================================================
# Tests for pipeline compatibility
# =============================================================================

class TestPipelineCompatibility:
    """Tests for probability diagnosis with sklearn Pipelines."""

    def test_pipeline_probability_signals(self, pipeline_classifier, classification_data):
        """Test that probability signals are extracted from pipelines."""
        X_train, X_val, y_train, y_val = classification_data
        
        evidence = gather_evidence(
            estimator=pipeline_classifier,
            datasets={
                "train": (X_train, y_train),
                "val": (X_val, y_val)
            },
            task="classification"
        )
        signals = extract_signals(evidence)
        
        # Probability signals should be available
        assert signals.has_probability_predictions is True
        assert signals.mean_prediction_confidence is not None
        assert signals.expected_calibration_error is not None

    def test_pipeline_full_diagnosis(self, pipeline_classifier, classification_data):
        """Test full diagnosis with a pipeline."""
        X_train, X_val, y_train, y_val = classification_data
        
        report = diagnose(
            estimator=pipeline_classifier,
            datasets={
                "train": (X_train, y_train),
                "val": (X_val, y_val)
            },
            task="classification"
        )
        
        assert report is not None
        assert report.has_pipeline is True
        assert report.signals.has_probability_predictions is True


# =============================================================================
# Tests for hypothesis generation
# =============================================================================

class TestProbabilityHypothesisGeneration:
    """Tests for generating probability calibration hypotheses."""

    def test_probability_calibration_hypothesis_detection(self, calibrated_classifier, classification_data):
        """Test that probability calibration issues can be detected."""
        X_train, X_val, y_train, y_val = classification_data
        
        evidence = gather_evidence(
            estimator=calibrated_classifier,
            datasets={
                "train": (X_train, y_train),
                "val": (X_val, y_val)
            },
            task="classification"
        )
        signals = extract_signals(evidence)
        
        # Manually create a high ECE scenario for testing
        signals.has_probability_predictions = True
        signals.expected_calibration_error = 0.18  # Severe calibration error
        signals.overconfidence_ratio = 0.45
        signals.low_confidence_ratio = 0.35
        
        from sklearn_diagnose.core.schemas import TaskType
        hypotheses = generate_rule_based_hypotheses(signals, TaskType.CLASSIFICATION)
        
        # Should detect probability calibration issues
        calibration_hypotheses = [
            h for h in hypotheses if h.name == FailureMode.PROBABILITY_CALIBRATION
        ]
        
        assert len(calibration_hypotheses) > 0
        assert calibration_hypotheses[0].confidence > 0.5
        assert len(calibration_hypotheses[0].evidence) > 0

    def test_low_confidence_detection(self, calibrated_classifier, classification_data):
        """Test detection of high ratio of low-confidence predictions."""
        X_train, X_val, y_train, y_val = classification_data
        
        evidence = gather_evidence(
            estimator=calibrated_classifier,
            datasets={
                "train": (X_train, y_train),
                "val": (X_val, y_val)
            },
            task="classification"
        )
        signals = extract_signals(evidence)
        
        # Manually set low confidence scenario
        signals.has_probability_predictions = True
        signals.low_confidence_ratio = 0.40  # 40% low confidence predictions
        signals.very_low_confidence_ratio = 0.20
        
        from sklearn_diagnose.core.schemas import TaskType
        hypotheses = generate_rule_based_hypotheses(signals, TaskType.CLASSIFICATION)
        
        calibration_hypotheses = [
            h for h in hypotheses if h.name == FailureMode.PROBABILITY_CALIBRATION
        ]
        
        if calibration_hypotheses:
            assert any("low confidence" in ev.lower() for ev in calibration_hypotheses[0].evidence)

    def test_poor_class_separation_detection(self, calibrated_classifier, classification_data):
        """Test detection of poor class separation."""
        X_train, X_val, y_train, y_val = classification_data
        
        evidence = gather_evidence(
            estimator=calibrated_classifier,
            datasets={
                "train": (X_train, y_train),
                "val": (X_val, y_val)
            },
            task="classification"
        )
        signals = extract_signals(evidence)
        
        # Manually set poor separation scenario
        signals.has_probability_predictions = True
        signals.class_separation_score = 0.25  # Below threshold
        signals.mean_confidence_margin = 0.15
        
        from sklearn_diagnose.core.schemas import TaskType
        hypotheses = generate_rule_based_hypotheses(signals, TaskType.CLASSIFICATION)
        
        calibration_hypotheses = [
            h for h in hypotheses if h.name == FailureMode.PROBABILITY_CALIBRATION
        ]
        
        if calibration_hypotheses:
            assert any("separation" in ev.lower() or "margin" in ev.lower() for ev in calibration_hypotheses[0].evidence)


# =============================================================================
# Tests for full diagnosis integration
# =============================================================================

class TestFullDiagnosisIntegration:
    """Integration tests for full diagnosis with probability analysis."""

    def test_full_diagnosis_with_probability(self, calibrated_classifier, classification_data):
        """Test complete diagnosis workflow with probability-enabled model."""
        X_train, X_val, y_train, y_val = classification_data
        
        report = diagnose(
            estimator=calibrated_classifier,
            datasets={
                "train": (X_train, y_train),
                "val": (X_val, y_val)
            },
            task="classification"
        )
        
        # Verify report structure
        assert report is not None
        assert report.task == "classification"
        assert report.signals is not None
        
        # Probability signals should be populated
        assert report.signals.has_probability_predictions is True
        
        # Hypotheses and recommendations should be generated
        assert report.hypotheses is not None
        assert report.recommendations is not None

    def test_diagnosis_without_validation_set(self, calibrated_classifier, classification_data):
        """Test diagnosis when no validation set is provided (should use CV)."""
        X_train, _, y_train, _ = classification_data
        
        report = diagnose(
            estimator=calibrated_classifier,
            datasets={
                "train": (X_train, y_train)
            },
            task="classification"
        )
        
        assert report is not None
        assert report.signals.has_probability_predictions is True

    def test_signal_comparison_prob_vs_non_prob(self, calibrated_classifier, non_probability_classifier, classification_data):
        """Compare signals between probability and non-probability models."""
        X_train, X_val, y_train, y_val = classification_data
        
        # Probability model
        evidence_prob = gather_evidence(
            estimator=calibrated_classifier,
            datasets={
                "train": (X_train, y_train),
                "val": (X_val, y_val)
            },
            task="classification"
        )
        signals_prob = extract_signals(evidence_prob)
        
        # Non-probability model
        evidence_noprob = gather_evidence(
            estimator=non_probability_classifier,
            datasets={
                "train": (X_train, y_train),
                "val": (X_val, y_val)
            },
            task="classification"
        )
        signals_noprob = extract_signals(evidence_noprob)
        
        # Key difference: probability signals
        assert signals_prob.has_probability_predictions is True
        assert signals_noprob.has_probability_predictions is False
        
        # But core performance signals should exist for both
        assert signals_prob.train_score is not None
        assert signals_noprob.train_score is not None
        assert signals_prob.val_score is not None
        assert signals_noprob.val_score is not None


# =============================================================================
# Tests for threshold analysis
# =============================================================================

class TestThresholdAnalysis:
    """Tests for threshold analysis signals."""

    def test_threshold_signals_extracted(self, calibrated_classifier, classification_data):
        """Test that threshold-related signals are correctly extracted for binary classification."""
        X_train, X_val, y_train, y_val = classification_data
        
        evidence = gather_evidence(
            estimator=calibrated_classifier,
            datasets={
                "train": (X_train, y_train),
                "val": (X_val, y_val)
            },
            task="classification"
        )
        signals = extract_signals(evidence)
        
        # Threshold signals should be populated for binary classification
        assert signals.has_probability_predictions is True
        
        # Check threshold analysis signals
        assert signals.threshold_sensitivity_high is not None
        assert signals.threshold_sensitivity_low is not None
        assert signals.score_stability_index is not None
        assert signals.optimal_threshold_f1 is not None
        assert signals.optimal_threshold_precision is not None
        assert signals.optimal_threshold_recall is not None
        
        # Check that threshold sensitivity is a valid ratio
        assert 0.0 <= signals.threshold_sensitivity_high <= 1.0
        assert 0.0 <= signals.threshold_sensitivity_low <= 1.0
        
        # Score stability index should be positive
        assert signals.score_stability_index >= 0.0
        
        # Optimal thresholds should be between 0 and 1
        assert 0.0 < signals.optimal_threshold_f1 < 1.0
        assert 0.0 < signals.optimal_threshold_precision < 1.0
        assert 0.0 < signals.optimal_threshold_recall < 1.0

    def test_threshold_analysis_structure(self, calibrated_classifier, classification_data):
        """Test that threshold analysis dictionary has correct structure."""
        X_train, X_val, y_train, y_val = classification_data
        
        evidence = gather_evidence(
            estimator=calibrated_classifier,
            datasets={
                "train": (X_train, y_train),
                "val": (X_val, y_val)
            },
            task="classification"
        )
        signals = extract_signals(evidence)
        
        assert signals.threshold_analysis is not None
        assert isinstance(signals.threshold_analysis, dict)
        
        # Check required keys
        assert "default_threshold" in signals.threshold_analysis
        assert "default_metrics" in signals.threshold_analysis
        assert "optimal_metrics" in signals.threshold_analysis
        assert "threshold_range_analysis" in signals.threshold_analysis
        assert "stability_metrics" in signals.threshold_analysis
        
        # Check nested structures
        default_metrics = signals.threshold_analysis["default_metrics"]
        assert "precision" in default_metrics
        assert "recall" in default_metrics
        assert "f1" in default_metrics
        
        optimal_metrics = signals.threshold_analysis["optimal_metrics"]
        assert "best_f1" in optimal_metrics
        assert "best_precision" in optimal_metrics
        assert "best_recall" in optimal_metrics

    def test_threshold_optimization_detection(self, calibrated_classifier, classification_data):
        """Test that threshold optimization opportunities are detected."""
        X_train, X_val, y_train, y_val = classification_data
        
        evidence = gather_evidence(
            estimator=calibrated_classifier,
            datasets={
                "train": (X_train, y_train),
                "val": (X_val, y_val)
            },
            task="classification"
        )
        signals = extract_signals(evidence)
        
        # Verify that optimal F1 is at least as good as default F1
        if signals.threshold_analysis is not None:
            default_f1 = signals.threshold_analysis["default_metrics"]["f1"]
            best_f1 = signals.threshold_analysis["optimal_metrics"]["best_f1"]
            assert best_f1 >= default_f1 - 0.01  # Allow small margin for floating point errors


# =============================================================================
# Tests for cross-validation scenarios
# =============================================================================

class TestCrossValidationProbabilityDiagnosis:
    """Tests for probability diagnosis with cross-validation results."""

    def test_probability_signals_with_cv(self, calibrated_classifier, classification_data):
        """Test that probability signals are correctly extracted when using CV."""
        from sklearn.model_selection import cross_val_predict
        
        X_train, X_val, y_train, y_val = classification_data
        
        # Generate CV predictions with probabilities
        cv_proba = cross_val_predict(
            calibrated_classifier, X_train, y_train, cv=5, method="predict_proba"
        )
        cv_pred = np.argmax(cv_proba, axis=1)
        
        cv_results = {
            "test_score": np.array([0.85, 0.87, 0.83, 0.86, 0.84]),
            "predictions": cv_pred,
            "probabilities": cv_proba
        }
        
        evidence = gather_evidence(
            estimator=calibrated_classifier,
            datasets={
                "train": (X_train, y_train),
                "val": (X_val, y_val)
            },
            task="classification",
            cv_results=cv_results
        )
        signals = extract_signals(evidence)
        
        assert signals.has_probability_predictions is True
        assert signals.mean_prediction_confidence is not None
        assert signals.expected_calibration_error is not None

    def test_threshold_analysis_with_cv_only(self, calibrated_classifier, classification_data):
        """Test threshold analysis when only training data and CV results are available."""
        from sklearn.model_selection import cross_val_predict
        
        X_train, _, y_train, _ = classification_data
        
        # Generate CV predictions
        cv_proba = cross_val_predict(
            calibrated_classifier, X_train, y_train, cv=5, method="predict_proba"
        )
        cv_pred = np.argmax(cv_proba, axis=1)
        
        cv_results = {
            "test_score": np.array([0.85, 0.87, 0.83, 0.86, 0.84]),
            "predictions": cv_pred,
            "probabilities": cv_proba
        }
        
        # No validation set provided - should use training data for analysis
        evidence = gather_evidence(
            estimator=calibrated_classifier,
            datasets={
                "train": (X_train, y_train)
            },
            task="classification",
            cv_results=cv_results
        )
        signals = extract_signals(evidence)
        
        # Probability signals should be available from training data
        assert signals.has_probability_predictions is True
        assert signals.mean_prediction_confidence is not None
        
        # Threshold analysis should be performed on training data when no validation set
        assert signals.threshold_sensitivity_high is not None
        assert signals.score_stability_index is not None

    def test_diagnosis_with_cv_results(self, calibrated_classifier, classification_data):
        """Test full diagnosis workflow with CV results."""
        from sklearn.model_selection import cross_val_predict
        
        X_train, X_val, y_train, y_val = classification_data
        
        # Generate CV predictions
        cv_proba = cross_val_predict(
            calibrated_classifier, X_train, y_train, cv=5, method="predict_proba"
        )
        cv_pred = np.argmax(cv_proba, axis=1)
        
        cv_results = {
            "test_score": np.array([0.85, 0.87, 0.83, 0.86, 0.84]),
            "predictions": cv_pred,
            "probabilities": cv_proba
        }
        
        report = diagnose(
            estimator=calibrated_classifier,
            datasets={
                "train": (X_train, y_train),
                "val": (X_val, y_val)
            },
            task="classification",
            cv_results=cv_results
        )
        
        assert report is not None
        assert report.signals.has_probability_predictions is True
        assert report.hypotheses is not None
        assert report.recommendations is not None
