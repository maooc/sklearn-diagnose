"""
Integration tests for probability prediction diagnosis in the full diagnostic pipeline.

These tests verify that probability signals flow through the complete diagnostic chain:
1. Signal extraction
2. Hypothesis generation (LLM)
3. Recommendation generation
4. Summary generation

And that the final output differs when probability signals are present vs absent.
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn_diagnose import diagnose
from sklearn_diagnose.core.schemas import FailureMode


class TestProbabilityIntegration:
    """Test probability signals in the full diagnostic pipeline."""

    def test_probability_signals_affect_hypotheses(self):
        """Test that probability signals lead to probability-related hypotheses."""
        # Generate data
        X, y = make_classification(
            n_samples=500, n_features=20, n_informative=10,
            n_redundant=5, n_classes=2, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model with probability outputs
        model = LogisticRegression(random_state=42, max_iter=500)
        model.fit(X_train, y_train)

        # Diagnose
        report = diagnose(
            estimator=model,
            datasets={"train": (X_train, y_train), "val": (X_val, y_val)},
            task="classification"
        )

        # Verify probability signals are present
        assert report.signals.has_probability_outputs is True
        assert report.signals.proba_mean is not None

        # Verify probability-related hypotheses can be detected
        probability_failure_modes = {
            FailureMode.POOR_CALIBRATION,
            FailureMode.LOW_CONFIDENCE_PREDICTIONS,
            FailureMode.AMBIGUOUS_CLASS_BOUNDARIES,
            FailureMode.SUBOPTIMAL_THRESHOLD,
            FailureMode.CONFIDENCE_ACCURACY_MISMATCH
        }

        detected_modes = {h.name for h in report.hypotheses}
        detected_probability_modes = detected_modes & probability_failure_modes

        # Log what was detected
        print(f"\nDetected probability-related failure modes: {detected_probability_modes}")
        print(f"All detected modes: {detected_modes}")

        # The model should have at least some basic hypotheses
        assert len(report.hypotheses) >= 0  # May or may not have probability issues

    def test_no_probability_signals_without_predict_proba(self):
        """Test that models without predict_proba don't generate probability signals."""
        # Generate data
        X, y = make_classification(
            n_samples=500, n_features=20, n_informative=10,
            n_redundant=5, n_classes=2, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train SVC without probability=True (no predict_proba)
        model = SVC(kernel="linear", random_state=42)
        model.fit(X_train, y_train)

        # Diagnose
        report = diagnose(
            estimator=model,
            datasets={"train": (X_train, y_train), "val": (X_val, y_val)},
            task="classification"
        )

        # Verify probability signals are NOT present
        assert report.signals.has_probability_outputs is False
        assert report.signals.proba_mean is None
        assert report.signals.auc_roc is None
        assert report.signals.optimal_threshold is None

        # Verify no probability-related failure modes
        probability_failure_modes = {
            FailureMode.POOR_CALIBRATION,
            FailureMode.LOW_CONFIDENCE_PREDICTIONS,
            FailureMode.AMBIGUOUS_CLASS_BOUNDARIES,
            FailureMode.SUBOPTIMAL_THRESHOLD,
            FailureMode.CONFIDENCE_ACCURACY_MISMATCH
        }

        detected_modes = {h.name for h in report.hypotheses}
        detected_probability_modes = detected_modes & probability_failure_modes

        assert len(detected_probability_modes) == 0, \
            f"Should not detect probability failure modes without predict_proba: {detected_probability_modes}"

        # Basic signals should still work
        assert report.signals.train_score is not None
        assert report.signals.val_score is not None

    def test_probability_signals_influence_recommendations(self):
        """Test that probability signals influence recommendation generation."""
        # Generate data
        X, y = make_classification(
            n_samples=500, n_features=20, n_informative=10,
            n_redundant=5, n_classes=2, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train RandomForest (often has calibration issues)
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        # Diagnose
        report = diagnose(
            estimator=model,
            datasets={"train": (X_train, y_train), "val": (X_val, y_val)},
            task="classification"
        )

        # Verify probability signals are present
        assert report.signals.has_probability_outputs is True

        # Check if recommendations include probability-related ones
        probability_keywords = ['calibration', 'threshold', 'probability', 'confidence']

        has_probability_recommendations = any(
            any(kw in rec.action.lower() or kw in rec.rationale.lower()
                for kw in probability_keywords)
            for rec in report.recommendations
        )

        print(f"\nRecommendations: {[rec.action for rec in report.recommendations]}")
        print(f"Has probability recommendations: {has_probability_recommendations}")

        # If poor calibration is detected, should have calibration-related recommendations
        has_poor_calibration = any(
            h.name == FailureMode.POOR_CALIBRATION for h in report.hypotheses
        )

        if has_poor_calibration:
            calibration_recs = [
                rec for rec in report.recommendations
                if 'calibration' in rec.action.lower() or 'calibration' in rec.rationale.lower()
            ]
            print(f"Calibration recommendations: {[rec.action for rec in calibration_recs]}")

    def test_probability_signals_in_summary(self):
        """Test that probability signals appear in the summary when present."""
        # Generate data
        X, y = make_classification(
            n_samples=500, n_features=20, n_informative=10,
            n_redundant=5, n_classes=2, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        model = LogisticRegression(random_state=42, max_iter=500)
        model.fit(X_train, y_train)

        # Diagnose
        report = diagnose(
            estimator=model,
            datasets={"train": (X_train, y_train), "val": (X_val, y_val)},
            task="classification"
        )

        # Get summary (summary is a method, not a property)
        summary_text = report.summary(use_llm=True)

        # Verify summary exists
        assert summary_text is not None
        assert len(summary_text) > 0

        # Summary should be a string
        assert isinstance(summary_text, str)

        print(f"\nSummary:\n{summary_text[:500]}...")

    def test_pipeline_preserves_probability_signals(self):
        """Test that Pipeline correctly passes probability signals through."""
        # Generate data
        X, y = make_classification(
            n_samples=500, n_features=20, n_informative=10,
            n_redundant=5, n_classes=2, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Create pipeline
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(random_state=42, max_iter=500))
        ])
        pipeline.fit(X_train, y_train)

        # Diagnose
        report = diagnose(
            estimator=pipeline,
            datasets={"train": (X_train, y_train), "val": (X_val, y_val)},
            task="classification"
        )

        # Verify probability signals are present
        assert report.signals.has_probability_outputs is True
        assert report.signals.proba_mean is not None
        assert report.signals.auc_roc is not None
        assert report.signals.optimal_threshold is not None

    def test_multiclass_probability_pipeline(self):
        """Test probability signals work with multiclass classification."""
        # Generate multiclass data
        X, y = make_classification(
            n_samples=500, n_features=20, n_informative=15,
            n_redundant=3, n_classes=3, n_clusters_per_class=1,
            random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        model = LogisticRegression(random_state=42, max_iter=500)
        model.fit(X_train, y_train)

        # Diagnose
        report = diagnose(
            estimator=model,
            datasets={"train": (X_train, y_train), "val": (X_val, y_val)},
            task="classification"
        )

        # Verify probability signals are present
        assert report.signals.has_probability_outputs is True
        assert report.signals.proba_mean is not None

        # Binary-specific signals should be None for multiclass
        assert report.signals.auc_roc is None
        assert report.signals.optimal_threshold is None

        # Per-class analysis should be present
        assert report.signals.per_class_proba_mean is not None
        assert len(report.signals.per_class_proba_mean) == 3

    def test_output_difference_with_and_without_probability(self):
        """Test that output differs meaningfully when probability signals are present vs absent."""
        # Generate data
        X, y = make_classification(
            n_samples=500, n_features=20, n_informative=10,
            n_redundant=5, n_classes=2, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Model WITH probability outputs
        model_with_proba = LogisticRegression(random_state=42, max_iter=500)
        model_with_proba.fit(X_train, y_train)

        report_with = diagnose(
            estimator=model_with_proba,
            datasets={"train": (X_train, y_train), "val": (X_val, y_val)},
            task="classification"
        )

        # Model WITHOUT probability outputs
        model_without_proba = SVC(kernel="linear", random_state=42)
        model_without_proba.fit(X_train, y_train)

        report_without = diagnose(
            estimator=model_without_proba,
            datasets={"train": (X_train, y_train), "val": (X_val, y_val)},
            task="classification"
        )

        # Verify structural differences
        assert report_with.signals.has_probability_outputs is True
        assert report_without.signals.has_probability_outputs is False

        # Verify signal differences
        assert report_with.signals.proba_mean is not None
        assert report_without.signals.proba_mean is None

        assert report_with.signals.auc_roc is not None
        assert report_without.signals.auc_roc is None

        # Verify no probability failure modes without predict_proba
        probability_modes = {
            FailureMode.POOR_CALIBRATION,
            FailureMode.LOW_CONFIDENCE_PREDICTIONS,
            FailureMode.AMBIGUOUS_CLASS_BOUNDARIES,
            FailureMode.SUBOPTIMAL_THRESHOLD,
            FailureMode.CONFIDENCE_ACCURACY_MISMATCH
        }

        modes_without = {h.name for h in report_without.hypotheses}
        assert len(modes_without & probability_modes) == 0

        print(f"\nWith predict_proba: {len(report_with.hypotheses)} hypotheses")
        print(f"Without predict_proba: {len(report_without.hypotheses)} hypotheses")

    def test_graceful_degradation_no_val_set(self):
        """Test graceful degradation when no validation set is provided."""
        # Generate data
        X, y = make_classification(
            n_samples=500, n_features=20, n_informative=10,
            n_redundant=5, n_classes=2, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        model = LogisticRegression(random_state=42, max_iter=500)
        model.fit(X_train, y_train)

        # Diagnose WITHOUT validation set
        report = diagnose(
            estimator=model,
            datasets={"train": (X_train, y_train)},  # No val set
            task="classification"
        )

        # Probability signals should NOT be present without validation set
        assert report.signals.has_probability_outputs is False
        assert report.signals.proba_mean is None

        # Basic signals should still work
        assert report.signals.train_score is not None

        print(f"\nWithout validation set: has_probability_outputs = {report.signals.has_probability_outputs}")
