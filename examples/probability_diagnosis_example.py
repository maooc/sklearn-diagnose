"""
Example: Probability Prediction Diagnosis

This example demonstrates the probability prediction diagnosis capabilities
added to sklearn-diagnose. It shows how to:
1. Train a classifier with probability outputs
2. Run diagnosis with probability signal extraction
3. Inspect probability-based signals and failure modes
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn_diagnose import setup_llm, diagnose
from sklearn_diagnose.core import FailureMode


def mock_llm_for_example():
    """Set up a mock LLM for this example (no API key needed)."""
    import sys
    sys.path.insert(0, '/Users/wen/workspace/字节项目/dogfood/dogfood-2-609-sklearn-diagnose/03-kimi/sklearn-diagnose')
    from sklearn_diagnose.llm.client import _set_global_client
    from tests.conftest import MockLLMClient
    _set_global_client(MockLLMClient())


def example_basic_probability_diagnosis():
    """Basic example with LogisticRegression showing probability signals."""
    print("=" * 70)
    print("Example 1: Basic Probability Diagnosis with LogisticRegression")
    print("=" * 70)

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

    # Display probability signals
    s = report.signals
    print("\nProbability Prediction Signals:")
    print(f"  Has probability outputs: {s.has_probability_outputs}")
    print(f"  Mean predicted probability: {s.proba_mean:.3f}")
    print(f"  Std of predicted probabilities: {s.proba_std:.3f}")
    print(f"  Prediction entropy: {s.proba_entropy:.3f}")
    print(f"  High confidence ratio (>0.9): {s.high_confidence_ratio:.3f}")
    print(f"  Low confidence ratio (<0.6): {s.low_confidence_ratio:.3f}")
    print(f"  Mean margin (top-2 classes): {s.proba_margin_mean:.3f}")
    print(f"  Ambiguous predictions ratio: {s.ambiguous_predictions_ratio:.3f}")
    print(f"  AUC-ROC: {s.auc_roc:.3f}")
    print(f"  AUC-PR: {s.auc_pr:.3f}")
    print(f"  Optimal threshold: {s.optimal_threshold:.3f}")
    print(f"  Calibration error: {s.proba_calibration_error:.3f}")

    print("\nPer-class Probability Analysis:")
    for cls, mean_proba in s.per_class_proba_mean.items():
        std_proba = s.per_class_proba_std[cls]
        print(f"  Class {cls}: mean={mean_proba:.3f}, std={std_proba:.3f}")

    print("\nDetected Issues:")
    for h in report.hypotheses:
        print(f"  - {h.name.value}: {h.confidence:.1%} confidence ({h.severity})")


def example_multiclass_probability():
    """Example with multiclass classification."""
    print("\n" + "=" * 70)
    print("Example 2: Multiclass Probability Diagnosis")
    print("=" * 70)

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

    s = report.signals
    print("\nMulticlass Probability Signals:")
    print(f"  Has probability outputs: {s.has_probability_outputs}")
    print(f"  Mean predicted probability: {s.proba_mean:.3f}")
    print(f"  Prediction entropy: {s.proba_entropy:.3f}")

    # Note: Binary-specific signals are None for multiclass
    print(f"  AUC-ROC (binary only): {s.auc_roc}")
    print(f"  Optimal threshold (binary only): {s.optimal_threshold}")

    print("\nPer-class Probability Analysis:")
    for cls, mean_proba in s.per_class_proba_mean.items():
        std_proba = s.per_class_proba_std[cls]
        print(f"  Class {cls}: mean={mean_proba:.3f}, std={std_proba:.3f}")


def example_pipeline_with_probability():
    """Example with Pipeline showing probability signals work through preprocessing."""
    print("\n" + "=" * 70)
    print("Example 3: Pipeline with Probability Diagnosis")
    print("=" * 70)

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

    s = report.signals
    print("\nPipeline Probability Signals:")
    print(f"  Has probability outputs: {s.has_probability_outputs}")
    print(f"  Mean predicted probability: {s.proba_mean:.3f}")
    print(f"  AUC-ROC: {s.auc_roc:.3f}")


def example_no_probability_support():
    """Example showing graceful degradation for classifiers without predict_proba."""
    print("\n" + "=" * 70)
    print("Example 4: Classifier Without Probability Support (SVC)")
    print("=" * 70)

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

    print("\nSignals for Classifier Without predict_proba:")
    print(f"  Has probability outputs: {report.signals.has_probability_outputs}")
    print(f"  proba_mean: {report.signals.proba_mean}")
    print(f"  auc_roc: {report.signals.auc_roc}")
    print("\nNote: Basic signals (train_score, val_score) still work:")
    print(f"  Train score: {report.signals.train_score:.3f}")
    print(f"  Val score: {report.signals.val_score:.3f}")


def example_calibration_detection():
    """Example showing calibration error detection with RandomForest."""
    print("\n" + "=" * 70)
    print("Example 5: Calibration Error Detection (RandomForest)")
    print("=" * 70)

    # Generate data
    X, y = make_classification(
        n_samples=500, n_features=20, n_informative=10,
        n_redundant=5, n_classes=2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # RandomForest often has calibration issues
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    # Diagnose
    report = diagnose(
        estimator=model,
        datasets={"train": (X_train, y_train), "val": (X_val, y_val)},
        task="classification"
    )

    s = report.signals
    print("\nCalibration Signals:")
    print(f"  Calibration error: {s.proba_calibration_error:.3f}")
    print(f"  Mean predicted probability: {s.proba_mean:.3f}")

    print("\nDetected Issues (may include POOR_CALIBRATION):")
    for h in report.hypotheses:
        print(f"  - {h.name.value}: {h.confidence:.1%} confidence")
        if h.name == FailureMode.POOR_CALIBRATION:
            for ev in h.evidence:
                print(f"      {ev}")


if __name__ == "__main__":
    # Set up mock LLM for examples
    mock_llm_for_example()

    # Run examples
    example_basic_probability_diagnosis()
    example_multiclass_probability()
    example_pipeline_with_probability()
    example_no_probability_support()
    example_calibration_detection()

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
