#!/usr/bin/env python
"""
Demo script for sklearn-diagnose probability prediction diagnosis capabilities.

This script demonstrates the enhanced probability diagnosis features including:
- Confidence distribution analysis
- Class separation metrics
- Calibration error (ECE)
- Threshold sensitivity analysis
- Optimal threshold recommendations
"""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_predict

# Use rule-based hypothesis generation (no LLM required)
from sklearn_diagnose.core.evidence import collect_evidence as gather_evidence
from sklearn_diagnose.core.signals import extract_all_signals as extract_signals
from sklearn_diagnose.core.hypotheses import generate_hypotheses as generate_rule_based_hypotheses
from sklearn_diagnose.core.recommendations import get_example_recommendations_for_failure_mode
from sklearn_diagnose.core.schemas import DiagnosisReport, TaskType


def diagnose_without_llm(estimator, datasets, task="classification", cv_results=None):
    """Run diagnosis without LLM using rule-based hypothesis generation."""
    
    # Step 1: Gather evidence
    evidence = gather_evidence(
        estimator=estimator,
        datasets=datasets,
        task=task,
        cv_results=cv_results
    )
    
    # Step 2: Extract signals
    signals = extract_signals(evidence)
    
    # Step 3: Generate rule-based hypotheses
    task_type = TaskType(task)
    hypotheses = generate_rule_based_hypotheses(signals, task_type)
    
    # Step 4: Generate recommendations
    recommendations = []
    seen_actions = set()
    
    for hyp in hypotheses:
        if hyp.is_actionable:
            examples = get_example_recommendations_for_failure_mode(hyp.name)
            for rec_example in examples[:2]:  # Take up to 2 recommendations per hypothesis
                action = rec_example["action"]
                if action not in seen_actions:
                    from sklearn_diagnose.core.schemas import Recommendation
                    recommendations.append(Recommendation(
                        action=action,
                        rationale=rec_example["rationale"],
                        related_hypothesis=hyp.name
                    ))
                    seen_actions.add(action)
    
    # Sort recommendations by related hypothesis confidence
    recommendations.sort(
        key=lambda r: next((h.confidence for h in hypotheses if h.name == r.related_hypothesis), 0),
        reverse=True
    )
    
    # Create report
    from sklearn.pipeline import Pipeline
    report = DiagnosisReport(
        hypotheses=hypotheses,
        recommendations=recommendations,
        signals=signals,
        task=task_type,
        estimator_type=type(estimator).__name__,
        has_pipeline=isinstance(estimator, Pipeline)
    )
    
    return report


def demo_probability_diagnosis():
    """Demonstrate probability prediction diagnosis with various models."""
    
    # Create synthetic classification data
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        weights=[0.7, 0.3],  # Slight imbalance
        random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("=" * 70)
    print("sklearn-diagnose: Probability Prediction Diagnosis Demo")
    print("=" * 70)
    
    # Test 1: Well-calibrated model (Logistic Regression)
    print("\n" + "-" * 70)
    print("Test 1: Logistic Regression (well-calibrated)")
    print("-" * 70)
    
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_train, y_train)
    
    report = diagnose_without_llm(
        estimator=clf,
        datasets={
            "train": (X_train, y_train),
            "val": (X_val, y_val)
        },
        task="classification"
    )
    
    print_diagnosis_report(report)
    
    # Test 2: Pipeline with preprocessing
    print("\n" + "-" * 70)
    print("Test 2: Pipeline with StandardScaler + Logistic Regression")
    print("-" * 70)
    
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(random_state=42, max_iter=1000))
    ])
    pipe.fit(X_train, y_train)
    
    report_pipe = diagnose_without_llm(
        estimator=pipe,
        datasets={
            "train": (X_train, y_train),
            "val": (X_val, y_val)
        },
        task="classification"
    )
    
    print_diagnosis_report(report_pipe)
    
    # Test 3: With cross-validation results
    print("\n" + "-" * 70)
    print("Test 3: Diagnosis with Cross-Validation Results")
    print("-" * 70)
    
    # Generate CV predictions
    cv_proba = cross_val_predict(
        clf, X_train, y_train, cv=5, method="predict_proba"
    )
    cv_pred = (cv_proba[:, 1] >= 0.5).astype(int)
    
    cv_results = {
        "test_score": [0.85, 0.87, 0.83, 0.86, 0.84],
        "predictions": cv_pred,
        "probabilities": cv_proba
    }
    
    report_cv = diagnose_without_llm(
        estimator=clf,
        datasets={
            "train": (X_train, y_train),
            "val": (X_val, y_val)
        },
        task="classification",
        cv_results=cv_results
    )
    
    print_diagnosis_report(report_cv)
    
    # Test 4: Non-probability model (graceful degradation)
    print("\n" + "-" * 70)
    print("Test 4: Non-probability model (SVM without predict_proba)")
    print("-" * 70)
    
    svm_no_proba = SVC(probability=False, random_state=42)
    svm_no_proba.fit(X_train, y_train)
    
    report_no_proba = diagnose_without_llm(
        estimator=svm_no_proba,
        datasets={
            "train": (X_train, y_train),
            "val": (X_val, y_val)
        },
        task="classification"
    )
    
    print_diagnosis_report(report_no_proba)
    
    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)


def print_diagnosis_report(report):
    """Print a formatted diagnosis report."""
    
    print(f"Model type: {report.estimator_type}")
    print(f"Has pipeline: {report.has_pipeline}")
    print(f"Has probability predictions: {report.signals.has_probability_predictions}")
    
    if report.signals.has_probability_predictions:
        print("\nProbability Quality Metrics:")
        metrics = [
            ("Mean prediction confidence", report.signals.mean_prediction_confidence, ".2f"),
            ("Median prediction confidence", report.signals.median_prediction_confidence, ".2f"),
            ("Class separation score", report.signals.class_separation_score, ".2f"),
            ("Expected Calibration Error", report.signals.expected_calibration_error, ".1%"),
            ("Overconfident predictions", report.signals.overconfidence_ratio, ".1%"),
            ("Underconfident predictions", report.signals.underconfidence_ratio, ".1%"),
            ("Low confidence predictions (<70%)", report.signals.low_confidence_ratio, ".1%"),
        ]
        
        for name, value, fmt in metrics:
            if value is not None:
                print(f"  - {name}: {value:{fmt}}")
        
        # Threshold analysis
        if report.signals.threshold_sensitivity_high is not None:
            print("\nThreshold Analysis:")
            threshold_metrics = [
                ("Threshold sensitivity (±0.1)", report.signals.threshold_sensitivity_high, ".1%"),
                ("Score stability index", report.signals.score_stability_index, ".2f"),
                ("Optimal F1 threshold", report.signals.optimal_threshold_f1, ".2f"),
                ("Optimal precision threshold", report.signals.optimal_threshold_precision, ".2f"),
                ("Optimal recall threshold", report.signals.optimal_threshold_recall, ".2f"),
            ]
            
            for name, value, fmt in threshold_metrics:
                if value is not None:
                    print(f"  - {name}: {value:{fmt}}")
    
    # Hypotheses
    print("\nDetected Issues:")
    actionable_hypotheses = [h for h in report.hypotheses if h.is_actionable]
    
    if not actionable_hypotheses:
        print("  - No significant issues detected")
    else:
        for hyp in sorted(actionable_hypotheses, key=lambda h: h.confidence, reverse=True):
            print(f"  - {hyp.name.value.replace('_', ' ').title()}: "
                  f"confidence={hyp.confidence:.2f}, severity={hyp.severity}")
            for evidence in hyp.evidence[:2]:  # Show top 2 evidence
                print(f"    * {evidence}")
    
    # Top recommendations
    print("\nTop Recommendations:")
    for i, rec in enumerate(report.recommendations[:3], 1):
        print(f"  {i}. {rec.action}")
        if rec.rationale:
            print(f"     Rationale: {rec.rationale}")


if __name__ == "__main__":
    demo_probability_diagnosis()