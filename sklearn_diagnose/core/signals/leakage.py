"""
Data leakage signal extractors.

Signals that may indicate data leakage or suspicious patterns
in the data.
"""

from typing import List

import numpy as np

from ._base import SignalExtractor, SignalResult, SignalCategory
from ..schemas import Evidence


class LeakageSignalExtractor(SignalExtractor):
    """
    Extractor for data leakage indicators.
    
    Identifies patterns that may indicate data leakage,
    such as suspiciously high correlations or unrealistic gaps.
    """
    
    def __init__(self):
        super().__init__("leakage", category=SignalCategory.LEAKAGE)
    
    def get_required_evidence(self) -> List[str]:
        return ["X_train", "y_train"]
    
    def extract(self, evidence: Evidence) -> List[SignalResult]:
        """Extract leakage indicator signals."""
        results = []
        
        # Feature-target leakage indicators
        results.extend(self._extract_feature_target_leakage(evidence))
        
        # CV vs holdout leakage indicators
        results.extend(self._extract_cv_holdout_leakage(evidence))
        
        # Perfect prediction indicators
        results.extend(self._extract_perfect_prediction_indicators(evidence))
        
        return results
    
    def _extract_feature_target_leakage(
        self, 
        evidence: Evidence
    ) -> List[SignalResult]:
        """Extract feature-target correlation leakage signals."""
        results = []
        
        try:
            X = evidence.X_train
            y = evidence.y_train.astype(float)
            
            suspicious_features = []
            very_suspicious_features = []
            
            for i in range(X.shape[1]):
                try:
                    corr = np.corrcoef(X[:, i], y)[0, 1]
                    if not np.isnan(corr):
                        abs_corr = abs(corr)
                        if abs_corr > 0.99:
                            very_suspicious_features.append({
                                "feature_index": i,
                                "correlation": float(corr),
                                "severity": "very_high"
                            })
                        elif abs_corr > 0.95:
                            suspicious_features.append({
                                "feature_index": i,
                                "correlation": float(corr),
                                "severity": "high"
                            })
                except Exception:
                    continue
            
            all_suspicious = very_suspicious_features + suspicious_features
            
            if all_suspicious:
                results.append(SignalResult(
                    name="suspicious_feature_correlations",
                    value=all_suspicious,
                    category=SignalCategory.LEAKAGE,
                    description="Features with suspiciously high target correlation",
                    metadata={
                        "n_suspicious": len(suspicious_features),
                        "n_very_suspicious": len(very_suspicious_features),
                        "threshold_high": 0.95,
                        "threshold_very_high": 0.99,
                        "may_indicate_leakage": len(all_suspicious) > 0,
                        "strong_leakage_indicator": len(very_suspicious_features) > 0
                    }
                ))
            
            # Count of features with correlation > 0.9
            high_corr_count = len(suspicious_features) + len(very_suspicious_features)
            results.append(SignalResult(
                name="high_feature_target_correlation_count",
                value=high_corr_count,
                category=SignalCategory.LEAKAGE,
                description="Number of features with |correlation| > 0.95",
                metadata={
                    "suspicious": high_corr_count > 0,
                    "interpretation": "potential_leakage" if high_corr_count > 0 else "normal"
                }
            ))
        
        except Exception:
            pass
        
        return results
    
    def _extract_cv_holdout_leakage(
        self, 
        evidence: Evidence
    ) -> List[SignalResult]:
        """Extract CV vs holdout discrepancy signals."""
        results = []
        
        if not evidence.has_cv_results or not evidence.has_validation_set:
            return results
        
        cv = evidence.cv_results
        if "test_score" not in cv:
            return results
        
        cv_scores = np.asarray(cv["test_score"])
        cv_mean = np.mean(cv_scores)
        
        # Compute holdout score
        from .performance import compute_score
        holdout_score = compute_score(
            evidence.y_val,
            evidence.y_pred_val,
            evidence.task
        )
        
        gap = cv_mean - holdout_score
        
        results.append(SignalResult(
            name="cv_holdout_gap",
            value=float(gap),
            category=SignalCategory.LEAKAGE,
            description="Difference between CV mean and holdout score",
            metadata={
                "cv_mean": float(cv_mean),
                "holdout_score": float(holdout_score),
                "gap_magnitude": abs(float(gap)),
                "interpretation": self._interpret_cv_holdout_gap(gap)
            }
        ))
        
        # Large discrepancy indicator
        if abs(gap) > 0.1:
            results.append(SignalResult(
                name="cv_holdout_discrepancy_alert",
                value=True,
                category=SignalCategory.LEAKAGE,
                description="Large discrepancy between CV and holdout performance",
                metadata={
                    "gap": float(gap),
                    "possible_causes": [
                        "data_leakage_in_cv",
                        "distribution_shift_between_splits",
                        "insufficient_validation_set_size"
                    ],
                    "recommendation": "investigate_data_splitting"
                }
            ))
        
        return results
    
    def _extract_perfect_prediction_indicators(
        self, 
        evidence: Evidence
    ) -> List[SignalResult]:
        """Extract signals for potentially perfect predictions."""
        results = []
        
        # Check training score
        if evidence.y_pred_train is not None:
            from .performance import compute_score
            train_score = compute_score(
                evidence.y_train,
                evidence.y_pred_train,
                evidence.task
            )
            
            is_perfect = train_score > 0.999
            is_near_perfect = train_score > 0.99
            
            if is_near_perfect:
                results.append(SignalResult(
                    name="near_perfect_training_score",
                    value=float(train_score),
                    category=SignalCategory.LEAKAGE,
                    description="Training score is suspiciously high",
                    metadata={
                        "is_perfect": is_perfect,
                        "is_near_perfect": is_near_perfect,
                        "possible_causes": [
                            "severe_overfitting",
                            "data_leakage",
                            "label_in_features",
                            "too_simple_dataset"
                        ],
                        "recommendation": "check_for_leakage_or_overfitting"
                    }
                ))
        
        # Check validation score if also suspiciously high
        if evidence.y_pred_val is not None and evidence.has_validation_set:
            from .performance import compute_score
            val_score = compute_score(
                evidence.y_val,
                evidence.y_pred_val,
                evidence.task
            )
            
            if val_score > 0.99:
                results.append(SignalResult(
                    name="near_perfect_validation_score",
                    value=float(val_score),
                    category=SignalCategory.LEAKAGE,
                    description="Validation score is suspiciously high",
                    metadata={
                        "score": float(val_score),
                        "possible_causes": [
                            "data_leakage",
                            "too_simple_dataset",
                            "test_set_contamination"
                        ],
                        "recommendation": "verify_data_integrity"
                    }
                ))
        
        return results
    
    def _interpret_cv_holdout_gap(self, gap: float) -> str:
        """Interpret CV vs holdout gap."""
        abs_gap = abs(gap)
        if abs_gap > 0.15:
            return "very_large_discrepancy"
        elif abs_gap > 0.1:
            return "large_discrepancy"
        elif abs_gap > 0.05:
            return "moderate_discrepancy"
        else:
            return "consistent"


class PreprocessingLeakageDetector(SignalExtractor):
    """
    Detector for preprocessing-related leakage.
    
    Identifies potential leakage from improper preprocessing
    (e.g., fitting on full data before CV).
    """
    
    def __init__(self):
        super().__init__("preprocessing_leakage", category=SignalCategory.LEAKAGE)
    
    def get_required_evidence(self) -> List[str]:
        return ["X_train", "y_train", "cv_results"]
    
    def extract(self, evidence: Evidence) -> SignalResult:
        """Extract preprocessing leakage signals."""
        # This is a heuristic detector based on patterns
        # that suggest improper preprocessing
        
        indicators = []
        
        # Check for extremely low variance in CV scores
        if evidence.has_cv_results and "test_score" in evidence.cv_results:
            cv_scores = np.asarray(evidence.cv_results["test_score"])
            cv_std = np.std(cv_scores)
            cv_mean = np.mean(cv_scores)
            
            if cv_mean > 0 and cv_std / cv_mean < 0.01:
                indicators.append({
                    "type": "suspiciously_stable_cv",
                    "description": "CV scores have suspiciously low variance",
                    "cv_std": float(cv_std),
                    "cv_mean": float(cv_mean),
                    "cv_coefficient": float(cv_std / cv_mean)
                })
        
        # Check for perfect separation in training
        if evidence.y_pred_train is not None:
            from .performance import compute_score
            train_score = compute_score(
                evidence.y_train,
                evidence.y_pred_train,
                evidence.task
            )
            
            if train_score > 0.999:
                indicators.append({
                    "type": "perfect_training_score",
                    "description": "Model achieves near-perfect training score",
                    "score": float(train_score)
                })
        
        return SignalResult(
            name="preprocessing_leakage_indicators",
            value=indicators,
            category=SignalCategory.LEAKAGE,
            description="Indicators of potential preprocessing leakage",
            metadata={
                "n_indicators": len(indicators),
                "has_leakage_risk": len(indicators) > 0,
                "recommendation": (
                    "review_preprocessing_pipeline" 
                    if indicators else "no_concerns"
                )
            }
        )
