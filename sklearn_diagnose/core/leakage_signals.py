"""
Leakage detection signal extractors.

This module contains signal extractors that identify potential data leakage
patterns, including suspicious feature correlations, CV-holdout gaps,
and other leakage indicators.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from .base import BaseSignalExtractor, signal_registry
from .schemas import Evidence, SignalCategory, SignalResult, TaskType


@signal_registry.register_class(
    name="leakage_indicators",
    category=SignalCategory.LEAKAGE,
    tags=["leakage", "detection", "suspicious"],
)
class LeakageIndicatorsExtractor(BaseSignalExtractor):
    """
    Extract leakage detection signals.
    
    This extractor analyzes various signals that may indicate data leakage,
    including:
    - Suspiciously high feature-target correlations
    - CV vs holdout performance gaps
    - Other suspicious patterns
    """
    
    category = SignalCategory.LEAKAGE
    
    def extract(
        self,
        evidence: Evidence,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[SignalResult]:
        """Extract leakage detection signals."""
        results = []
        
        params = params or {}
        corr_threshold = params.get("correlation_threshold", 0.95)
        gap_threshold = params.get("gap_threshold", 0.1)
        
        # 1. Check for suspicious feature-target correlations
        X = evidence.X_train
        if len(X.shape) == 2:
            try:
                y = evidence.y_train.astype(float)
                suspicious = []
                
                for i in range(X.shape[1]):
                    feature = X[:, i]
                    if np.std(feature) < 1e-10:
                        continue
                        
                    corr = np.corrcoef(feature, y)[0, 1]
                    if not np.isnan(corr) and abs(corr) > corr_threshold:
                        feature_name = evidence.feature_names[i] if evidence.feature_names else i
                        suspicious.append((feature_name, float(corr)))
                
                if suspicious:
                    suspicious_sorted = sorted(
                        suspicious,
                        key=lambda x: abs(x[1]),
                        reverse=True
                    )
                    
                    results.append(SignalResult(
                        name="suspicious_feature_correlations",
                        value=suspicious_sorted,
                        category=SignalCategory.LEAKAGE,
                        description=f"Features with suspiciously high correlation to target (> {corr_threshold})",
                        metadata={
                            "threshold": corr_threshold,
                            "n_features": X.shape[1],
                            "n_suspicious": len(suspicious_sorted),
                        },
                    ))
                    
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Error checking feature correlations: {e}")
        
        # 2. Check CV vs holdout gap (already in CV extractors but replicated here for convenience)
        if evidence.has_cv_results and evidence.has_validation_set:
            cv = evidence.cv_results
            if "test_score" in cv and evidence.y_pred_val is not None:
                try:
                    test_scores = np.asarray(cv["test_score"])
                    cv_mean = float(np.mean(test_scores))
                    
                    # Compute validation score
                    from .score_based_signals import compute_score
                    val_score = compute_score(
                        evidence.y_val,
                        evidence.y_pred_val,
                        evidence.task,
                        "default"
                    )
                    
                    cv_holdout_gap = cv_mean - val_score
                    
                    if abs(cv_holdout_gap) > gap_threshold:
                        results.append(SignalResult(
                            name="significant_cv_holdout_gap",
                            value=cv_holdout_gap,
                            category=SignalCategory.LEAKAGE,
                            description=f"Significant gap between CV and holdout scores (> {gap_threshold})",
                            metadata={
                                "cv_mean": cv_mean,
                                "val_score": val_score,
                                "threshold": gap_threshold,
                            },
                        ))
                        
                except Exception as e:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Error checking CV-holdout gap: {e}")
        
        # 3. Check for suspiciously high validation performance (near-perfect)
        if evidence.has_validation_set and evidence.y_pred_val is not None:
            try:
                from .score_based_signals import compute_score
                val_score = compute_score(
                    evidence.y_val,
                    evidence.y_pred_val,
                    evidence.task,
                    "default"
                )
                
                perfect_threshold = params.get("perfect_threshold", 0.99)
                if evidence.task == TaskType.CLASSIFICATION and val_score > perfect_threshold:
                    results.append(SignalResult(
                        name="near_perfect_validation",
                        value=val_score,
                        category=SignalCategory.LEAKAGE,
                        description=f"Near-perfect validation score (> {perfect_threshold}) - potential leakage",
                        metadata={
                            "score": val_score,
                            "threshold": perfect_threshold,
                            "task": evidence.task.value,
                        },
                    ))
                    
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Error checking perfect score: {e}")
        
        return results


@signal_registry.register_class(
    name="leakage_risk_assessment",
    category=SignalCategory.LEAKAGE,
    tags=["leakage", "risk", "assessment"],
)
class LeakageRiskAssessmentExtractor(BaseSignalExtractor):
    """
    Comprehensive leakage risk assessment.
    
    This extractor combines multiple leakage signals to provide
    an overall risk assessment for data leakage.
    """
    
    category = SignalCategory.LEAKAGE
    
    def extract(
        self,
        evidence: Evidence,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[SignalResult]:
        """Extract comprehensive leakage risk assessment."""
        results = []
        
        risk_score = 0.0
        risk_factors = []
        params = params or {}
        
        # 1. Check feature-target correlations
        X = evidence.X_train
        if len(X.shape) == 2:
            try:
                y = evidence.y_train.astype(float)
                max_corr = 0.0
                
                for i in range(X.shape[1]):
                    feature = X[:, i]
                    if np.std(feature) < 1e-10:
                        continue
                    corr = np.corrcoef(feature, y)[0, 1]
                    if not np.isnan(corr):
                        max_corr = max(max_corr, abs(corr))
                
                # Risk based on maximum correlation
                if max_corr > 0.95:
                    risk_score += 0.4
                    risk_factors.append({
                        "factor": "extreme_feature_correlation",
                        "value": float(max_corr),
                        "contribution": 0.4,
                        "description": "Feature with > 0.95 correlation to target"
                    })
                elif max_corr > 0.9:
                    risk_score += 0.2
                    risk_factors.append({
                        "factor": "high_feature_correlation",
                        "value": float(max_corr),
                        "contribution": 0.2,
                        "description": "Feature with > 0.9 correlation to target"
                    })
                    
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Error in correlation risk assessment: {e}")
        
        # 2. Check CV-holdout gap
        if evidence.has_cv_results and evidence.has_validation_set:
            cv = evidence.cv_results
            if "test_score" in cv and evidence.y_pred_val is not None:
                try:
                    test_scores = np.asarray(cv["test_score"])
                    cv_mean = float(np.mean(test_scores))
                    
                    from .score_based_signals import compute_score
                    val_score = compute_score(
                        evidence.y_val,
                        evidence.y_pred_val,
                        evidence.task,
                        "default"
                    )
                    
                    cv_holdout_gap = cv_mean - val_score
                    abs_gap = abs(cv_holdout_gap)
                    
                    if abs_gap > 0.15:
                        risk_score += 0.3
                        risk_factors.append({
                            "factor": "large_cv_holdout_gap",
                            "value": float(abs_gap),
                            "contribution": 0.3,
                            "description": "CV-holdout gap > 0.15"
                        })
                    elif abs_gap > 0.1:
                        risk_score += 0.15
                        risk_factors.append({
                            "factor": "moderate_cv_holdout_gap",
                            "value": float(abs_gap),
                            "contribution": 0.15,
                            "description": "CV-holdout gap > 0.1"
                        })
                        
                except Exception as e:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Error in CV gap assessment: {e}")
        
        # 3. Check suspicious validation performance
        if evidence.has_validation_set and evidence.y_pred_val is not None:
            try:
                from .score_based_signals import compute_score
                val_score = compute_score(
                    evidence.y_val,
                    evidence.y_pred_val,
                    evidence.task,
                    "default"
                )
                
                if evidence.task == TaskType.CLASSIFICATION:
                    if val_score > 0.99:
                        risk_score += 0.2
                        risk_factors.append({
                            "factor": "near_perfect_score",
                            "value": float(val_score),
                            "contribution": 0.2,
                            "description": "Near-perfect classification score (> 0.99)"
                        })
                    elif val_score > 0.95:
                        risk_score += 0.1
                        risk_factors.append({
                            "factor": "very_high_score",
                            "value": float(val_score),
                            "contribution": 0.1,
                            "description": "Very high classification score (> 0.95)"
                        })
                        
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Error in score assessment: {e}")
        
        # Determine risk level
        risk_score = min(risk_score, 1.0)  # Cap at 1.0
        
        if risk_score >= 0.7:
            risk_level = "high"
        elif risk_score >= 0.4:
            risk_level = "medium"
        elif risk_score >= 0.2:
            risk_level = "low"
        else:
            risk_level = "minimal"
        
        results.append(SignalResult(
            name="leakage_risk_score",
            value=risk_score,
            category=SignalCategory.LEAKAGE,
            description="Overall leakage risk score (0.0 to 1.0)",
            metadata={
                "risk_level": risk_level,
                "n_risk_factors": len(risk_factors),
            },
        ))
        
        if risk_factors:
            results.append(SignalResult(
                name="leakage_risk_factors",
                value=risk_factors,
                category=SignalCategory.LEAKAGE,
                description="Specific factors contributing to leakage risk",
                metadata={
                    "total_risk": risk_score,
                    "risk_level": risk_level,
                },
            ))
        
        return results


# Utility functions
def assess_leakage_risk(
    evidence: Evidence,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Assess leakage risk based on multiple indicators.
    
    Args:
        evidence: Evidence object containing model and data
        params: Optional parameters for thresholds
        
    Returns:
        Dictionary with leakage risk assessment
    """
    extractor = LeakageRiskAssessmentExtractor()
    results = extractor.extract(evidence, params)
    
    assessment = {
        "risk_score": 0.0,
        "risk_level": "minimal",
        "risk_factors": [],
    }
    
    for result in results:
        if result.name == "leakage_risk_score":
            assessment["risk_score"] = result.value
            assessment["risk_level"] = result.metadata.get("risk_level", "minimal")
        elif result.name == "leakage_risk_factors":
            assessment["risk_factors"] = result.value
            
    return assessment


def detect_leakage(
    evidence: Evidence,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Comprehensive leakage detection combining all leakage signals.
    
    Args:
        evidence: Evidence object containing model and data
        params: Optional parameters for thresholds
        
    Returns:
        Dictionary with leakage detection results
    """
    results = {}
    
    # Get individual leakage signals
    indicators_extractor = LeakageIndicatorsExtractor()
    indicator_results = indicators_extractor.extract(evidence, params)
    
    for signal in indicator_results:
        results[signal.name] = {
            "value": signal.value,
            "description": signal.description,
            "metadata": signal.metadata,
        }
    
    # Get risk assessment
    risk_assessment = assess_leakage_risk(evidence, params)
    results["risk_assessment"] = risk_assessment
    
    return results
