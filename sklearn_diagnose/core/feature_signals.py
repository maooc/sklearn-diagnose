"""
Feature-based signal extractors.

This module contains signal extractors that analyze feature relationships,
including feature correlations, feature importance, and feature-target
relationships.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import BaseSignalExtractor, signal_registry
from .schemas import Evidence, SignalCategory, SignalResult


@signal_registry.register_class(
    name="feature_correlations",
    category=SignalCategory.FEATURE,
    tags=["feature", "correlation", "redundancy"],
)
class FeatureCorrelationsExtractor(BaseSignalExtractor):
    """
    Extract feature correlation signals.
    
    This extractor analyzes feature-feature correlations to identify
    redundancy and multicollinearity.
    """
    
    category = SignalCategory.FEATURE
    
    def extract(
        self,
        evidence: Evidence,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[SignalResult]:
        """Extract feature correlation signals."""
        results = []
        
        X = evidence.X_train
        
        if len(X.shape) != 2 or X.shape[1] < 2:
            return results  # Need at least 2 features
            
        params = params or {}
        high_corr_threshold = params.get("high_corr_threshold", 0.9)
        
        try:
            # Handle potential NaN/inf values
            X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Only compute if we have enough variance
            variances = np.var(X_clean, axis=0)
            if not np.all(variances > 1e-10):
                return results
                
            corr_matrix = np.corrcoef(X_clean, rowvar=False)
            
            results.append(SignalResult(
                name="feature_correlations",
                value=corr_matrix,
                category=SignalCategory.FEATURE,
                description="Feature-feature correlation matrix",
                metadata={"n_features": X.shape[1]},
            ))
            
            # Find highly correlated feature pairs
            high_corr_pairs = []
            n_features = corr_matrix.shape[0]
            
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    corr = abs(corr_matrix[i, j])
                    if corr > high_corr_threshold:
                        feature_i = evidence.feature_names[i] if evidence.feature_names else i
                        feature_j = evidence.feature_names[j] if evidence.feature_names else j
                        high_corr_pairs.append((feature_i, feature_j, float(corr)))
            
            if high_corr_pairs:
                high_corr_pairs_sorted = sorted(
                    high_corr_pairs,
                    key=lambda x: x[2],
                    reverse=True
                )
                
                results.append(SignalResult(
                    name="high_correlation_pairs",
                    value=high_corr_pairs_sorted,
                    category=SignalCategory.FEATURE,
                    description=f"Feature pairs with correlation > {high_corr_threshold}",
                    metadata={
                        "n_features": X.shape[1],
                        "threshold": high_corr_threshold,
                        "n_pairs": len(high_corr_pairs_sorted),
                    },
                ))
                
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Error computing feature correlations: {e}")
            
        return results


@signal_registry.register_class(
    name="feature_target_correlations",
    category=SignalCategory.FEATURE,
    tags=["feature", "correlation", "target"],
)
class FeatureTargetCorrelationsExtractor(BaseSignalExtractor):
    """
    Extract feature-target correlation signals.
    
    This extractor analyzes correlations between individual features
    and the target variable to identify predictive power.
    """
    
    category = SignalCategory.FEATURE
    
    def extract(
        self,
        evidence: Evidence,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[SignalResult]:
        """Extract feature-target correlation signals."""
        results = []
        
        X = evidence.X_train
        
        if len(X.shape) != 2:
            return results
            
        params = params or {}
        suspicious_threshold = params.get("suspicious_threshold", 0.95)
        
        try:
            y = evidence.y_train.astype(float)
            feature_target_corr = []
            
            for i in range(X.shape[1]):
                feature = X[:, i]
                # Handle constant features
                if np.std(feature) < 1e-10:
                    feature_target_corr.append(0.0)
                    continue
                    
                corr = np.corrcoef(feature, y)[0, 1]
                if not np.isnan(corr):
                    feature_target_corr.append(corr)
                else:
                    feature_target_corr.append(0.0)
            
            # Create dictionary with feature names if available
            corr_dict = {}
            for i, corr in enumerate(feature_target_corr):
                feature_name = evidence.feature_names[i] if evidence.feature_names else str(i)
                corr_dict[feature_name] = float(corr)
            
            results.append(SignalResult(
                name="feature_target_correlations",
                value=corr_dict,
                category=SignalCategory.FEATURE,
                description="Feature-target correlation coefficients",
                metadata={"n_features": X.shape[1]},
            ))
            
            # Find features with suspiciously high correlations (potential leakage)
            suspicious = []
            for i, corr in enumerate(feature_target_corr):
                if abs(corr) > suspicious_threshold:
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
                    category=SignalCategory.FEATURE,
                    description=f"Features with suspiciously high correlation to target (> {suspicious_threshold})",
                    metadata={
                        "n_features": X.shape[1],
                        "threshold": suspicious_threshold,
                        "n_suspicious": len(suspicious_sorted),
                    },
                ))
                
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Error computing feature-target correlations: {e}")
            
        return results


@signal_registry.register_class(
    name="feature_importance",
    category=SignalCategory.FEATURE,
    tags=["feature", "importance", "selection"],
)
class FeatureImportanceExtractor(BaseSignalExtractor):
    """
    Extract feature importance signals.
    
    This extractor analyzes feature importance from tree-based models
    or computes basic importance measures like correlation to target.
    """
    
    category = SignalCategory.FEATURE
    
    def extract(
        self,
        evidence: Evidence,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[SignalResult]:
        """Extract feature importance signals."""
        results = []
        
        X = evidence.X_train
        
        if len(X.shape) != 2:
            return results
            
        # Try to get feature importance from estimator if available
        estimator = getattr(evidence, "estimator", None)
        if estimator is not None and hasattr(estimator, "feature_importances_"):
            try:
                importances = estimator.feature_importances_
                
                # Create dictionary with feature names
                importance_dict = {}
                for i, imp in enumerate(importances):
                    feature_name = evidence.feature_names[i] if evidence.feature_names else str(i)
                    importance_dict[feature_name] = float(imp)
                
                # Sort by importance
                sorted_importances = sorted(
                    importance_dict.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )
                
                results.append(SignalResult(
                    name="feature_importances",
                    value=dict(sorted_importances),
                    category=SignalCategory.FEATURE,
                    description="Feature importance scores from estimator",
                    metadata={"n_features": X.shape[1], "source": "estimator"},
                ))
                
                # Identify top features
                params = params or {}
                top_n = params.get("top_n", 5)
                top_features = sorted_importances[:top_n]
                
                results.append(SignalResult(
                    name="top_features",
                    value=top_features,
                    category=SignalCategory.FEATURE,
                    description=f"Top {top_n} most important features",
                    metadata={"n_features": X.shape[1], "top_n": top_n},
                ))
                
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Error extracting estimator feature importance: {e}")
        
        return results


# Utility functions
def find_highly_correlated_features(
    X: np.ndarray,
    threshold: float = 0.9,
    feature_names: Optional[List[str]] = None,
) -> List[Tuple[Any, Any, float]]:
    """
    Find pairs of highly correlated features.
    
    Args:
        X: Feature matrix
        threshold: Correlation threshold for flagging features
        feature_names: Optional list of feature names
        
    Returns:
        List of (feature_i, feature_j, correlation) tuples
    """
    if len(X.shape) != 2 or X.shape[1] < 2:
        return []
        
    X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    variances = np.var(X_clean, axis=0)
    
    if not np.all(variances > 1e-10):
        return []
        
    corr_matrix = np.corrcoef(X_clean, rowvar=False)
    high_corr_pairs = []
    n_features = corr_matrix.shape[0]
    
    for i in range(n_features):
        for j in range(i + 1, n_features):
            corr = abs(corr_matrix[i, j])
            if corr > threshold:
                feature_i = feature_names[i] if feature_names else i
                feature_j = feature_names[j] if feature_names else j
                high_corr_pairs.append((feature_i, feature_j, float(corr)))
    
    return sorted(high_corr_pairs, key=lambda x: x[2], reverse=True)


def find_features_by_correlation_to_target(
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.5,
    feature_names: Optional[List[str]] = None,
    sort: bool = True,
) -> List[Tuple[Any, float]]:
    """
    Find features with high correlation to the target.
    
    Args:
        X: Feature matrix
        y: Target vector
        threshold: Correlation threshold
        feature_names: Optional list of feature names
        sort: Whether to sort results by absolute correlation
        
    Returns:
        List of (feature_name, correlation) tuples above threshold
    """
    if len(X.shape) != 2:
        return []
        
    results = []
    y_float = y.astype(float)
    
    for i in range(X.shape[1]):
        feature = X[:, i]
        if np.std(feature) < 1e-10:
            continue
            
        corr = np.corrcoef(feature, y_float)[0, 1]
        if not np.isnan(corr) and abs(corr) > threshold:
            feature_name = feature_names[i] if feature_names else str(i)
            results.append((feature_name, float(corr)))
    
    if sort:
        results = sorted(results, key=lambda x: abs(x[1]), reverse=True)
    
    return results
