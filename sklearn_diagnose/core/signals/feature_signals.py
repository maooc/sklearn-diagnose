"""
Feature-based signal extractors.

This module contains signal extractors related to feature analysis,
including correlations, redundancy detection, and feature-target relationships.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from .base import Signal, SignalCategory, SignalResult
from .registry import register_signal

if TYPE_CHECKING:
    from ..schemas import Evidence


@register_signal("n_features")
class NFeaturesSignal(Signal):
    """Number of features in the dataset."""
    
    name = "n_features"
    category = SignalCategory.FEATURE
    description = "Number of features"
    
    def extract(self, evidence: "Evidence") -> SignalResult:
        n_features = evidence.n_features
        
        return SignalResult(
            name=self.name,
            value=n_features,
            category=self.category,
        )


@register_signal("n_samples_train")
class NSamplesTrainSignal(Signal):
    """Number of training samples."""
    
    name = "n_samples_train"
    category = SignalCategory.FEATURE
    description = "Number of training samples"
    
    def extract(self, evidence: "Evidence") -> SignalResult:
        n_samples = evidence.n_samples_train
        
        return SignalResult(
            name=self.name,
            value=n_samples,
            category=self.category,
        )


@register_signal("feature_to_sample_ratio")
class FeatureToSampleRatioSignal(Signal):
    """Ratio of features to samples (high ratio indicates potential issues)."""
    
    name = "feature_to_sample_ratio"
    category = SignalCategory.FEATURE
    description = "Ratio of features to training samples"
    
    def extract(self, evidence: "Evidence") -> SignalResult:
        if evidence.n_samples_train == 0:
            return SignalResult(
                name=self.name,
                value=None,
                category=self.category,
                metadata={"reason": "No training samples"},
            )
        
        ratio = evidence.n_features / evidence.n_samples_train
        
        return SignalResult(
            name=self.name,
            value=ratio,
            category=self.category,
            is_anomaly=ratio > 0.1,
        )


@register_signal("feature_correlations")
class FeatureCorrelationsSignal(Signal):
    """Feature correlation matrix."""
    
    name = "feature_correlations"
    category = SignalCategory.FEATURE
    description = "Feature correlation matrix"
    
    def extract(self, evidence: "Evidence") -> SignalResult:
        X = evidence.X_train
        
        if len(X.shape) != 2 or X.shape[1] < 2:
            return SignalResult(
                name=self.name,
                value=None,
                category=self.category,
                metadata={"reason": "Need at least 2 features"},
            )
        
        try:
            X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            variances = np.var(X_clean, axis=0)
            
            if np.all(variances > 1e-10):
                corr_matrix = np.corrcoef(X_clean, rowvar=False)
                
                return SignalResult(
                    name=self.name,
                    value=corr_matrix.tolist(),
                    category=self.category,
                    metadata={"n_features": X.shape[1]},
                )
            else:
                return SignalResult(
                    name=self.name,
                    value=None,
                    category=self.category,
                    metadata={"reason": "Features with zero variance detected"},
                )
        except Exception as e:
            return SignalResult(
                name=self.name,
                value=None,
                category=self.category,
                metadata={"error": str(e)},
            )


@register_signal("high_correlation_pairs")
class HighCorrelationPairsSignal(Signal):
    """Pairs of features with high correlation (potential redundancy)."""
    
    name = "high_correlation_pairs"
    category = SignalCategory.FEATURE
    description = "Feature pairs with correlation > 0.9"
    
    def extract(self, evidence: "Evidence") -> SignalResult:
        X = evidence.X_train
        
        if len(X.shape) != 2 or X.shape[1] < 2:
            return SignalResult(
                name=self.name,
                value=[],
                category=self.category,
                metadata={"reason": "Need at least 2 features"},
            )
        
        try:
            X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            variances = np.var(X_clean, axis=0)
            
            if not np.all(variances > 1e-10):
                return SignalResult(
                    name=self.name,
                    value=[],
                    category=self.category,
                    metadata={"reason": "Features with zero variance detected"},
                )
            
            corr_matrix = np.corrcoef(X_clean, rowvar=False)
            n_features = corr_matrix.shape[0]
            
            high_corr_pairs = []
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    corr = abs(corr_matrix[i, j])
                    if corr > 0.9:
                        high_corr_pairs.append({
                            "feature_i": i,
                            "feature_j": j,
                            "correlation": float(corr),
                        })
            
            high_corr_pairs.sort(key=lambda x: x["correlation"], reverse=True)
            
            return SignalResult(
                name=self.name,
                value=high_corr_pairs,
                category=self.category,
                metadata={"n_pairs": len(high_corr_pairs)},
                is_anomaly=len(high_corr_pairs) > 0,
            )
        except Exception as e:
            return SignalResult(
                name=self.name,
                value=[],
                category=self.category,
                metadata={"error": str(e)},
            )


@register_signal("feature_target_correlations")
class FeatureTargetCorrelationsSignal(Signal):
    """Correlation between each feature and the target."""
    
    name = "feature_target_correlations"
    category = SignalCategory.FEATURE
    description = "Feature-target correlations"
    
    def extract(self, evidence: "Evidence") -> SignalResult:
        X = evidence.X_train
        
        if len(X.shape) != 2:
            return SignalResult(
                name=self.name,
                value=None,
                category=self.category,
                metadata={"reason": "Invalid feature matrix shape"},
            )
        
        try:
            y = evidence.y_train.astype(float)
            correlations = []
            
            for i in range(X.shape[1]):
                try:
                    corr = np.corrcoef(X[:, i], y)[0, 1]
                    if np.isnan(corr):
                        correlations.append(0.0)
                    else:
                        correlations.append(float(corr))
                except Exception:
                    correlations.append(0.0)
            
            max_corr = max(abs(c) for c in correlations) if correlations else 0.0
            
            return SignalResult(
                name=self.name,
                value=correlations,
                category=self.category,
                metadata={
                    "n_features": len(correlations),
                    "max_abs_correlation": max_corr,
                },
            )
        except Exception as e:
            return SignalResult(
                name=self.name,
                value=None,
                category=self.category,
                metadata={"error": str(e)},
            )


@register_signal("zero_variance_features")
class ZeroVarianceFeaturesSignal(Signal):
    """Features with zero variance."""
    
    name = "zero_variance_features"
    category = SignalCategory.FEATURE
    description = "Indices of features with zero variance"
    
    def extract(self, evidence: "Evidence") -> SignalResult:
        X = evidence.X_train
        
        if len(X.shape) != 2:
            return SignalResult(
                name=self.name,
                value=[],
                category=self.category,
                metadata={"reason": "Invalid feature matrix shape"},
            )
        
        try:
            X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            variances = np.var(X_clean, axis=0)
            
            zero_var_indices = np.where(variances < 1e-10)[0].tolist()
            
            return SignalResult(
                name=self.name,
                value=zero_var_indices,
                category=self.category,
                metadata={"n_zero_variance": len(zero_var_indices)},
                is_anomaly=len(zero_var_indices) > 0,
            )
        except Exception as e:
            return SignalResult(
                name=self.name,
                value=[],
                category=self.category,
                metadata={"error": str(e)},
            )


@register_signal("feature_importance_available")
class FeatureImportanceAvailableSignal(Signal):
    """Check if feature importances are available from the model."""
    
    name = "feature_importance_available"
    category = SignalCategory.FEATURE
    description = "Whether feature importances are available"
    
    def extract(self, evidence: "Evidence") -> SignalResult:
        return SignalResult(
            name=self.name,
            value=False,
            category=self.category,
            metadata={"note": "Feature importance extraction not implemented"},
        )


FEATURE_SIGNALS = [
    NFeaturesSignal,
    NSamplesTrainSignal,
    FeatureToSampleRatioSignal,
    FeatureCorrelationsSignal,
    HighCorrelationPairsSignal,
    FeatureTargetCorrelationsSignal,
    ZeroVarianceFeaturesSignal,
    FeatureImportanceAvailableSignal,
]
