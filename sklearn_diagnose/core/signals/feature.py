"""
Feature-based signal extractors.

Signals related to feature correlations, redundancies,
and feature-target relationships.
"""

from typing import List, Optional, Tuple

import numpy as np

from ._base import SignalExtractor, SignalResult, SignalCategory
from ..schemas import Evidence


class FeatureSignalExtractor(SignalExtractor):
    """
    Extractor for feature-level signals.
    
    Analyzes feature correlations, redundancies, and
    feature-target relationships.
    """
    
    def __init__(self):
        super().__init__("feature", category=SignalCategory.FEATURE)
    
    def get_required_evidence(self) -> List[str]:
        return ["X_train", "y_train"]
    
    def extract(self, evidence: Evidence) -> List[SignalResult]:
        """Extract feature-based signals."""
        results = []
        
        X = evidence.X_train
        
        if len(X.shape) != 2:
            return results
        
        n_samples, n_features = X.shape
        
        if n_features < 2:
            return results
        
        # Basic feature statistics
        results.append(SignalResult(
            name="n_features",
            value=n_features,
            category=SignalCategory.FEATURE,
            description="Number of features",
            metadata={"n_samples": n_samples}
        ))
        
        # Feature-to-sample ratio
        ratio = n_features / n_samples if n_samples > 0 else 0
        results.append(SignalResult(
            name="feature_to_sample_ratio",
            value=ratio,
            category=SignalCategory.FEATURE,
            description="Ratio of features to samples",
            metadata={
                "n_features": n_features,
                "n_samples": n_samples,
                "high_dimensional": ratio > 0.1
            }
        ))
        
        # Feature correlations
        results.extend(self._extract_correlation_signals(evidence))
        
        # Feature-target correlations
        results.extend(self._extract_feature_target_signals(evidence))
        
        return results
    
    def _extract_correlation_signals(
        self, 
        evidence: Evidence
    ) -> List[SignalResult]:
        """Extract feature correlation signals."""
        results = []
        X = evidence.X_train
        
        try:
            # Clean data
            X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Check variance
            variances = np.var(X_clean, axis=0)
            constant_features = np.sum(variances < 1e-10)
            
            results.append(SignalResult(
                name="constant_features",
                value=int(constant_features),
                category=SignalCategory.FEATURE,
                description="Number of features with near-zero variance",
                metadata={
                    "should_remove": constant_features > 0,
                    "indices": [i for i, v in enumerate(variances) if v < 1e-10]
                }
            ))
            
            # Compute correlation matrix
            if np.all(variances > 1e-10):
                corr_matrix = np.corrcoef(X_clean, rowvar=False)
                
                results.append(SignalResult(
                    name="feature_correlation_matrix",
                    value=corr_matrix.tolist(),
                    category=SignalCategory.FEATURE,
                    description="Pairwise correlation matrix of features",
                    metadata={"shape": corr_matrix.shape}
                ))
                
                # Find highly correlated pairs
                high_corr_pairs = self._find_high_correlations(
                    corr_matrix, threshold=0.9
                )
                
                results.append(SignalResult(
                    name="high_correlation_pairs",
                    value=high_corr_pairs,
                    category=SignalCategory.FEATURE,
                    description="Feature pairs with correlation > 0.9",
                    metadata={
                        "n_pairs": len(high_corr_pairs),
                        "threshold": 0.9,
                        "has_redundancy": len(high_corr_pairs) > 0
                    }
                ))
                
                # Correlation statistics
                abs_corr = np.abs(corr_matrix)
                # Get upper triangle (excluding diagonal)
                upper_tri = abs_corr[np.triu_indices_from(abs_corr, k=1)]
                
                results.append(SignalResult(
                    name="mean_feature_correlation",
                    value=float(np.mean(upper_tri)),
                    category=SignalCategory.FEATURE,
                    description="Mean absolute correlation between features",
                    metadata={
                        "max": float(np.max(upper_tri)),
                        "std": float(np.std(upper_tri))
                    }
                ))
        
        except Exception:
            pass
        
        return results
    
    def _extract_feature_target_signals(
        self, 
        evidence: Evidence
    ) -> List[SignalResult]:
        """Extract feature-target correlation signals."""
        results = []
        X = evidence.X_train
        
        try:
            y = evidence.y_train.astype(float)
            feature_target_corr = []
            
            for i in range(X.shape[1]):
                try:
                    corr = np.corrcoef(X[:, i], y)[0, 1]
                    if not np.isnan(corr):
                        feature_target_corr.append(float(corr))
                    else:
                        feature_target_corr.append(0.0)
                except Exception:
                    feature_target_corr.append(0.0)
            
            feature_target_corr = np.array(feature_target_corr)
            
            results.append(SignalResult(
                name="feature_target_correlations",
                value=feature_target_corr.tolist(),
                category=SignalCategory.FEATURE,
                description="Correlation of each feature with target",
                metadata={
                    "max_correlation": float(np.max(np.abs(feature_target_corr))),
                    "mean_correlation": float(np.mean(np.abs(feature_target_corr)))
                }
            ))
            
            # Find highly correlated features
            high_corr_features = [
                {"feature_index": i, "correlation": float(corr)}
                for i, corr in enumerate(feature_target_corr)
                if abs(corr) > 0.95
            ]
            
            if high_corr_features:
                results.append(SignalResult(
                    name="suspicious_feature_target_corr",
                    value=high_corr_features,
                    category=SignalCategory.FEATURE,
                    description="Features with suspiciously high target correlation",
                    metadata={
                        "n_features": len(high_corr_features),
                        "threshold": 0.95,
                        "may_indicate_leakage": len(high_corr_features) > 0
                    }
                ))
            
            # Feature importance ranking
            ranked_indices = np.argsort(np.abs(feature_target_corr))[::-1]
            top_features = [
                {"rank": rank + 1, "feature_index": int(idx), 
                 "correlation": float(feature_target_corr[idx])}
                for rank, idx in enumerate(ranked_indices[:10])  # Top 10
            ]
            
            results.append(SignalResult(
                name="top_features_by_correlation",
                value=top_features,
                category=SignalCategory.FEATURE,
                description="Top features ranked by absolute correlation with target",
                metadata={"n_ranked": len(top_features)}
            ))
        
        except Exception:
            pass
        
        return results
    
    def _find_high_correlations(
        self,
        corr_matrix: np.ndarray,
        threshold: float = 0.9
    ) -> List[tuple]:
        """Find feature pairs with correlation above threshold.

        Returns list of tuples (feature_i, feature_j, correlation)
        for backward compatibility with existing code.
        """
        pairs = []
        n_features = corr_matrix.shape[0]

        for i in range(n_features):
            for j in range(i + 1, n_features):
                corr = abs(corr_matrix[i, j])
                if corr > threshold:
                    # Return tuple format for backward compatibility
                    pairs.append((i, j, float(corr)))

        # Sort by correlation (descending)
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs


class FeatureRedundancyAnalyzer(SignalExtractor):
    """
    Specialized extractor for feature redundancy analysis.
    
    Identifies redundant features that could be removed
    without losing information.
    """
    
    def __init__(self):
        super().__init__("feature_redundancy", category=SignalCategory.FEATURE)
    
    def get_required_evidence(self) -> List[str]:
        return ["X_train"]
    
    def extract(self, evidence: Evidence) -> SignalResult:
        """Extract feature redundancy analysis."""
        X = evidence.X_train
        
        if len(X.shape) != 2 or X.shape[1] < 2:
            return SignalResult.failure(
                name="feature_redundancy",
                error="Need at least 2 features",
                category=SignalCategory.FEATURE,
                description="Feature redundancy analysis"
            )
        
        try:
            X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            corr_matrix = np.corrcoef(X_clean, rowvar=False)
            
            # Find clusters of highly correlated features
            high_threshold = 0.95
            moderate_threshold = 0.9
            
            high_pairs = []
            moderate_pairs = []
            
            n_features = corr_matrix.shape[0]
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    corr = abs(corr_matrix[i, j])
                    if corr > high_threshold:
                        high_pairs.append((i, j, float(corr)))
                    elif corr > moderate_threshold:
                        moderate_pairs.append((i, j, float(corr)))
            
            # Identify redundant features (those in multiple high correlations)
            redundant_candidates = set()
            for i, j, _ in high_pairs:
                redundant_candidates.add(i)
                redundant_candidates.add(j)
            
            return SignalResult(
                name="feature_redundancy_analysis",
                value={
                    "high_correlation_pairs": [
                        {"f1": i, "f2": j, "corr": c} for i, j, c in high_pairs
                    ],
                    "moderate_correlation_pairs": [
                        {"f1": i, "f2": j, "corr": c} for i, j, c in moderate_pairs
                    ],
                    "redundant_feature_candidates": sorted(redundant_candidates),
                },
                category=SignalCategory.FEATURE,
                description="Analysis of feature redundancy",
                metadata={
                    "n_high_corr": len(high_pairs),
                    "n_moderate_corr": len(moderate_pairs),
                    "n_redundant_candidates": len(redundant_candidates),
                    "recommendation": self._recommend_action(
                        len(high_pairs), len(redundant_candidates)
                    )
                }
            )
        
        except Exception as e:
            return SignalResult.failure(
                name="feature_redundancy",
                error=str(e),
                category=SignalCategory.FEATURE,
                description="Feature redundancy analysis"
            )
    
    def _recommend_action(self, n_high: int, n_redundant: int) -> str:
        """Recommend action based on redundancy metrics."""
        if n_high > 10 or n_redundant > 5:
            return "consider_feature_selection_or_pca"
        elif n_high > 0:
            return "review_highly_correlated_pairs"
        else:
            return "no_action_needed"
