"""
Distribution-based signal extractors.

This module contains signal extractors that analyze data distributions,
including class distribution for classification and residual analysis
for regression.
"""

from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from .base import BaseSignalExtractor, signal_registry
from .schemas import Evidence, SignalCategory, SignalResult, TaskType


@signal_registry.register_class(
    name="class_distribution",
    category=SignalCategory.DISTRIBUTION,
    tags=["classification", "distribution", "imbalance"],
)
class ClassDistributionExtractor(BaseSignalExtractor):
    """
    Extract class distribution from training data.
    
    This extractor analyzes the class distribution in classification tasks
    and identifies potential class imbalance issues.
    """
    
    category = SignalCategory.DISTRIBUTION
    supported_tasks = {TaskType.CLASSIFICATION}
    
    def extract(
        self,
        evidence: Evidence,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[SignalResult]:
        """Extract class distribution information."""
        results = []
        
        if evidence.task != TaskType.CLASSIFICATION:
            return results
            
        unique, counts = np.unique(evidence.y_train, return_counts=True)
        total = len(evidence.y_train)
        
        # Class distribution
        class_dist = {str(cls): count / total for cls, count in zip(unique, counts)}
        
        results.append(SignalResult(
            name="class_distribution",
            value=class_dist,
            category=SignalCategory.DISTRIBUTION,
            description="Class distribution in training data",
            metadata={"n_classes": len(unique), "n_samples": total},
        ))
        
        # Minority class ratio (imbalance indicator)
        if len(counts) > 1:
            minority_ratio = float(np.min(counts) / total)
            
            results.append(SignalResult(
                name="minority_class_ratio",
                value=minority_ratio,
                category=SignalCategory.DISTRIBUTION,
                description="Ratio of minority class (imbalance indicator)",
                metadata={
                    "n_classes": len(unique),
                    "minority_class": str(unique[np.argmin(counts)]),
                    "imbalance_threshold": 0.1,  # Typical threshold for concern
                },
            ))
        
        return results


@signal_registry.register_class(
    name="classification_metrics",
    category=SignalCategory.DISTRIBUTION,
    tags=["classification", "metrics", "per-class"],
)
class ClassificationMetricsExtractor(BaseSignalExtractor):
    """
    Extract per-class classification metrics.
    
    This extractor computes per-class precision, recall, and confusion matrix
    for classification tasks.
    """
    
    category = SignalCategory.DISTRIBUTION
    supported_tasks = {TaskType.CLASSIFICATION}
    
    def extract(
        self,
        evidence: Evidence,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[SignalResult]:
        """Extract per-class classification metrics."""
        results = []
        
        if evidence.task != TaskType.CLASSIFICATION:
            return results
            
        if not evidence.has_validation_set or evidence.y_pred_val is None:
            return results
            
        try:
            # Get unique classes
            unique, _ = np.unique(evidence.y_train, return_counts=True)
            
            # Confusion matrix
            cm = confusion_matrix(evidence.y_val, evidence.y_pred_val)
            
            results.append(SignalResult(
                name="confusion_matrix",
                value=cm,
                category=SignalCategory.DISTRIBUTION,
                description="Confusion matrix on validation set",
                metadata={"n_classes": len(unique)},
            ))
            
            # Per-class recall
            recalls = recall_score(
                evidence.y_val, evidence.y_pred_val,
                average=None, zero_division=0
            )
            
            per_class_recall = {
                str(cls): float(rec)
                for cls, rec in zip(unique, recalls)
            }
            
            results.append(SignalResult(
                name="per_class_recall",
                value=per_class_recall,
                category=SignalCategory.DISTRIBUTION,
                description="Per-class recall scores on validation set",
                metadata={"n_classes": len(unique)},
            ))
            
            # Per-class precision
            precisions = precision_score(
                evidence.y_val, evidence.y_pred_val,
                average=None, zero_division=0
            )
            
            per_class_precision = {
                str(cls): float(prec)
                for cls, prec in zip(unique, precisions)
            }
            
            results.append(SignalResult(
                name="per_class_precision",
                value=per_class_precision,
                category=SignalCategory.DISTRIBUTION,
                description="Per-class precision scores on validation set",
                metadata={"n_classes": len(unique)},
            ))
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Error computing classification metrics: {e}")
        
        return results


@signal_registry.register_class(
    name="residual_analysis",
    category=SignalCategory.DISTRIBUTION,
    tags=["regression", "residual", "distribution"],
)
class ResidualAnalysisExtractor(BaseSignalExtractor):
    """
    Extract residual analysis signals for regression tasks.
    
    This extractor analyzes residuals from regression predictions
    including mean, std, skewness, and kurtosis.
    """
    
    category = SignalCategory.DISTRIBUTION
    supported_tasks = {TaskType.REGRESSION}
    
    def extract(
        self,
        evidence: Evidence,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[SignalResult]:
        """Extract residual analysis signals."""
        results = []
        
        if evidence.task != TaskType.REGRESSION:
            return results
            
        if evidence.y_pred_train is None:
            return results
            
        # Training residuals
        residuals = evidence.y_train - evidence.y_pred_train
        
        # Residual mean
        residual_mean = float(np.mean(residuals))
        results.append(SignalResult(
            name="residual_mean",
            value=residual_mean,
            category=SignalCategory.DISTRIBUTION,
            description="Mean of training residuals (bias indicator)",
            metadata={"n_samples": len(residuals)},
        ))
        
        # Residual standard deviation
        residual_std = float(np.std(residuals))
        results.append(SignalResult(
            name="residual_std",
            value=residual_std,
            category=SignalCategory.DISTRIBUTION,
            description="Standard deviation of training residuals",
            metadata={"n_samples": len(residuals)},
        ))
        
        # Skewness and kurtosis (need at least 4 samples)
        if len(residuals) > 3:
            try:
                residual_skew = float(stats.skew(residuals))
                results.append(SignalResult(
                    name="residual_skew",
                    value=residual_skew,
                    category=SignalCategory.DISTRIBUTION,
                    description="Skewness of training residuals (asymmetry indicator)",
                    metadata={"n_samples": len(residuals)},
                ))
                
                residual_kurtosis = float(stats.kurtosis(residuals))
                results.append(SignalResult(
                    name="residual_kurtosis",
                    value=residual_kurtosis,
                    category=SignalCategory.DISTRIBUTION,
                    description="Kurtosis of training residuals (tail weight indicator)",
                    metadata={"n_samples": len(residuals)},
                ))
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Error computing residual skewness/kurtosis: {e}")
        
        return results


@signal_registry.register_class(
    name="data_quality",
    category=SignalCategory.DATA_QUALITY,
    tags=["data", "quality", "basic"],
)
class DataQualityExtractor(BaseSignalExtractor):
    """
    Extract basic data quality signals.
    
    This extractor computes basic data characteristics like
    sample counts, feature counts, and feature-to-sample ratios.
    """
    
    category = SignalCategory.DATA_QUALITY
    
    def extract(
        self,
        evidence: Evidence,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[SignalResult]:
        """Extract basic data quality signals."""
        results = []
        
        # Sample counts
        results.append(SignalResult(
            name="n_samples_train",
            value=evidence.n_samples_train,
            category=SignalCategory.DATA_QUALITY,
            description="Number of training samples",
        ))
        
        if evidence.n_samples_val is not None:
            results.append(SignalResult(
                name="n_samples_val",
                value=evidence.n_samples_val,
                category=SignalCategory.DATA_QUALITY,
                description="Number of validation samples",
            ))
        
        # Feature counts
        results.append(SignalResult(
            name="n_features",
            value=evidence.n_features,
            category=SignalCategory.DATA_QUALITY,
            description="Number of features",
        ))
        
        # Feature-to-sample ratio
        if evidence.n_samples_train > 0:
            feature_to_sample_ratio = evidence.n_features / evidence.n_samples_train
            
            results.append(SignalResult(
                name="feature_to_sample_ratio",
                value=feature_to_sample_ratio,
                category=SignalCategory.DATA_QUALITY,
                description="Feature-to-sample ratio (curse of dimensionality indicator)",
                metadata={"threshold": 0.1, "high_dimensional": feature_to_sample_ratio > 0.1},
            ))
        
        return results
