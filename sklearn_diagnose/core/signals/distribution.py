"""
Distribution-based signal extractors.

Signals related to class distribution, label distribution, and
data distribution characteristics.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from ._base import SignalExtractor, SignalResult, SignalCategory
from ..schemas import Evidence, TaskType


class DistributionSignalExtractor(SignalExtractor):
    """
    Extractor for distribution-based signals.
    
    Analyzes class distributions, label distributions, and
    distributional properties of the data.
    """
    
    def __init__(self):
        super().__init__("distribution", category=SignalCategory.DISTRIBUTION)
    
    def get_required_evidence(self) -> List[str]:
        return ["y_train", "task"]
    
    def extract(self, evidence: Evidence) -> List[SignalResult]:
        """Extract distribution-based signals."""
        results = []

        # Basic dataset statistics
        results.append(SignalResult(
            name="n_samples_train",
            value=len(evidence.y_train),
            category=SignalCategory.DISTRIBUTION,
            description="Number of training samples",
        ))

        if evidence.y_val is not None:
            results.append(SignalResult(
                name="n_samples_val",
                value=len(evidence.y_val),
                category=SignalCategory.DISTRIBUTION,
                description="Number of validation samples",
            ))

        if evidence.task == TaskType.CLASSIFICATION:
            results.extend(self._extract_classification_distribution(evidence))
        else:
            results.extend(self._extract_regression_distribution(evidence))

        return results
    
    def _extract_classification_distribution(
        self, 
        evidence: Evidence
    ) -> List[SignalResult]:
        """Extract classification-specific distribution signals."""
        results = []
        
        y = evidence.y_train
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        n_classes = len(unique)
        
        # Class distribution
        class_dist = {
            str(cls): float(count / total)
            for cls, count in zip(unique, counts)
        }
        results.append(SignalResult(
            name="class_distribution",
            value=class_dist,
            category=SignalCategory.DISTRIBUTION,
            description="Proportion of samples in each class",
            metadata={"n_classes": n_classes, "total_samples": total}
        ))
        
        # Minority class ratio
        minority_ratio = float(np.min(counts) / total)
        results.append(SignalResult(
            name="minority_class_ratio",
            value=minority_ratio,
            category=SignalCategory.DISTRIBUTION,
            description="Ratio of smallest class to total samples",
            metadata={
                "n_classes": n_classes,
                "imbalance_ratio": float(np.max(counts) / np.min(counts)) if n_classes > 1 else 1.0,
                "is_imbalanced": minority_ratio < 0.2
            }
        ))
        
        # Entropy of class distribution
        probs = counts / total
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        max_entropy = np.log2(n_classes) if n_classes > 1 else 0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        results.append(SignalResult(
            name="class_entropy",
            value=normalized_entropy,
            category=SignalCategory.DISTRIBUTION,
            description="Normalized entropy of class distribution (0=imbalanced, 1=uniform)",
            metadata={
                "raw_entropy": float(entropy),
                "max_entropy": float(max_entropy),
                "n_classes": n_classes
            }
        ))
        
        # Per-class metrics (if predictions available)
        if evidence.y_pred_val is not None and evidence.y_val is not None:
            results.extend(self._extract_per_class_metrics(evidence, unique))
        
        return results
    
    def _extract_per_class_metrics(
        self, 
        evidence: Evidence,
        classes: np.ndarray
    ) -> List[SignalResult]:
        """Extract per-class performance metrics."""
        results = []
        
        try:
            # Confusion matrix
            cm = confusion_matrix(evidence.y_val, evidence.y_pred_val)
            results.append(SignalResult(
                name="confusion_matrix",
                value=cm.tolist(),
                category=SignalCategory.DISTRIBUTION,
                description="Confusion matrix on validation set",
                metadata={"shape": cm.shape}
            ))
            
            # Per-class recall
            recalls = recall_score(
                evidence.y_val, 
                evidence.y_pred_val,
                average=None, 
                zero_division=0
            )
            per_class_recall = {
                str(cls): float(rec) 
                for cls, rec in zip(classes, recalls)
            }
            results.append(SignalResult(
                name="per_class_recall",
                value=per_class_recall,
                category=SignalCategory.DISTRIBUTION,
                description="Recall for each class",
                metadata={"mean_recall": float(np.mean(recalls))}
            ))
            
            # Per-class precision
            precisions = precision_score(
                evidence.y_val,
                evidence.y_pred_val,
                average=None,
                zero_division=0
            )
            per_class_precision = {
                str(cls): float(prec)
                for cls, prec in zip(classes, precisions)
            }
            results.append(SignalResult(
                name="per_class_precision",
                value=per_class_precision,
                category=SignalCategory.DISTRIBUTION,
                description="Precision for each class",
                metadata={"mean_precision": float(np.mean(precisions))}
            ))
            
            # Class-wise performance disparity
            recall_std = float(np.std(recalls))
            results.append(SignalResult(
                name="class_performance_disparity",
                value=recall_std,
                category=SignalCategory.DISTRIBUTION,
                description="Standard deviation of per-class recall (higher = more disparity)",
                metadata={
                    "recall_range": float(np.max(recalls) - np.min(recalls)),
                    "interpretation": "high_disparity" if recall_std > 0.2 else "moderate" if recall_std > 0.1 else "balanced"
                }
            ))
            
        except Exception:
            pass  # Handle edge cases gracefully
        
        return results
    
    def _extract_regression_distribution(
        self, 
        evidence: Evidence
    ) -> List[SignalResult]:
        """Extract regression-specific distribution signals."""
        results = []
        
        y = evidence.y_train.astype(float)
        
        # Target distribution statistics
        results.append(SignalResult(
            name="target_mean",
            value=float(np.mean(y)),
            category=SignalCategory.DISTRIBUTION,
            description="Mean of target variable",
            metadata={}
        ))
        
        results.append(SignalResult(
            name="target_std",
            value=float(np.std(y)),
            category=SignalCategory.DISTRIBUTION,
            description="Standard deviation of target variable",
            metadata={}
        ))
        
        # Target distribution shape
        if len(y) > 3:
            try:
                skewness = float(stats.skew(y))
                kurt = float(stats.kurtosis(y))
                results.append(SignalResult(
                    name="target_skewness",
                    value=skewness,
                    category=SignalCategory.DISTRIBUTION,
                    description="Skewness of target distribution",
                    metadata={"is_symmetric": abs(skewness) < 0.5}
                ))
                results.append(SignalResult(
                    name="target_kurtosis",
                    value=kurt,
                    category=SignalCategory.DISTRIBUTION,
                    description="Excess kurtosis of target distribution",
                    metadata={"is_normal": abs(kurt) < 3.0}
                ))
            except Exception:
                pass
        
        # Residual analysis (if predictions available)
        if evidence.y_pred_train is not None:
            results.extend(self._extract_residual_signals(evidence))
        
        return results
    
    def _extract_residual_signals(self, evidence: Evidence) -> List[SignalResult]:
        """Extract residual-based signals for regression."""
        results = []
        
        residuals = evidence.y_train - evidence.y_pred_train
        
        results.append(SignalResult(
            name="residual_mean",
            value=float(np.mean(residuals)),
            category=SignalCategory.DISTRIBUTION,
            description="Mean of residuals (should be near 0)",
            metadata={"is_biased": abs(np.mean(residuals)) > 0.01}
        ))
        
        results.append(SignalResult(
            name="residual_std",
            value=float(np.std(residuals)),
            category=SignalCategory.DISTRIBUTION,
            description="Standard deviation of residuals",
            metadata={}
        ))
        
        # Residual distribution shape
        if len(residuals) > 3:
            try:
                skewness = float(stats.skew(residuals))
                kurt = float(stats.kurtosis(residuals))
                
                results.append(SignalResult(
                    name="residual_skewness",
                    value=skewness,
                    category=SignalCategory.DISTRIBUTION,
                    description="Skewness of residual distribution",
                    metadata={
                        "is_symmetric": abs(skewness) < 0.5,
                        "interpretation": self._interpret_skewness(skewness)
                    }
                ))
                
                results.append(SignalResult(
                    name="residual_kurtosis",
                    value=kurt,
                    category=SignalCategory.DISTRIBUTION,
                    description="Excess kurtosis of residual distribution",
                    metadata={
                        "is_normal": abs(kurt) < 3.0,
                        "has_heavy_tails": kurt > 3.0
                    }
                ))
                
                # Normality test (simplified)
                # Shapiro-Wilk for small samples, D'Agostino for larger
                if len(residuals) < 50:
                    try:
                        stat, pvalue = stats.shapiro(residuals)
                        results.append(SignalResult(
                            name="residual_normality_pvalue",
                            value=float(pvalue),
                            category=SignalCategory.DISTRIBUTION,
                            description="Shapiro-Wilk test p-value for normality",
                            metadata={
                                "is_normal": pvalue > 0.05,
                                "test": "shapiro"
                            }
                        ))
                    except Exception:
                        pass
                
            except Exception:
                pass
        
        return results
    
    def _interpret_skewness(self, skewness: float) -> str:
        """Interpret residual skewness."""
        if abs(skewness) < 0.5:
            return "approximately_symmetric"
        elif skewness > 0.5:
            return "right_skewed"
        else:
            return "left_skewed"


class ClassBalanceAnalyzer(SignalExtractor):
    """
    Specialized extractor for class imbalance analysis.
    
    Provides detailed metrics for understanding class imbalance
    and its potential impact on model performance.
    """
    
    def __init__(self):
        super().__init__("class_balance", category=SignalCategory.DISTRIBUTION)
    
    def get_required_evidence(self) -> List[str]:
        return ["y_train", "task"]
    
    def extract(self, evidence: Evidence) -> SignalResult:
        """Extract class balance analysis."""
        if evidence.task != TaskType.CLASSIFICATION:
            return SignalResult.failure(
                name="class_balance",
                error="Only applicable to classification tasks",
                category=SignalCategory.DISTRIBUTION,
                description="Class balance analysis"
            )
        
        y = evidence.y_train
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        
        # Sort by count (descending)
        sorted_indices = np.argsort(counts)[::-1]
        sorted_classes = unique[sorted_indices]
        sorted_counts = counts[sorted_indices]
        
        class_info = []
        for i, (cls, count) in enumerate(zip(sorted_classes, sorted_counts)):
            class_info.append({
                "class": str(cls),
                "count": int(count),
                "proportion": float(count / total),
                "rank": i + 1
            })
        
        # Imbalance metrics
        imbalance_ratio = (
            float(sorted_counts[0] / sorted_counts[-1]) 
            if len(sorted_counts) > 1 else 1.0
        )
        
        gini = self._compute_gini(counts)
        
        return SignalResult(
            name="class_balance_analysis",
            value={
                "classes": class_info,
                "imbalance_ratio": imbalance_ratio,
                "gini_coefficient": float(gini),
            },
            category=SignalCategory.DISTRIBUTION,
            description="Detailed class balance analysis",
            metadata={
                "n_classes": len(unique),
                "is_balanced": imbalance_ratio < 2.0,
                "is_moderately_imbalanced": 2.0 <= imbalance_ratio < 10.0,
                "is_highly_imbalanced": imbalance_ratio >= 10.0,
                "recommendation": self._recommend_strategy(imbalance_ratio, gini)
            }
        )
    
    def _compute_gini(self, counts: np.ndarray) -> float:
        """Compute Gini coefficient for class distribution."""
        n = len(counts)
        if n <= 1:
            return 0.0
        
        proportions = counts / np.sum(counts)
        return 1 - np.sum(proportions ** 2)
    
    def _recommend_strategy(self, imbalance_ratio: float, gini: float) -> str:
        """Recommend handling strategy based on imbalance metrics."""
        if imbalance_ratio < 2.0:
            return "no_action_needed"
        elif imbalance_ratio < 5.0:
            return "consider_class_weights"
        elif imbalance_ratio < 10.0:
            return "use_stratified_sampling_and_class_weights"
        else:
            return "consider_resampling_techniques"
