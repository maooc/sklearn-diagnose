# 概率预测诊断功能迭代说明

## 概述

本次迭代增强了 sklearn-diagnose 在 scikit-learn 分类任务中的**概率预测诊断**能力，使系统能够系统性地分析概率输出质量，并将这些信息融入完整的诊断链路。

## 增强的诊断能力

### 新增故障模式

| 故障模式 | 检测条件 | 诊断意义 |
|----------|----------|----------|
| `poor_calibration` | ECE > 15% 或 Brier score > 0.20 | 预测概率与实际结果不匹配，模型过度自信或不够自信 |
| `low_confidence` | 平均置信度 < 60% 或低置信度比例 > 30% | 模型对预测结果普遍不确定 |
| `poor_class_separation` | AUC-ROC < 70% | 模型难以区分不同类别 |

### 新增信号字段

**校准质量**
- `calibration_error`: Expected Calibration Error (ECE)，衡量概率可靠性
- `brier_score`: Brier 分数，衡量概率预测质量（越低越好）

**置信度指标**
- `avg_predicted_probability`: 平均预测置信度
- `high_confidence_ratio`: 高置信度预测比例 (≥90%)
- `low_confidence_ratio`: 低置信度预测比例 (<60%)
- `avg_confidence_correct`: 正确预测的平均置信度
- `avg_confidence_incorrect`: 错误预测的平均置信度
- `confidence_gap`: 正确与错误预测的置信度差距

**类别区分能力**
- `class_separation_score`: AUC-ROC，衡量类别区分能力
- `per_class_avg_probability`: 每个类别的平均预测概率
- `per_class_probability_std`: 每个类别的概率标准差

**阈值分析（仅二分类）**
- `threshold_metrics`: 不同阈值下的 precision/recall/F1 变化
- `optimal_threshold`: 最优决策阈值（最大化 F1）
- `optimal_threshold_f1`: 最优阈值下的 F1 分数

## 覆盖的 scikit-learn 使用场景

| 场景 | 支持情况 | 说明 |
|------|----------|------|
| 普通分类器 | ✅ 完全支持 | LogisticRegression、RandomForest、GradientBoosting 等 |
| Pipeline | ✅ 完全支持 | 自动从 Pipeline 中提取概率预测 |
| 带验证集 | ✅ 完全支持 | 使用验证集计算概率信号 |
| 带交叉验证结果 | ✅ 完全支持 | CV 结果与概率信号独立，可同时使用 |
| 无概率输出能力 | ✅ 平稳退化 | LinearSVC 等无 `predict_proba` 的模型，`has_probability_predictions=False`，其他功能正常 |
| 二分类 | ✅ 完全支持 | 包含阈值分析 |
| 多分类 | ✅ 完全支持 | 使用 OvR AUC 和多类 Brier 分数 |

## 如何验证功能有效

### 运行概率诊断测试

```bash
cd /path/to/sklearn-diagnose
python -m pytest tests/unit_test_diagnose.py::TestProbabilityDiagnostics -v
```

### 测试覆盖清单

| 测试名称 | 验证内容 |
|----------|----------|
| `test_probability_signals_extracted_for_classifier` | 验证分类器概率信号正确提取 |
| `test_probability_signals_with_pipeline` | 验证 Pipeline 兼容性 |
| `test_probability_signals_with_cv_results` | 验证 CV 结果场景 |
| `test_probability_signals_graceful_degradation_without_proba` | 验证无概率模型的平稳退化 |
| `test_calibration_error_detection` | 验证校准误差计算 |
| `test_threshold_metrics_computed` | 验证阈值分析计算 |
| `test_optimal_threshold_computed` | 验证最优阈值计算 |
| `test_per_class_probability_stats` | 验证每类概率统计 |
| `test_confidence_metrics` | 验证置信度指标 |
| `test_probability_signals_in_to_dict` | 验证信号序列化 |
| `test_multiclass_probability_signals` | 验证多分类支持 |

### 验证信号存在与不存在的输出差异

**有概率输出的模型（如 LogisticRegression）**:
```python
from sklearn.linear_model import LogisticRegression
from sklearn_diagnose import diagnose

model = LogisticRegression()
model.fit(X_train, y_train)

report = diagnose(model, datasets={"train": (X_train, y_train), "val": (X_val, y_val)}, task="classification")

# 概率信号存在
assert report.signals.has_probability_predictions == True
assert report.signals.brier_score is not None
assert report.signals.calibration_error is not None
assert report.signals.class_separation_score is not None
```

**无概率输出的模型（如 LinearSVC）**:
```python
from sklearn.svm import LinearSVC
from sklearn_diagnose import diagnose

model = LinearSVC()
model.fit(X_train, y_train)

report = diagnose(model, datasets={"train": (X_train, y_train), "val": (X_val, y_val)}, task="classification")

# 概率信号不存在，但其他功能正常
assert report.signals.has_probability_predictions == False
assert report.signals.brier_score is None
assert report.signals.train_score is not None  # 其他信号正常
```

### 运行完整测试套件

```bash
python -m pytest tests/unit_test_diagnose.py -v
```

预期结果：47 passed, 3 skipped（3 个跳过的是需要真实 API key 的 LLM 集成测试）

## 文件变更清单

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `sklearn_diagnose/core/schemas.py` | 修改 | 新增 3 个 FailureMode，Signals 新增 17 个概率字段 |
| `sklearn_diagnose/core/signals.py` | 修改 | 新增概率信号提取函数 |
| `sklearn_diagnose/core/hypotheses.py` | 修改 | 新增概率相关假设检测规则 |
| `sklearn_diagnose/core/recommendations.py` | 修改 | 新增概率相关建议模板 |
| `sklearn_diagnose/llm/client.py` | 修改 | 更新 LLM 提示词 |
| `tests/conftest.py` | 修改 | MockLLMClient 支持新故障模式 |
| `tests/unit_test_diagnose.py` | 修改 | 新增 11 个概率诊断测试 |
| `README.md` | 修改 | 更新故障模式表和信号文档 |
| `CHANGELOG.md` | 修改 | 记录本次迭代变更 |

## 设计原则遵循

- ✅ **只读保证**：不重新训练模型，不修改参数，不修改数据
- ✅ **架构延续**：遵循证据收集 → 信号提取 → 诊断生成 → 总结输出的主链路
- ✅ **兼容性**：兼容 scikit-learn estimator 与 Pipeline
- ✅ **平稳退化**：无概率输出能力的模型不影响其他诊断功能
- ✅ **风格一致**：复用现有数据结构、测试风格和文档风格
