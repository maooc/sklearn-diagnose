# 概率预测诊断功能迭代说明

## 概述

本次迭代增强了 sklearn-diagnose 在分类任务中的**概率预测诊断**能力。系统现在能够系统性地分析模型的概率输出质量，提取确定性信号，并生成相应的诊断结论、修复建议和可读性总结。

## 增强的诊断能力

### 1. 概率分布分析
- **proba_mean**: 最大预测概率的均值（反映整体置信度水平）
- **proba_std**: 预测概率的标准差（反映置信度分布的离散程度）
- **proba_entropy**: 预测熵（反映预测的不确定性）
- **proba_calibration_error**: 期望校准误差（反映概率校准质量）

### 2. 置信度分析
- **high_confidence_ratio**: 高置信度预测比例（概率 > 0.9）
- **low_confidence_ratio**: 低置信度预测比例（概率 < 0.6）
- **confidence_accuracy_correlation**: 置信度与正确性的相关性

### 3. 类别区分度分析
- **proba_margin_mean**: Top-2 类别概率的均值差距
- **proba_margin_std**: 概率差距的标准差
- **ambiguous_predictions_ratio**: 模糊预测比例（差距 < 0.2）

### 4. 阈值变化分析（二分类）
- **optimal_threshold**: 最大化 F1 分数的最优阈值
- **threshold_sensitivity**: 阈值变化对性能的影响程度
- **auc_roc**: ROC 曲线下面积
- **auc_pr**: PR 曲线下面积

### 5. 每类别概率分析
- **per_class_proba_mean**: 每个真实类别的平均预测概率
- **per_class_proba_std**: 每个真实类别的预测概率标准差

## 新增故障模式

本次迭代新增了 5 种与概率预测相关的故障模式：

| 故障模式 | 描述 | 检测条件 |
|---------|------|---------|
| **POOR_CALIBRATION** | 概率校准不良 | 校准误差 > 10% |
| **LOW_CONFIDENCE_PREDICTIONS** | 系统性低置信度 | 平均概率 < 65% 或低置信度比例 > 25% |
| **AMBIGUOUS_CLASS_BOUNDARIES** | 类别边界模糊 | 平均差距 < 30% 或模糊预测比例 > 20% |
| **SUBOPTIMAL_THRESHOLD** | 次优决策阈值 | 最优阈值与 0.5 差距 > 15% |
| **CONFIDENCE_ACCURACY_MISMATCH** | 置信度-准确性不匹配 | 置信度与正确性相关性 < 0.2 |

## 覆盖的 scikit-learn 使用场景

### 1. 普通分类器
- ✅ LogisticRegression
- ✅ RandomForestClassifier
- ✅ GradientBoostingClassifier
- ✅ DecisionTreeClassifier

### 2. Pipeline
- ✅ 支持包含预处理的 Pipeline
- ✅ 自动通过 Pipeline 获取概率预测

### 3. 验证集场景
- ✅ 有验证集时提取概率信号
- ✅ 无验证集时优雅降级（has_probability_outputs = False）

### 4. 交叉验证场景
- ✅ 与 CV 结果共存
- ✅ 概率信号与 CV 信号互补

### 5. 无概率输出能力的模型
- ✅ SVC（无 probability=True）
- ✅ SGDClassifier（hinge loss）
- ✅ 其他无 predict_proba 的分类器

### 6. 多分类场景
- ✅ 自动检测类别数
- ✅ 二分类专用信号（AUC、最优阈值）仅在二分类时计算

## 完整诊断链路集成

概率预测诊断已完全集成到诊断链路中：

### 1. 信号提取 → LLM 输入
- 所有概率信号通过 `_build_hypothesis_prompt()` 函数格式化为 LLM 提示词
- 概率信号以结构化格式呈现，包含关键指标和警告信息
- 故障模式列表扩展至 12 个，包含 5 个概率相关模式

### 2. LLM 诊断生成
- 系统提示词指导 LLM 关注概率相关故障模式
- LLM 根据概率信号生成诊断结论（Hypotheses）
- 置信度、严重程度和证据均基于概率信号计算

### 3. 建议生成
- 建议生成器优先处理概率相关故障模式
- 针对概率问题提供专门的修复建议（校准、阈值调整等）
- 建议按影响程度排序

### 4. 总结输出
- 总结生成时包含概率分析结果
- 摘要中体现概率诊断发现的问题
- 概率指标（校准误差、AUC-ROC、最优阈值等）出现在最终报告中

### 5. 平稳退化
- 无 `predict_proba` 的模型：`has_probability_outputs = False`，概率信号为 None
- 无验证集时：不提取概率信号，诊断基于传统指标
- 多分类场景：二分类专用信号自动设为 None

## 如何验证功能有效

### 1. 运行测试套件
```bash
# 运行概率信号提取测试
python -m pytest tests/test_probability_signals.py -v

# 运行完整链路集成测试
python -m pytest tests/test_probability_integration.py -v

# 运行所有测试
python -m pytest tests/ -v
```

### 2. 运行示例脚本
```bash
python examples/probability_diagnosis_example.py
```

### 3. 检查完整链路输出
```python
from sklearn_diagnose import diagnose

report = diagnose(estimator=model, datasets=datasets, task="classification")

# 检查概率信号
print(f"Has probability outputs: {report.signals.has_probability_outputs}")
print(f"Mean probability: {report.signals.proba_mean}")
print(f"Calibration error: {report.signals.proba_calibration_error}")

# 检查概率相关故障模式
for h in report.hypotheses:
    if "calibration" in h.name.value or "threshold" in h.name.value:
        print(f"Detected: {h.name.value} ({h.confidence:.1%})")

# 检查概率相关建议
for rec in report.recommendations:
    if rec.related_hypothesis and "calibration" in rec.related_hypothesis.value:
        print(f"Recommendation: {rec.action}")

# 查看包含概率分析的总结
print(report.summary(use_llm=True))
print(f"AUC-ROC: {report.signals.auc_roc}")

# 检查故障模式检测
for h in report.hypotheses:
    if "calibration" in h.name.value or "threshold" in h.name.value:
        print(f"Detected: {h.name.value} ({h.confidence:.1%})")
```

## 架构兼容性

- ✅ **只读保证**: 不重新训练模型，不修改参数，不修改输入数据
- ✅ **诊断架构兼容**: 完全融入证据收集 → 信号提取 → 诊断生成 → 总结输出链路
- ✅ **scikit-learn 兼容**: 支持所有标准 estimator 和 Pipeline 使用方式
- ✅ **数据结构复用**: 复用现有的 Signals、Hypothesis、Recommendation 数据结构
- ✅ **测试风格一致**: 遵循现有的测试模式和风格

## 文件变更清单

### 核心代码
- `sklearn_diagnose/core/schemas.py`: 添加概率信号字段和故障模式
- `sklearn_diagnose/core/signals.py`: 实现概率信号提取函数
- `sklearn_diagnose/core/hypotheses.py`: 添加概率故障模式检测规则
- `sklearn_diagnose/core/recommendations.py`: 添加概率故障模式建议模板

### 测试
- `tests/test_probability_signals.py`: 新增 16 个测试用例
- `tests/conftest.py`: 更新 MockLLMClient 支持概率故障模式

### 文档和示例
- `README.md`: 更新故障模式表格和信号说明
- `CHANGELOG.md`: 记录迭代内容
- `examples/probability_diagnosis_example.py`: 功能演示示例
- `PROBABILITY_DIAGNOSIS_SUMMARY.md`: 本说明文档

## 统计

- **新增信号**: 17 个概率相关信号字段
- **新增故障模式**: 5 个
- **新增建议模板**: 5 组（每组 3 个建议）
- **新增测试**: 16 个测试用例
- **代码行数**: 约 +600 行（核心功能 + 测试）
