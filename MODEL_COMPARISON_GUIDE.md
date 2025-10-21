# Model Variant Comparison Guide

This guide explains how to train and compare different model configurations with various preprocessing options (SMOTE and Chi-Squared feature selection).

## Overview

The project now supports training models with different preprocessing configurations:
- **SMOTE**: Synthetic Minority Over-sampling Technique for handling class imbalance
- **Chi-Squared Feature Selection**: Dimensionality reduction using Chi-Squared statistical test

Each model variant saves its preprocessing configuration along with the model artifacts, ensuring that evaluation uses the exact same preprocessing pipeline.

## Quick Start

### 1. Train All Variants

To train all four combinations of SMOTE and Chi2:

```bash
uv run python scripts/train_all_variants.py
```

This will train models in the following directories:
- `models/smote_chi2` - With SMOTE and Chi2 feature selection
- `models/smote_no_chi2` - With SMOTE, without Chi2
- `models/no_smote_chi2` - Without SMOTE, with Chi2
- `models/no_smote_no_chi2` - Without SMOTE or Chi2 (baseline)

### 2. Compare All Variants

After training, compare performance:

```bash
uv run python scripts/compare_variants.py
```

This will:
- Evaluate each model on the test set
- Display a comparison table
- Identify the best-performing model for each metric
- Save detailed results to `models/comparison_results.json`

## Training Individual Models

You can also train individual model variants using the CLI:

```bash
# With SMOTE and Chi2 (20 features)
uv run ids-train --output-dir models/smote_chi2

# With SMOTE, without Chi2
uv run ids-train --no-chi2 --output-dir models/smote_no_chi2

# Without SMOTE, with Chi2
uv run ids-train --no-smote --output-dir models/no_smote_chi2

# Without SMOTE or Chi2 (baseline)
uv run ids-train --no-smote --no-chi2 --output-dir models/no_smote_no_chi2

# Customize feature count for Chi2
uv run ids-train --chi2-features 30 --output-dir models/custom
```

## Evaluating Individual Models

To evaluate a specific model:

```bash
uv run ids-evaluate --model-dir models/no_smote_chi2
```

The evaluation will automatically:
1. Load the preprocessing configuration
2. Apply the same preprocessing transformations used during training
3. Display the configuration being used
4. Report performance metrics

## Model Artifacts

Each trained model directory contains:

```
models/variant_name/
├── model.json                      # XGBoost model
├── label_mapping.json              # Class label encoding
├── best_hyperparameters.json       # Model hyperparameters
├── preprocessing_config.json       # Preprocessing settings (NEW!)
├── scaler.pkl                      # MinMaxScaler (if Chi2 was used)
├── selector.pkl                    # SelectKBest selector (if Chi2 was used)
└── model_metrics.json              # Evaluation metrics
```

### Preprocessing Configuration Format

The `preprocessing_config.json` file stores the exact preprocessing settings:

```json
{
  "use_smote": true,
  "smote_k_neighbors": 5,
  "use_chi2": true,
  "chi2_k_features": 20
}
```

## Comparison Output

The comparison script produces:

### Console Output
```
================================================================================
COMPARISON SUMMARY
================================================================================

Configuration                  Accuracy     Weighted F1  Macro F1
--------------------------------------------------------------------------------
No SMOTE + No Chi2             0.9988       0.9987       0.8402
No SMOTE + Chi2                0.9844       0.9840       0.7383

================================================================================
BEST PERFORMING MODELS
================================================================================
Best Accuracy:    No SMOTE + No Chi2 (0.9988)
Best Weighted F1: No SMOTE + No Chi2 (0.9987)
Best Macro F1:    No SMOTE + No Chi2 (0.8402)
```

### JSON Output (`models/comparison_results.json`)

Contains detailed metrics for all evaluated models, sorted by accuracy.

## Best Practices

1. **Consistency**: Always use the saved configuration when evaluating models. The evaluation script automatically handles this.

2. **Naming Convention**: Use descriptive directory names that indicate the configuration:
   - `models/smote_chi2` - Clear what preprocessing was used
   - Avoid generic names like `models/test` or `models/model1`

3. **Documentation**: The preprocessing configuration is automatically saved, so you don't need to manually track which settings were used for each model.

4. **Comparison**: Run the comparison script after training multiple variants to identify the best configuration for your use case.

## Understanding the Metrics

- **Accuracy**: Overall correctness across all classes
- **Weighted F1**: F1-score weighted by class support (good for imbalanced datasets)
- **Macro F1**: Unweighted average F1-score across all classes (treats all classes equally)

For intrusion detection with class imbalance:
- Use **Weighted F1** if you want to prioritize overall detection rate
- Use **Macro F1** if you want to ensure all attack types are detected equally well

## Example Workflow

```bash
# 1. Train all variants
uv run python scripts/train_all_variants.py

# 2. Compare performance
uv run python scripts/compare_variants.py

# 3. Evaluate specific model in detail
uv run ids-evaluate --model-dir models/best_model

# 4. Use best model for predictions (future feature)
```

## Notes

- Training with SMOTE significantly increases training time due to synthetic sample generation
- Chi-Squared feature selection reduces features from 78 to 20 by default
- GPU acceleration is used by default when available (use `--no-gpu` to disable)
- Models trained with different configurations cannot be directly compared unless evaluated with the same preprocessing pipeline
