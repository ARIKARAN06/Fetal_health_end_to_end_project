# Fetal Health Classification Using Machine Learning

A machine learning project that analyzes fetal cardiotocography-related measurements and builds multiclass classification models to predict fetal health categories.

This project demonstrates a complete data science workflow including data cleaning, exploratory data analysis, preprocessing, model development, evaluation, cross-validation, hyperparameter tuning, and feature importance analysis.

## Project Overview

The objective of this project is to classify fetal health status using 21 predictor variables available in the dataset.

**Domain:** Healthcare  
**Problem Type:** Multiclass Classification  
**Target Variable:** `fetal_health`  
**Cleaned Observations:** 2,113  
**Predictor Variables:** 21

The project is intended as a machine learning study and should not be considered a replacement for professional medical assessment.

## Dataset

The original dataset contained:

- 2,126 observations
- 22 columns
- 21 predictor variables
- 1 target variable

During data-quality analysis:

- No missing values were found.
- No infinite values were found.
- 13 duplicate rows were identified and removed.
- The final cleaned dataset contained 2,113 observations.

### Target Distribution

| Class | Count | Percentage |
|---|---:|---:|
| 1.0 | 1,646 | 77.90% |
| 2.0 | 292 | 13.82% |
| 3.0 | 175 | 8.28% |

The target variable is imbalanced, so model evaluation was not based on accuracy alone.

## Project Workflow

The project follows an end-to-end machine learning workflow:

1. Problem definition
2. Dataset understanding
3. Data-quality assessment
4. Duplicate removal
5. Exploratory data analysis
6. Feature and target separation
7. Stratified train-test splitting
8. Feature standardization for scale-sensitive models
9. Baseline model development
10. Multiple-model comparison
11. Cross-validation
12. Hyperparameter optimization
13. Final model evaluation
14. Feature importance analysis
15. Findings and conclusions

## Machine Learning Models

The following classification algorithms were evaluated:

- Baseline Classifier
- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest

### Initial Model Performance

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| Random Forest | 0.9527 | 0.9522 | 0.8785 | 0.9105 |
| Decision Tree | 0.9338 | 0.8982 | 0.8713 | 0.8843 |
| Logistic Regression | 0.8889 | 0.8214 | 0.7613 | 0.7885 |
| K-Nearest Neighbors | 0.8889 | 0.8447 | 0.7225 | 0.7720 |
| Baseline Classifier | 0.7801 | 0.2600 | 0.3333 | 0.2922 |

Random Forest produced the strongest initial performance.

Macro F1-score was treated as an important evaluation metric because it gives equal importance to each target class despite the class imbalance.

## Hyperparameter Optimization

Random Forest was further optimized using `RandomizedSearchCV` with stratified cross-validation.

Best parameters:

```text
n_estimators: 500
max_depth: 20
min_samples_split: 2
min_samples_leaf: 2
max_features: sqrt
class_weight: balanced
```

**Best Cross-Validation Macro F1:** `0.8881`

## Final Model Performance

The optimized Random Forest achieved:

- **Accuracy:** 94.80%
- **Macro Precision:** 0.9271
- **Macro Recall:** 0.8944
- **Macro F1-score:** 0.9092

### Class-Level Performance

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| 1.0 | 0.96 | 0.98 | 0.97 |
| 2.0 | 0.88 | 0.76 | 0.81 |
| 3.0 | 0.94 | 0.94 | 0.94 |

Class 2.0 was the most challenging category for the optimized model, while Classes 1.0 and 3.0 achieved strong classification performance.

## Important Features

The ten most influential predictors identified by the optimized Random Forest were:

1. `abnormal_short_term_variability`
2. `percentage_of_time_with_abnormal_long_term_variability`
3. `histogram_mean`
4. `histogram_median`
5. `accelerations`
6. `mean_value_of_short_term_variability`
7. `mean_value_of_long_term_variability`
8. `prolongued_decelerations`
9. `histogram_mode`
10. `baseline value`

Feature importance represents predictive contribution within the fitted model and should not be interpreted as evidence of medical causation.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook

## Key Findings

The project showed that tree-based models performed particularly well on this dataset. Random Forest achieved the strongest overall results, substantially outperforming the majority-class baseline when evaluated using Macro F1-score.

The baseline achieved approximately 78% accuracy but only 0.2922 Macro F1, demonstrating why accuracy alone can be misleading when working with imbalanced classification data.

The optimized Random Forest maintained strong balanced performance, although Class 2.0 remained more difficult to identify than Classes 1.0 and 3.0.

## Limitations

- The target classes are imbalanced.
- Minority classes contain relatively few observations.
- Results are based on a limited dataset.
- Feature importance does not establish causal relationships.
- Independent external validation would be required to assess generalization to other datasets or populations.
- This project is intended for data science and machine learning study and is not a clinically validated diagnostic system.

## Conclusion

This project demonstrates a complete machine learning workflow for multiclass fetal health classification.

Random Forest achieved the strongest initial performance with **95.27% accuracy and 0.9105 Macro F1**. After systematic cross-validation and hyperparameter optimization, the optimized Random Forest achieved **94.80% accuracy and 0.9092 Macro F1** on the held-out test set.

The project highlights the importance of appropriate evaluation metrics, stratified sampling, leakage prevention, cross-validation, model comparison, and responsible interpretation when working with imbalanced healthcare datasets.