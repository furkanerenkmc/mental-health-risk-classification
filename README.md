# Mental Health Risk Classification: Overfitting Analysis & Ensemble Benchmarking

This repository contains an end-to-end Machine Learning project designed to classify mental health risk levels (`Low`, `Medium`, `High`) using lifestyle, clinical, and demographic attributes. 

A primary focus of this project is to implement state-of-the-art **Ensemble Learning algorithms** and strictly diagnose **Overfitting Behaviors** by analyzing the performance gap between the Training and Test partitions.

---

## 🚀 Project Workflow & Engineering Steps

### 1. Exploratory Data Analysis (EDA) & Visualization
- **Risk Factor Identification:** Analyzed correlations and statistical properties across features. Discovered that indicators like `depression_score`, `anxiety_score`, and `social_support_score` carry the highest predictive power for mental health risk stratification.
- **Visual Insights:** Plotted data distributions and feature behaviors using custom Matplotlib and Seaborn visualization pipelines.

### 2. Preprocessing & Data Engineering Pipeline
- **Categorical Mapping:** Safely converted binary attributes (e.g., `gender`, `ever_married`) into clear numeric flags.
- **One-Hot Encoding:** Applied multi-class categorical encoding via Scikit-Learn's preprocessing tools to correctly transform non-numeric variables like `work_type`, `Residence_type`, and `smoking_status`.
- **Data Scaling:** Implemented `StandardScaler` to ensure normalized numerical feature scales, preventing distance distortions and strictly isolating transformations to avoid data leakage.

### 3. Model Architecture & Ensemble Methods
To identify the most robust classifier and evaluate variance, 5 distinct tree-based ensemble models were trained and tested:
1. **XGBoost Classifier** (Gradient Boosting with regularization)
2. **LightGBM Classifier** (Histogram-based optimization)
3. **Random Forest Classifier** (Bagging method)
4. **Gradient Boosting Classifier** (Sequential stage-wise additive modeling)
5. **AdaBoost Classifier** (Adaptive boosting with decision stumps)

---

## 📊 Performance Comparison & Overfitting Diagnosis

To detect whether models memorized the patterns (overfitting) or generalized successfully, we cross-examined the **Train Accuracy vs. Test Accuracy** and mapped out the exact **Overfit Gap**:

| Machine Learning Model | Train Accuracy | Test Accuracy | Overfit Gap (Train - Test) |
| :--- | :---: | :---: | :---: |
| **XGBoost** | 1.0000 (100%) | **1.0000 (100%)** | 0.0000 (Perfect Fit) |
| **LightGBM** | 1.0000 (100%) | **1.0000 (100%)** | 0.0000 (Perfect Fit) |
| **Gradient Boosting** | 0.9997 | **0.9968** | 0.0029 (Very Low Variance) |
| **Random Forest** | 1.0000 (100%) | **0.9784** | 0.0216 (Slight Overfitting) |
| **AdaBoost** | 0.8036 | **0.8004** | 0.0032 (Underfitted but Consistent) |

### 🔍 Key Takeaways from Overfitting Analysis:
- **XGBoost & LightGBM** achieved flawless generalization, recording perfect **100% Accuracy** on both train and test partitions without showing any variance or signs of overfitting.
- **Random Forest** showed a small overfit gap of **0.0216**, meaning it slightly memorized the training split but still managed a strong performance on unseen data.
- **AdaBoost** proved to be immune to overfitting (Gap: 0.0032), but underperformed in raw accuracy (~80%) compared to the more advanced gradient boosters.

---

## 🛠️ Tech Stack & Dependencies
- **Language:** Python 3.13
- **Libraries:** Pandas, NumPy, Scikit-Learn, XGBoost, LightGBM, Matplotlib, Seaborn
- **Environment:** Jupyter Notebook / Anaconda
