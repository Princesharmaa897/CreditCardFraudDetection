# Credit Card Fraud Detection (E9)

A machine learning project that detects fraudulent credit card transactions using multiple classification algorithms with various class imbalance handling techniques.

## 📊 Dataset

- **Source**: Credit Card Fraud Detection dataset (`archive (2)/creditcard.csv`)
- **Size**: Large imbalanced dataset with thousands of transactions
- **Target**: Class (0: Normal transaction, 1: Fraudulent transaction)
- **Imbalance Ratio**: Fraud cases are a tiny minority (~0.17% in typical fraud datasets)
- **Features**: Time, Amount, and 28 PCA-transformed anonymized features (V1-V28)

## 🎯 Objective

Detect fraudulent credit card transactions accurately while handling severe class imbalance. Compare multiple machine learning algorithms with different resampling techniques.

## 🏗️ Project Structure

```
E9 CreditCardFraud/
├── CreditCard.py                  # Main script
├── archive (2)/
│   └── creditcard.csv             # Fraud detection dataset
├── class_distribution.png          # Generated: Class imbalance visualization
├── correlation_heatmap.png         # Generated: Feature correlations
├── eda_plots.png                   # Generated: EDA visualizations
└── README.md                       # This file
```

## 🔧 Technologies Used

- **Libraries**: scikit-learn, pandas, numpy, matplotlib, seaborn, imbalanced-learn (imblearn), XGBoost
- **Python**: 3.7+

## 📈 Models Implemented

1. **Logistic Regression** - Linear classification with max_iter=1000
2. **K-Nearest Neighbors (KNN)** - k=5 neighbors
3. **Decision Tree** - Single tree classifier
4. **Random Forest** - Ensemble with 10 estimators
5. **XGBoost** - Gradient boosting with eval_metric='logloss'

## ⚖️ Class Imbalance Handling Techniques

### 1. **Original (No Resampling)**
- Uses raw imbalanced data
- Baseline for comparison
- Shows how models perform with severe class imbalance

### 2. **Random Oversampling**
- Randomly duplicates minority class samples
- Increases training set size
- Risk of overfitting but simple approach

### 3. **Random Undersampling**
- Randomly removes majority class samples
- Reduces training set size
- Faster training but loses information

### 4. **SMOTE (Synthetic Minority Over-sampling Technique)**
- Creates synthetic samples for minority class
- Interpolates between existing minority class samples
- More sophisticated approach that reduces overfitting

## 🔄 Workflow

### 1. Data Loading & Exploration
- Load credit card fraud dataset from CSV
- Display dataset shape and basic statistics
- Print class distribution showing severe imbalance
- Display first few rows

### 2. Exploratory Data Analysis (EDA)
- **Class Distribution**: Bar chart and pie chart showing fraud vs. normal transactions
- **Correlation Heatmap**: Correlation matrix for selected features (V1-V5, Amount, Time, Class)
- **Amount Analysis**: Box plot showing transaction amounts by class
- **Time Distribution**: Histogram showing transaction time patterns by class

### 3. Data Preparation
- Separate features (X) and target (y)
- 80/20 train-test split
- Stratified split to maintain class distribution
- Random state = 42 for reproducibility

### 4. Model Training & Evaluation
For each resampling technique:
- Apply resampling to training data (if applicable)
- Train all 5 models on resampled training data
- Evaluate on original test data using:
  - **AUC (Area Under ROC Curve)**: Robust metric for imbalanced data
  - **F1-Score**: Harmonic mean of precision and recall
  - **Recall**: Critical for fraud detection (catching fraudsters)
  - **Classification Report**: Detailed per-class metrics

### 5. Model Comparison
- Pivot table showing AUC across all models and techniques
- Confusion matrices for all model-technique combinations
- Visual grid showing performance of each model under different sampling strategies

### 6. Results Analysis
- Compare which technique works best for each model
- Identify best overall model-technique combination
- Analyze trade-offs between recall and precision for fraud detection

## 📊 Expected Output

### Console Output
- Dataset shape and class distribution
- Fraud percentage in original training data
- AUC scores for original imbalanced dataset
- Resampling statistics (e.g., "Resampled fraud %: 0.5000")
- AUC scores for each model with each resampling technique
- Pivot table comparing AUC across all combinations

### Generated Visualizations
1. **class_distribution.png** - Bar chart and pie chart of class imbalance
2. **correlation_heatmap.png** - Feature correlation matrix
3. **eda_plots.png** - Box plot of amounts and histogram of time by class
4. **confusion_matrices.png** - 5x4 grid (5 models × 4 techniques) with AUC scores

## 🚀 Running the Project

```bash
# Navigate to the project directory
cd "E9 CreditCardFraud"

# Ensure virtual environment is activated
# On Windows PowerShell:
..\..\\.venv\Scripts\Activate.ps1

# Run the script
python CreditCard.py
```

## 🔑 Key Features

- **Handles Class Imbalance**: Tests 4 different resampling techniques
- **Comprehensive Evaluation**: Uses AUC, F1-score, recall, and confusion matrices
- **Multi-Model Comparison**: Evaluates 5 different algorithms
- **Visualization-Rich**: Multiple charts for understanding imbalance and performance
- **Stratified Splitting**: Maintains class balance in train/test sets
- **Advanced Models**: Includes XGBoost for powerful gradient boosting
- **Detailed Metrics**: Focuses on recall to minimize false negatives (missed frauds)

## 💡 Key Insights

### Why Class Imbalance Matters
- Fraud is rare (~0.17% of transactions)
- Models trained on imbalanced data tend to predict "normal" for everything
- Resampling techniques help balance this tendency

### Why AUC is Important
- Accuracy is misleading with imbalanced data (a model predicting all "normal" gets >99% accuracy!)
- AUC measures true positive rate vs. false positive rate, ideal for imbalanced problems
- ROC curve shows model's ability to distinguish between classes

### Why Recall Matters Most
- For fraud detection, missing a fraud (false negative) is costly
- High recall means catching most frauds, even if some false alarms occur
- F1-score balances precision and recall

## 🔍 Analysis Tips

1. **Compare AUC scores** across techniques to see which resampling works best
2. **Check recall for fraud class** (Class 1) - prioritize catching frauds
3. **Review confusion matrices** to understand false positives vs. false negatives
4. **Try different techniques**: Sometimes SMOTE outperforms oversampling
5. **Balance business needs**: More recall = catch more fraud but more false alarms

## 📋 Expected Results

- **Original dataset**: Lower recall for fraud, high AUC can be deceptive
- **Oversampling**: Usually improves fraud detection, slight overfitting
- **Undersampling**: Faster training, may lose important patterns
- **SMOTE**: Often best balanced approach, good fraud detection with reasonable precision

## ✅ Success Criteria

- Dataset loads and displays class imbalance clearly
- All EDA visualizations are generated
- All 5 models train successfully on original data
- All 4 resampling techniques apply without errors
- Models retrain on resampled data and evaluate on original test set
- AUC scores improve with resampling techniques
- Comparison tables clearly show best-performing combinations
- Confusion matrix grid effectively compares all approaches
- Recall for fraud class (Class 1) is properly optimized
