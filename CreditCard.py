import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids
from imblearn.under_sampling import TomekLinks
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

plt.ioff()  # Turn off interactive mode to avoid blocking on show()

# Step 1: Load dataset
dataset_path = r'd:/SEM 4/AML LB/Experiment 9/archive (2)/creditcard.csv'
df = pd.read_csv(dataset_path)
print("Dataset shape:", df.shape)
print(df.head())
print(df['Class'].value_counts())

# Step 2: Visualize structure and imbalance
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
df['Class'].value_counts().plot(kind='bar')
plt.title('Class Distribution (Imbalanced)')
plt.xlabel('Class (0: Normal, 1: Fraud)')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
plt.pie(df['Class'].value_counts(), labels=['Normal', 'Fraud'], autopct='%1.1f%%')
plt.title('Class Proportion')
plt.savefig('class_distribution.png')
# plt.show()

# Dataset info
print(df.info())
print(df.describe())

# Step 3: EDA
# Correlation heatmap for selected features (V1-V5, Amount, Class)
plt.figure(figsize=(10, 8))
corr_cols = ['V1', 'V2', 'V3', 'V4', 'Time', 'Amount', 'Class']
sns.heatmap(df[corr_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.show()

# Amount distribution by class
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.boxplot(x='Class', y='Amount', data=df)
plt.title('Amount by Class')

plt.subplot(1, 2, 2)
sns.histplot(data=df, x='Time', hue='Class', stat='density', common_norm=False)
plt.title('Time distribution by Class')
plt.savefig('eda_plots.png')
# plt.show()

# Step 4: Prepare data
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Original train fraud %:", y_train.mean())

# Define models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=10),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
}

# Function to train and evaluate
def evaluate_model(model, X_tr, y_tr, X_te, y_te, name):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_pred_proba = model.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, y_pred_proba)
    report = classification_report(y_te, y_pred, output_dict=True)
    return {
        'AUC': auc,
        'F1 Fraud': report['1']['f1-score'],
        'Recall Fraud': report['1']['recall'],
        'y_pred': y_pred
    }

# Results storage
results = pd.DataFrame()

# Original imbalanced
print("\n--- Original Imbalanced Dataset ---")
orig_results = []
all_predictions_orig = {}
for name, model in models.items():
    scores = evaluate_model(model, X_train, y_train, X_test, y_test, name)
    scores['Technique'] = 'Original'
    orig_results.append({**scores, 'Model': name})
    print(f"{name}: AUC = {scores['AUC']:.4f}")
    all_predictions_orig[name] = scores['y_pred']

results = pd.concat([results, pd.DataFrame(orig_results)], ignore_index=True)

all_predictions = {'Original': all_predictions_orig}

# Sampling techniques
samplers = {
    'Random Oversampling': RandomOverSampler(random_state=42),
    'Random Undersampling': RandomUnderSampler(random_state=42),
    # 'Tomek Links': TomekLinks(),  # Slow on large data
    # 'Cluster Centroids': ClusterCentroids(random_state=42),  # Slow on large data
    'SMOTE': SMOTE(random_state=42)
}

for tech, sampler in samplers.items():
    print(f"\n--- {tech} ---")
    X_res, y_res = sampler.fit_resample(X_train, y_train)
    print(f"Resampled fraud %: {y_res.mean():.4f}")
    
    tech_results = []
    all_predictions_tech = {}
    for name, model in models.items():
        scores = evaluate_model(model, X_res, y_res, X_test, y_test, name)
        scores['Technique'] = tech
        tech_results.append({**scores, 'Model': name})
        print(f"{name}: AUC = {scores['AUC']:.4f}")
        all_predictions_tech[name] = scores['y_pred']
    
    results = pd.concat([results, pd.DataFrame(tech_results)], ignore_index=True)
    all_predictions[tech] = all_predictions_tech

# Step 6: Compare performances
print("\n--- Full Results ---")
print(results.pivot_table(index='Model', columns='Technique', values='AUC').round(4))

# Confusion Matrices for All Models per Technique
techniques = ['Original'] + list(samplers.keys())
model_names = list(models.keys())
fig, axes = plt.subplots(len(techniques), len(model_names), figsize=(25, 20))

for i, tech in enumerate(techniques):
    for j, model_name in enumerate(model_names):
        y_pred = all_predictions[tech][model_name]
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i, j],
                    xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
        auc_val = results[(results['Model'] == model_name) & (results['Technique'] == tech)]['AUC'].iloc[0]
        axes[i, j].set_title(f"{model_name} - {tech}\nAUC: {auc_val:.4f}")

plt.suptitle('Confusion Matrices - All Models per Technique', fontsize=22)
plt.tight_layout()
plt.savefig('all_confusion_matrices.png')
# plt.show()

print("\n--- Confusion Matrix Summaries (All Models) ---")
for tech in techniques:
    print(f"\n{tech}:")
    for model_name in model_names:
        cm = confusion_matrix(y_test, all_predictions[tech][model_name])
        print(f"  {model_name}:\n{cm}")

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.barplot(data=results, x='Model', y='AUC', hue='Technique', ax=axes[0])
axes[0].set_title('AUC-ROC by Model and Technique')
axes[0].tick_params(axis='x', rotation=45)

sns.barplot(data=results, x='Model', y='Recall Fraud', hue='Technique', ax=axes[1])
axes[1].set_title('Fraud Recall by Model and Technique')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('model_comparison.png')
# plt.show()

# Best model
best_row = results.loc[results['AUC'].idxmax()]
print(f"\nBest Model: {best_row['Model']} with {best_row['Technique']} - AUC: {best_row['AUC']:.4f}")

print("\nExperiment completed successfully!")

