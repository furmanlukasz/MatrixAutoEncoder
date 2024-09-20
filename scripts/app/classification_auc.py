# MatrixAutoEncoder/scripts/classification_auc.py

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
import xgboost as xgb
import matplotlib.pyplot as plt

def classification_auc(input_csv, output_dir='results/classification_auc'):
    """
    Perform classification using XGBoost and evaluate AUC.

    Args:
        input_csv (str): Path to the input CSV file with RQA features.
        output_dir (str): Directory to save classification results and plots.
    
    Returns:
        dict: AUC results.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = load_data(input_csv)
    X, y, groups = prepare_data(df)
    results = train_and_evaluate(X, y, groups, output_dir)
    # Save results
    results_df = pd.DataFrame(results, index=['XGBoost'])
    results_df.to_csv(os.path.join(output_dir, 'xgboost_auc_results.csv'))
    print("\n💾 XGBoost AUC results saved to 'xgboost_auc_results.csv'")
    print("\n🎉 XGBoost classification and AUC calculation complete! 🚀")
    return results

def load_data(input_csv):
    df = pd.read_csv(input_csv)
    # Ensure that RQA features are in the DataFrame
    rqa_features = ['RR', 'DET', 'L', 'Lmax', 'ENTR', 'LAM', 'TT', 'Vmax']
    for feature in rqa_features:
        if feature not in df.columns:
            print(f"❌ Missing RQA feature '{feature}' in the input CSV.")
            exit(1)
    # Ensure 'Condition' and 'Subject' columns exist
    if 'Condition' not in df.columns:
        print("❌ 'Condition' column not found in the input CSV.")
        exit(1)
    if 'Subject' not in df.columns:
        print("❌ 'Subject' column not found in the input CSV.")
        exit(1)
    return df

def prepare_data(df):
    X = df[['RR', 'DET', 'L', 'Lmax', 'ENTR', 'LAM', 'TT', 'Vmax']]
    y = df['Condition']
    groups = df['Subject']
    
    # Use only the data points that were not filtered out by clustering
    # mask = df['Cluster'] != -1
    # X = X[mask]
    # y = y[mask]
    # groups = groups[mask]
    # Intead use a ForFurtherAnalysis column
    mask = df['ForFurtherAnalysis'] == True
    X = X[mask]
    y = y[mask]
    groups = groups[mask]
    
    # Encode labels if they are categorical
    if y.dtype == object:
        le = LabelEncoder()
        y = le.fit_transform(y)
    # Handle missing values if any
    X = X.fillna(X.mean())
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, groups

def train_and_evaluate(X, y, groups, output_dir):
    # Initialize GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Verify that no groups overlap
    train_groups = np.unique(groups.iloc[train_idx])
    test_groups = np.unique(groups.iloc[test_idx])
    overlap = set(train_groups).intersection(set(test_groups))
    if overlap:
        print(f"❌ Overlapping subjects found in train and test sets: {overlap}")
        exit(1)
    else:
        print("✅ No overlapping subjects between training and test sets.")
    
    # Initialize XGBoost classifier
    clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    
    print("\n🚀 Training XGBoost classifier...")
    clf.fit(X_train, y_train)
    
    # Predict probabilities
    y_pred_proba = clf.predict_proba(X_test)
    
    # Calculate AUC
    unique_classes = np.unique(y)
    if len(unique_classes) > 2:
        auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
    else:
        auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    print(f"🎯 XGBoost AUC: {auc:.4f}")
    
    # Calculate ROC curve for each class
    fpr = dict()
    tpr = dict()
    roc_auc_dict = dict()
    for i in unique_classes:
        fpr[i], tpr[i], _ = roc_curve(y_test == i, y_pred_proba[:, i])
        roc_auc_dict[i] = roc_auc_score(y_test == i, y_pred_proba[:, i])
    
    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
    for i, color in zip(unique_classes, colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {i} (AUC = {roc_auc_dict[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    print("\n💾 ROC curve saved to 'roc_curve.png'")
    
    return {'auc': auc}