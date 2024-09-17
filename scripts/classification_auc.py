# scripts/classification.py

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Classification using RQA features")
    parser.add_argument('--input_csv', type=str, required=True,
                        help='Path to the input CSV file with RQA features')
    parser.add_argument('--output_dir', type=str, default='results/classification',
                        help='Directory to save classification results')
    return parser.parse_args()

def load_data(input_csv):
    df = pd.read_csv(input_csv)
    # Ensure that RQA features are in the DataFrame
    rqa_features = ['RR', 'DET', 'L', 'Lmax', 'ENTR', 'LAM', 'TT', 'Vmax']
    for feature in rqa_features:
        if feature not in df.columns:
            print(f"❌ Missing RQA feature '{feature}' in the input CSV.")
            exit(1)
    return df

def prepare_data(df):
    # Assuming 'label' column contains the target variable
    if 'Condition' not in df.columns:
        print("❌ 'Condition' column not found in the input CSV.")
        exit(1)
    X = df[['RR', 'DET', 'L', 'Lmax', 'ENTR', 'LAM', 'TT', 'Vmax']]
    y = df['Condition']
    # Encode labels if they are categorical
    if y.dtype == object:
        le = LabelEncoder()
        y = le.fit_transform(y)
    # Handle missing values if any
    X = X.fillna(X.mean())
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def train_and_evaluate(X, y, output_dir):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize XGBoost classifier
    clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    
    print("\n🚀 Training XGBoost classifier...")
    clf.fit(X_train, y_train)
    
    # Predict probabilities
    y_pred_proba = clf.predict_proba(X_test)
    
    # Calculate AUC
    auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
    print(f"🎯 XGBoost AUC: {auc:.4f}")
    
    # Calculate ROC curve for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(np.unique(y))):
        fpr[i], tpr[i], _ = roc_curve(y_test == i, y_pred_proba[:, i])
        roc_auc[i] = roc_auc_score(y_test == i, y_pred_proba[:, i])
    
    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green']
    for i, color in zip(range(len(np.unique(y))), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {i} (AUC = {roc_auc[i]:.2f})')
    
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

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    df = load_data(args.input_csv)
    X, y = prepare_data(df)
    results = train_and_evaluate(X, y, args.output_dir)
    # Save results
    results_df = pd.DataFrame(results, index=['XGBoost'])
    results_df.to_csv(os.path.join(args.output_dir, 'xgboost_auc_results.csv'))
    print("\n💾 XGBoost AUC results saved to 'xgboost_auc_results.csv'")
    print("\n🎉 XGBoost classification and AUC calculation complete! 🚀")

if __name__ == "__main__":
    main()