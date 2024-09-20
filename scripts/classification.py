# scripts/classification.py

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb

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

def train_and_evaluate(X, y, groups):
    # Initialize GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Optional: Verify that no groups overlap
    train_groups = groups.iloc[train_idx].unique()
    test_groups = groups.iloc[test_idx].unique()
    overlap = set(train_groups).intersection(set(test_groups))
    if overlap:
        print(f"❌ Overlapping subjects found in train and test sets: {overlap}")
        exit(1)
    else:
        print("✅ No overlapping subjects between training and test sets.")
    
    # Initialize classifiers
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    results = {}
    for name, clf in classifiers.items():
        print(f"\n🚀 Training {name} classifier...")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f"🎯 {name} Accuracy: {acc:.4f}")
        print(f"🏆 {name} F1 Score: {f1:.4f}")
        results[name] = {'accuracy': acc, 'f1_score': f1}
        # Detailed classification report
        print(f"📊 Classification Report for {name}:\n")
        print(classification_report(y_test, y_pred))
    return results

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    df = load_data(args.input_csv)
    X, y, groups = prepare_data(df)
    results = train_and_evaluate(X, y, groups)
    # Save results
    results_df = pd.DataFrame(results).T
    results_df.to_csv(os.path.join(args.output_dir, 'classification_results.csv'))
    print("\n💾 Classification results saved to 'classification_results.csv'")
    print("\n🎉 Classification complete! 🚀")

if __name__ == "__main__":
    main()