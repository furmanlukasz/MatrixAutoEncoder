# scripts/classification.py

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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

def train_and_evaluate(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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
    X, y = prepare_data(df)
    results = train_and_evaluate(X, y)
    # Save results
    results_df = pd.DataFrame(results).T
    results_df.to_csv(os.path.join(args.output_dir, 'classification_results.csv'))
    print("\n💾 Classification results saved to 'classification_results.csv'")
    print("\n🎉 Classification complete! 🚀")

if __name__ == "__main__":
    main()