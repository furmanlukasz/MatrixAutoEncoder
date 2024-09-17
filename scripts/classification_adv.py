# scripts/classification_adv.py

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV
import xgboost as xgb

def parse_args():
    parser = argparse.ArgumentParser(description="Advanced Classification using RQA features with Group K-Fold and RFE")
    parser.add_argument('--input_csv', type=str, required=True,
                        help='Path to the input CSV file with RQA features')
    parser.add_argument('--output_dir', type=str, default='results/classification_adv',
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
    # Ensure 'Condition' and 'subject_id' columns are present
    if 'Condition' not in df.columns:
        print("❌ 'Condition' column not found in the input CSV.")
        exit(1)
    if 'subject' not in df.columns:
        print("❌ 'subject' column not found in the input CSV.")
        exit(1)
    return df

def prepare_data(df):
    X = df[['RR', 'DET', 'L', 'Lmax', 'ENTR', 'LAM', 'TT', 'Vmax']]
    y = df['Condition']
    groups = df['subject']
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
    # Initialize classifiers
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    results = {}
    group_kfold = GroupKFold(n_splits=5)
    fold = 1
    for train_index, test_index in group_kfold.split(X, y, groups):
        print(f"\n📂 Fold {fold}:")

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        groups_train, groups_test = groups.iloc[train_index], groups.iloc[test_index]

        # Check for data leakage
        overlap = set(groups_train) & set(groups_test)
        if overlap:
            print(f"❌ Data leakage detected in fold {fold}. Overlapping subjects: {overlap}")
            exit(1)
        else:
            print(f"✅ No data leakage detected in fold {fold}.")

        for name, clf in classifiers.items():
            print(f"\n🚀 Training {name} classifier...")
            if name == 'SVM':
                # For SVM, we'll use all features without RFE
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
            else:
                # Recursive Feature Elimination with Cross-Validation
                rfecv = RFECV(estimator=clf, step=1, cv=GroupKFold(n_splits=5), scoring='accuracy', n_jobs=-1)
                rfecv.fit(X_train, y_train, groups=groups_train)
                print(f"🔍 Optimal number of features for {name}: {rfecv.n_features_}")
                print(f"📋 Selected features for {name}: {list(np.array(['RR', 'DET', 'L', 'Lmax', 'ENTR', 'LAM', 'TT', 'Vmax'])[rfecv.support_])}")
                y_pred = rfecv.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            print(f"🎯 {name} Accuracy: {acc:.4f}")
            print(f"🏆 {name} F1 Score: {f1:.4f}")
            # Detailed classification report
            print(f"📊 Classification Report for {name} (Fold {fold}):\n")
            print(classification_report(y_test, y_pred))

            # Store results
            if name not in results:
                results[name] = {'accuracy': [], 'f1_score': []}
            results[name]['accuracy'].append(acc)
            results[name]['f1_score'].append(f1)

        fold += 1

    # Aggregate results
    for name in classifiers.keys():
        avg_acc = np.mean(results[name]['accuracy'])
        avg_f1 = np.mean(results[name]['f1_score'])
        print(f"\n📈 Average Results for {name}:")
        print(f"🎯 Average Accuracy: {avg_acc:.4f}")
        print(f"🏆 Average F1 Score: {avg_f1:.4f}")
        results[name]['avg_accuracy'] = avg_acc
        results[name]['avg_f1_score'] = avg_f1

    return results

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    df = load_data(args.input_csv)
    X, y, groups = prepare_data(df)
    results = train_and_evaluate(X, y, groups, args.output_dir)
    # Save results
    results_df = pd.DataFrame({
        name: {
            'Average Accuracy': metrics['avg_accuracy'],
            'Average F1 Score': metrics['avg_f1_score']
        } for name, metrics in results.items()
    }).T
    results_df.to_csv(os.path.join(args.output_dir, 'classification_results.csv'))
    print("\n💾 Classification results saved to 'classification_results.csv'")
    print("\n🎉 Classification complete! 🚀")

if __name__ == "__main__":
    main()