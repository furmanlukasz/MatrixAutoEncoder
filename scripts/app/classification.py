# MatrixAutoEncoder/scripts/classification.py

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
import matplotlib.pyplot as plt
import seaborn as sns
from joypy import joyplot

def classification(input_csv, output_dir='results/classification', selected_features=None, selected_classifiers=None):
    """
    Perform classification using RQA features.

    Args:
        input_csv (str): Path to the input CSV file with RQA features.
        output_dir (str): Directory to save classification results.
        selected_features (list): List of selected features for classification.
        selected_classifiers (list): List of selected classifiers to use.
    
    Returns:
        dict: Classification metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = load_data(input_csv)
    X, y, groups = prepare_data(df, selected_features)
    results = train_and_evaluate(X, y, groups, selected_classifiers)
    # Save results
    results_df = pd.DataFrame(results).T
    results_df.to_csv(os.path.join(output_dir, 'classification_results.csv'))
    print("\n💾 Classification results saved to 'classification_results.csv'")
    print("\n🎉 Classification complete! 🚀")
    create_ridgeline_plot(df, output_dir)
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

def prepare_data(df, selected_features=None):
    all_features = ['RR', 'DET', 'L', 'Lmax', 'ENTR', 'LAM', 'TT', 'Vmax']
    features_to_use = selected_features if selected_features else all_features
    
    X = df[features_to_use]
    y = df['Condition']
    groups = df['Subject']
    
    # Use only the data points that were not filtered out by clustering
    # mask = df['Cluster'] != -1
    # X = X[mask]
    # y = y[mask]
    # groups = groups[mask]

    # 
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

def train_and_evaluate(X, y, groups, selected_classifiers=None):
    # Initialize GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Optional: Verify that no groups overlap
    train_groups = np.unique(groups.iloc[train_idx])
    test_groups = np.unique(groups.iloc[test_idx])
    overlap = set(train_groups).intersection(set(test_groups))
    if overlap:
        print(f"❌ Overlapping subjects found in train and test sets: {overlap}")
        exit(1)
    else:
        print("✅ No overlapping subjects between training and test sets.")
    
    # Initialize classifiers
    all_classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    classifiers_to_use = {name: clf for name, clf in all_classifiers.items() if name in selected_classifiers} if selected_classifiers else all_classifiers

    results = {}
    for name, clf in classifiers_to_use.items():
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

def create_ridgeline_plot(df, output_dir):
    features = ['RR', 'DET', 'L', 'ENTR', 'LAM', 'TT', 'Vmax'] # 'Lmax',
    conditions = df['Condition'].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(conditions)))
    
    fig, axes = plt.subplots(len(features), 1, figsize=(12, 20))
    fig.suptitle("Distribution of RQA Features by Condition", fontsize=16)
    
    for i, feature in enumerate(features):
        ax = axes[i]
        for j, condition in enumerate(conditions):
            data = df[df['Condition'] == condition][feature]
            
            # Remove outliers using IQR method
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            data_filtered = data[(data >= lower_bound) & (data <= upper_bound)]
            
            sns.kdeplot(data=data_filtered, ax=ax, fill=True, color=colors[j], label=condition)
        
        ax.set_title(f"{feature} Distribution")
        ax.set_ylabel("Density")
        ax.set_xlabel("Value")
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ridgeline_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()