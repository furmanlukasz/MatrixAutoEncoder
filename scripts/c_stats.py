# scripts/classification.py

import os
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Generate statistics and plots from RQA features")
    parser.add_argument('--input_csv', type=str, required=True,
                        help='Path to the input CSV file with RQA features')
    parser.add_argument('--output_dir', type=str, default='results/stats',
                        help='Directory to save statistical results and plots')
    return parser.parse_args()

def load_data(input_csv):
    df = pd.read_csv(input_csv)
    # Verify necessary columns
    required_columns = ['Condition', 'RR', 'DET', 'L', 'Lmax', 'ENTR', 'LAM', 'TT', 'Vmax']
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        print(f"❌ Missing columns in the input CSV: {missing}")
        exit(1)
    return df

def remove_outliers_quantile(df):
    q_low = 0.05
    q_high = 0.95
    df = df[df['RR'] > df['RR'].quantile(q_low)]
    df = df[df['RR'] < df['RR'].quantile(q_high)]
    df = df[df['DET'] > df['DET'].quantile(q_low)]
    df = df[df['DET'] < df['DET'].quantile(q_high)]
    df = df[df['L'] > df['L'].quantile(q_low)]
    df = df[df['L'] < df['L'].quantile(q_high)]
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

def generate_correlation_heatmap(df, output_dir):
    charts_dir = os.path.join(output_dir, 'charts_stats')
    os.makedirs(charts_dir, exist_ok=True)
    
    rqa_features = ['RR', 'DET', 'L', 'Lmax', 'ENTR', 'LAM', 'TT', 'Vmax']
    correlation = df[rqa_features].corr()
    
    plt.figure(figsize=(10,8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap of RQA Features')
    plt.tight_layout()
    plot_path = os.path.join(charts_dir, 'correlation_heatmap.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"💾 Correlation heatmap saved to '{plot_path}'")

def generate_bar_plots(df, output_dir, mode='average'):
    charts_dir = os.path.join(output_dir, 'charts_stats')
    os.makedirs(charts_dir, exist_ok=True)
    
    rqa_features = ['RR', 'DET', 'L', 'Lmax', 'ENTR', 'LAM', 'TT', 'Vmax']
    if mode == 'average':
        condition_group = df.groupby('Condition')[rqa_features].mean().reset_index()
    elif mode == 'median':
        condition_group = df.groupby('Condition')[rqa_features].median().reset_index()
    else:
        print("❌ Invalid mode. Please specify 'average' or 'median'.")
        exit(1)
    
    for feature in rqa_features:
        plt.figure(figsize=(8,6))
        plt.bar(condition_group['Condition'], condition_group[feature], color='skyblue')
        plt.xlabel('Condition')
        plt.ylabel(f'Average {feature}')
        plt.title(f'Average {feature} by Condition')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plot_path = os.path.join(charts_dir, f'average_{feature}_by_condition.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"💾 Plot saved to '{plot_path}'")

def generate_box_plots(df, output_dir):
    charts_dir = os.path.join(output_dir, 'charts_stats')
    os.makedirs(charts_dir, exist_ok=True)
    
    rqa_features = ['RR', 'DET', 'L', 'Lmax', 'ENTR', 'LAM', 'TT', 'Vmax']
    for feature in rqa_features:
        plt.figure(figsize=(10,6))
        sns.boxplot(x='Condition', y=feature, data=df, hue='Condition')
        plt.xlabel('Condition')
        plt.ylabel(feature)
        plt.title(f'Box Plot of {feature} by Condition')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plot_path = os.path.join(charts_dir, f'boxplot_{feature}_by_condition.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"💾 Box plot saved to '{plot_path}'")

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    df = load_data(args.input_csv)
    df = remove_outliers_quantile(df)
    # Generate statistical summaries (optional)
    stats_summary = df.describe()
    stats_summary.to_csv(os.path.join(args.output_dir, 'stats_summary.csv'))
    print("💾 Statistical summary saved to 'stats_summary.csv'")
    
    # Generate bar plots
    generate_bar_plots(df, args.output_dir)
    generate_box_plots(df, args.output_dir)
    generate_correlation_heatmap(df, args.output_dir)
    print("🎉 Statistics and plots generation complete! 🚀")

if __name__ == "__main__":
    main()