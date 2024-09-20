import os
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def grid_search_heatmap(input_csv, output_dir='results/grid_search', eps_range=(0.1, 1.0, 0.05), min_samples_range=(2, 20)):
    """
    Perform grid search for DBSCAN parameters and create a heatmap of results.

    Args:
        input_csv (str): Path to the input CSV file with latent space data.
        output_dir (str): Directory to save results and plots.
        eps_range (tuple): Range for eps parameter (start, stop, step).
        min_samples_range (tuple): Range for min_samples parameter (start, stop).
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv)

    eps_values = np.arange(*eps_range)
    min_samples_values = range(*min_samples_range)

    results = []

    total_iterations = len(eps_values) * len(min_samples_values)
    progress_bar = tqdm(total=total_iterations, desc="Grid Search Progress")

    for eps in eps_values:
        for min_samples in min_samples_values:
            auc, acc = evaluate_clustering(df, eps, min_samples)
            results.append({'eps': eps, 'min_samples': min_samples, 'auc': auc, 'accuracy': acc})
            progress_bar.update(1)

    progress_bar.close()

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'grid_search_results.csv'), index=False)

    create_heatmap(results_df, 'auc', 'AUC', output_dir)
    create_heatmap(results_df, 'accuracy', 'Accuracy', output_dir)

def evaluate_clustering(df, eps, min_samples):
    umap_embeddings = df[['UMAP1', 'UMAP2', 'UMAP3']].values
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df['Cluster'] = dbscan.fit_predict(umap_embeddings)

    cluster_condition_counts = df.groupby('Cluster')['Condition'].nunique()
    mixed_clusters = cluster_condition_counts[cluster_condition_counts > 1].index.tolist()
    df['ForFurtherAnalysis'] = ~df['Cluster'].isin(mixed_clusters)

    filtered_df = df[df['ForFurtherAnalysis']]

    X = filtered_df[['RR', 'DET', 'L', 'Lmax', 'ENTR', 'LAM', 'TT', 'Vmax']]
    y = filtered_df['Condition']
    groups = filtered_df['Subject']

    # Encode string labels to numerical values
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Ensure all classes are present in both train and test sets
    unique_classes = np.unique(y_encoded)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y_encoded, groups))

    # Check if all classes are present in both sets
    while len(np.unique(y_encoded[train_idx])) != len(unique_classes) or len(np.unique(y_encoded[test_idx])) != len(unique_classes):
        train_idx, test_idx = next(gss.split(X, y_encoded, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    y_pred_proba = clf.predict_proba(X_test_scaled)

    auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
    acc = accuracy_score(y_test, y_pred)

    return auc, acc

def create_heatmap(df, metric, metric_name, output_dir):
    pivot_table = df.pivot(index='min_samples', columns='eps', values=metric)
    plt.figure(figsize=(16, 12))
    sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='YlGnBu', cbar_kws={'label': metric_name})
    plt.title(f'Grid Search Heatmap - {metric_name}')
    plt.xlabel('eps')
    plt.ylabel('min_samples')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'heatmap_{metric}.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    input_csv = "path/to/your/input.csv"
    grid_search_heatmap(input_csv)