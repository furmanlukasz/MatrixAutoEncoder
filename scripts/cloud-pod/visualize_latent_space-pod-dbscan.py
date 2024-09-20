# scripts/visualize_latent_space.py

# Place this at the very top of your script to limit the number of threads
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import mne
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pathlib
from sklearn.model_selection import train_test_split
import torch.nn as nn
import umap
import plotly.graph_objects as go
import sys
from tqdm import tqdm
import argparse
from multiprocessing import cpu_count
from torch.cuda.amp import autocast, GradScaler
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import pandas as pd
from sklearn.cluster import DBSCAN

# Import utility functions
from utils import ConvLSTMEEGEncoder

# Disable MNE info messages
mne.set_log_level('ERROR')

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize latent space with cluster analysis")
    parser.add_argument('--n_neighbors', type=int, choices=[5, 10, 15], default=10,
                        help='Number of neighbors for UMAP (5, 10, or 15)')
    parser.add_argument('--min_dist', type=float, choices=[0.05, 0.25, 0.5], default=0.25,
                        help='Minimum distance for UMAP (0.05, 0.25, or 0.5)')
    parser.add_argument('--metric', type=str, choices=['cosine', 'euclidean', 'correlation'], default='cosine',
                        help='Metric for UMAP (cosine, euclidean, or correlation)')
    parser.add_argument('--complexity', type=int, choices=[0, 1, 2, 3], default=0,
                        help='Model complexity (0, 1, or 2)')
    parser.add_argument('--n_subjects_per_group', type=int, default=100,
                        help='Number of subjects per group to process')
    parser.add_argument('--chunk_duration', type=float, default=2.0,
                        help='Duration of each data chunk in seconds')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for DataLoader')
    parser.add_argument('--dbscan_eps', type=float, default=0.1,
                        help='The maximum distance between two samples for DBSCAN to consider them as in the same neighborhood')
    parser.add_argument('--dbscan_min_samples', type=int, default=3,
                        help='The number of samples (or total weight) in a neighborhood for a point to be considered as a core point in DBSCAN')
    return parser.parse_args()

def select_model():
    models = list(pathlib.Path('/workspace/MatrixAutoEncoder/models').glob('*.pth'))
    if not models:
        print("❌ No models found in the '/workspace/MatrixAutoEncoder/models' directory.")
        sys.exit(1)

    print("📚 Available models:")
    for i, model in enumerate(models, 1):
        print(f"  {i}. 🤖 {model.name}")

    while True:
        try:
            choice = int(input("\n🔢 Enter the number of the model you want to visualize: "))
            if 1 <= choice <= len(models):
                return models[choice - 1]
            else:
                print("❌ Invalid choice. Please try again.")
        except ValueError:
            print("❌ Please enter a valid number.")

class EEGDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = np.load(self.file_paths[idx])
        label = self.labels[idx]
        if self.transform:
            data = self.transform(data)
        return torch.tensor(data, dtype=torch.float32), label

def main():
    print("\n🚀 Welcome to the Latent Space Visualizer with Clustering! 👩‍💻👨‍💻\n")

    # Parse command-line arguments
    args = parse_args()

    # Set the paths
    data_dir = pathlib.Path('/workspace/MatrixAutoEncoder/data')
    print(f"📂 Data directory: {data_dir}")
    group_dirs = {
        'AD': data_dir / 'AD',
        'HID': data_dir / 'HID',
        'MCI': data_dir / 'MCI'
    }

    # Use the same preprocessed data directory as in train_model.py
    output_dir = '/workspace/MatrixAutoEncoder/preprocessed_data'

    # Check if preprocessed data exists
    if not os.path.exists(output_dir) or len(os.listdir(output_dir)) == 0:
        print("❌ Preprocessed data not found. Please run 'train_model.py' first to generate preprocessed data.")
        sys.exit(1)
    else:
        print("🗂️ Using preprocessed data from 'train_model.py'.")
        # Collect all file paths and labels
        file_paths = []
        condition_labels = []
        label_mapping = {'AD': 0, 'HID': 1, 'MCI': 2}
        for file_name in os.listdir(output_dir):
            if file_name.endswith('.npy'):
                file_path = os.path.join(output_dir, file_name)
                file_paths.append(file_path)
                group_name = file_name.split('_')[0]
                condition_labels.append(label_mapping.get(group_name, -1))

        # Filter out any files with unrecognized group names
        valid_indices = [i for i, label in enumerate(condition_labels) if label != -1]
        file_paths = [file_paths[i] for i in valid_indices]
        condition_labels = [condition_labels[i] for i in valid_indices]

    # Load metadata CSV
    metadata_csv_path = os.path.join(output_dir, 'chunk_metadata.csv')
    if os.path.exists(metadata_csv_path):
        metadata_df = pd.read_csv(metadata_csv_path)
        print(f"✅ Loaded metadata CSV from '{metadata_csv_path}'")
    else:
        print(f"❌ Metadata CSV not found at '{metadata_csv_path}'. Please ensure it was generated during preprocessing.")
        sys.exit(1)

    # Make sure the metadata corresponds to the file paths
    metadata_df = metadata_df[metadata_df['chunk_file'].isin(file_paths)]

    # Create Dataset and DataLoader
    batch_size = args.batch_size
    dataset = EEGDataset(file_paths, condition_labels)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(8, cpu_count()),
        pin_memory=True
    )

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")

    # Initialize the GradScaler for mixed-precision inference
    scaler = GradScaler()

    # Model parameters
    sample_data, _ = dataset[0]
    n_channels = sample_data.shape[0]
    hidden_size = 64  # Should match the trained model's hidden size
    complexity = args.complexity

    # Load the trained encoder
    print("🧠 Loading trained encoder...")
    model_path = select_model()
    encoder = ConvLSTMEEGEncoder(n_channels=n_channels, hidden_size=hidden_size, complexity=complexity).to(device)
    model_state_dict = torch.load(model_path, map_location=device)
    if 'model_state_dict' in model_state_dict:
        encoder_state_dict = {k.replace('encoder.', ''): v for k, v in model_state_dict['model_state_dict'].items() if 'encoder.' in k}
    else:
        encoder_state_dict = {k.replace('encoder.', ''): v for k, v in model_state_dict.items() if 'encoder.' in k}
    encoder.load_state_dict(encoder_state_dict)
    encoder.eval()
    print(f"✅ Loaded model: {os.path.basename(model_path)}")

    # Initialize a list to store latent representations
    latent_representations = []

    print("🔄 Processing data through encoder...")
    # No need to compute gradients during inference
    with torch.no_grad():
        for batch, _ in tqdm(data_loader, desc="Processing batches"):
            batch = batch.to(device, non_blocking=True)
            # Use autocast for mixed-precision inference
            with autocast():
                lstm_out = encoder(batch)
                latent_vec = lstm_out.mean(dim=1)
            latent_vec = latent_vec.cpu().numpy()
            latent_representations.append(latent_vec)

    # Concatenate all latent representations
    latent_representations = np.concatenate(latent_representations, axis=0)
    print("✅ Latent representations shape:", latent_representations.shape)
    print("✅ Total samples:", len(condition_labels))

    # Apply UMAP to reduce dimensions to 3D
    print("🔬 Applying UMAP dimensionality reduction...")
    reducer = umap.UMAP(n_components=3,
                        n_neighbors=args.n_neighbors,
                        min_dist=args.min_dist,
                        metric=args.metric)
    embedding = reducer.fit_transform(latent_representations)
    print("✅ UMAP embedding shape:", embedding.shape)

    # Define condition names
    condition_names = {0: 'AD', 1: 'HID', 2: 'MCI'}

    # Create a DataFrame with all metadata
    df = pd.DataFrame({
        'chunk_file': file_paths,
        'UMAP1': embedding[:, 0],
        'UMAP2': embedding[:, 1],
        'UMAP3': embedding[:, 2],
        'Condition': [condition_names[label] for label in condition_labels],
    })

    # Merge with metadata_df
    df = df.merge(metadata_df, on='chunk_file', how='left')

    # Add labels for plotting
    df['Label'] = df.apply(lambda row: f"Condition: {row['Condition']}, Subject: {row['subject']}, File: {os.path.basename(row['original_fif_file'])}, Chunk Index: {row['chunk_index']}", axis=1)

    # Save the DataFrame
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    results_dir = '/workspace/MatrixAutoEncoder/results'
    os.makedirs(results_dir, exist_ok=True)
    df_file = f'{results_dir}/latent_space_data_{model_name}_n{args.n_neighbors}_d{args.min_dist}_{args.metric}.csv'
    df.to_csv(df_file, index=False)
    print(f"💾 DataFrame with all metadata and embeddings saved to '{df_file}'")

    # Perform clustering using DBSCAN
    print("\n🔍 Performing cluster analysis using DBSCAN...")
    dbscan = DBSCAN(eps=args.dbscan_eps, min_samples=args.dbscan_min_samples)
    df['Cluster'] = dbscan.fit_predict(df[['UMAP1', 'UMAP2', 'UMAP3']])
    print(f"✅ Cluster analysis complete!")

    # Identify clusters with mixed conditions
    print("📝 Identifying clusters with mixed conditions...")
    cluster_condition_counts = df.groupby('Cluster')['Condition'].nunique()
    mixed_clusters = cluster_condition_counts[cluster_condition_counts > 1].index.tolist()
    print(f"Clusters with mixed conditions: {mixed_clusters}")

    # Mark clusters for exclusion
    df['ForFurtherAnalysis'] = ~df['Cluster'].isin(mixed_clusters)

    # Save the updated DataFrame
    df_file_updated = f'{results_dir}/latent_space_data_{model_name}_n{args.n_neighbors}_d{args.min_dist}_{args.metric}_clusters.csv'
    df.to_csv(df_file_updated, index=False)
    print(f"💾 Updated DataFrame with clustering information saved to '{df_file_updated}'")

    # Plotting
    print("\n📊 Creating plots...")

    # Plot with all data points
    fig_all = px.scatter_3d(
        df,
        x='UMAP1',
        y='UMAP2',
        z='UMAP3',
        color='Condition',
        hover_data=['Label'],
        labels={'color': 'Condition'},
        title=f'3D UMAP Projection - All Data (n_neighbors={args.n_neighbors}, min_dist={args.min_dist}, metric={args.metric})'
    )
    fig_all.update_traces(marker=dict(size=3))
    fig_all.update_layout(width=1000, height=800)
    html_file_all = f'{results_dir}/latent_space_umap_{model_name}_all_data.html'
    fig_all.write_html(html_file_all)
    print(f"💾 Plot with all data saved to '{html_file_all}'")

    # Filter data for further analysis
    filtered_df = df[df['ForFurtherAnalysis']]
    print(f"✅ Filtered data contains {len(filtered_df)} samples.")

    # Plot with filtered data
    fig_filtered = px.scatter_3d(
        filtered_df,
        x='UMAP1',
        y='UMAP2',
        z='UMAP3',
        color='Condition',
        hover_data=['Label'],
        labels={'color': 'Condition'},
        title=f'3D UMAP Projection - Filtered Data (n_neighbors={args.n_neighbors}, min_dist={args.min_dist}, metric={args.metric})'
    )
    fig_filtered.update_traces(marker=dict(size=3))
    fig_filtered.update_layout(width=1000, height=800)
    html_file_filtered = f'{results_dir}/latent_space_umap_{model_name}_filtered_data.html'
    fig_filtered.write_html(html_file_filtered)
    print(f"💾 Plot with filtered data saved to '{html_file_filtered}'")

    print("\n🎉 Latent space visualization with clustering complete! 🎉")

if __name__ == '__main__':
    main()