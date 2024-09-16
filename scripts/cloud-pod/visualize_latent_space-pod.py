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
import re
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import pandas as pd
from itertools import cycle
import colorsys
import random

# Import utility functions
from utils import ConvLSTMEEGEncoder

# Disable MNE info messages
mne.set_log_level('ERROR')

# Add this flag at the beginning of the script
COLOR_BY_SUBJECT = True  # Set to True to color by subject, False to color by condition

def generate_distinct_colors(n):
    hue_partition = 1.0 / (n + 1)
    colors = []
    for i in range(n):
        hue = i * hue_partition
        saturation = random.uniform(0.7, 1.0)
        lightness = random.uniform(0.4, 0.6)
        r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append(f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}')
    
    # Shuffle the colors to avoid local similarities
    random.shuffle(colors)
    return colors

# Generate 300 distinct colors
distinct_colors = generate_distinct_colors(300)

def get_distinct_colors(n):
    return distinct_colors[:n]

def extract_base_subject_id(subject_id):
    """Extract the base subject ID, everything before '_good_'"""
    parts = subject_id.split('_good_')
    return parts[0] if len(parts) > 1 else subject_id

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

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize latent space with customizable UMAP parameters")
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

def extract_subject_info(file_path):
    file_name = os.path.basename(file_path)
    # Try to extract subject ID and index
    match = re.search(r'(\w+)_(\d+)', file_name)
    if match:
        subject_id = match.group(1)
        subject_index = match.group(2)
    else:
        # If the pattern doesn't match, use the whole filename as subject_id
        subject_id = os.path.splitext(file_name)[0]
        subject_index = "N/A"
    return subject_id, subject_index

def create_and_save_plot(df, color_by, args, model_name, results_dir):
    fig = px.scatter_3d(
        df,
        x='UMAP1',
        y='UMAP2',
        z='UMAP3',
        color=color_by,
        hover_data=['Condition', 'BaseSubject', 'Subject', 'Index', 'Label', 'original_fif_file'],
        labels={'color': color_by},
    )

    fig.update_traces(marker=dict(size=3))

    fig.update_layout(
        title=f'3D UMAP Projection of Latent Space (Colored by {color_by}, n_neighbors={args.n_neighbors}, min_dist={args.min_dist}, metric={args.metric})',
        scene=dict(
            xaxis_title='UMAP1',
            yaxis_title='UMAP2',
            zaxis_title='UMAP3',
            xaxis=dict(showgrid=False, zeroline=False, showline=False, showticklabels=False, title=''),
            yaxis=dict(showgrid=False, zeroline=False, showline=False, showticklabels=False, title=''),
            zaxis=dict(showgrid=False, zeroline=False, showline=False, showticklabels=False, title=''),
        ),
        width=1024,
        height=1024,
        margin=dict(r=0, b=0, l=0, t=50),
        template="plotly_white"
    )

    html_file = f'{results_dir}/latent_space_umap_{model_name}_n{args.n_neighbors}_d{args.min_dist}_{args.metric}_{color_by}.html'
    fig.write_html(html_file)
    print(f"💾 Interactive plot (colored by {color_by}) saved to '{html_file}'")

def main():
    print("\n🚀 Welcome to the Latent Space Visualizer! 👩‍💻👨‍💻\n")

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
        # Collect all file paths, labels, and subject info
        file_paths = []
        condition_labels = []
        subject_infos = []
        label_mapping = {'AD': 0, 'HID': 1, 'MCI': 2}
        for file_name in os.listdir(output_dir):
            if file_name.endswith('.npy'):
                file_path = os.path.join(output_dir, file_name)
                file_paths.append(file_path)
                group_name = file_name.split('_')[0]
                condition_labels.append(label_mapping.get(group_name, -1))
                subject_id, subject_index = extract_subject_info(file_path)
                subject_infos.append((subject_id, subject_index))

        # Filter out any files with unrecognized group names
        valid_indices = [i for i, label in enumerate(condition_labels) if label != -1]
        file_paths = [file_paths[i] for i in valid_indices]
        condition_labels = [condition_labels[i] for i in valid_indices]
        subject_infos = [subject_infos[i] for i in valid_indices]

    # Load metadata CSV
    metadata_csv_path = os.path.join(output_dir, 'chunk_metadata.csv')
    if os.path.exists(metadata_csv_path):
        metadata_df = pd.read_csv(metadata_csv_path)
        print(f"✅ Loaded metadata CSV from '{metadata_csv_path}'")
    else:
        print(f"❌ Metadata CSV not found at '{metadata_csv_path}'. Please ensure it was generated during preprocessing.")
        sys.exit(1)

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

    # Define condition_names here
    condition_names = {0: 'AD', 1: 'HID', 2: 'MCI'}

    # Create a DataFrame with all metadata
    df = pd.DataFrame({
        'UMAP1': embedding[:, 0],
        'UMAP2': embedding[:, 1],
        'UMAP3': embedding[:, 2],
        'Condition': [condition_names[label] for label in condition_labels],
        'Subject': [info[0] for info in subject_infos],
        'Index': [info[1] for info in subject_infos],
        'BaseSubject': [extract_base_subject_id(info[0]) for info in subject_infos],
        'Label': [f"Condition: {condition_names[label]}, Subject: {extract_base_subject_id(info[0])}, Full ID: {info[0]}, Index: {info[1]}" 
                  for label, info in zip(condition_labels, subject_infos)]
    })

    # After creating the main DataFrame 'df', merge it with 'metadata_df' on 'chunk_file'
    df_metadata = pd.DataFrame({
        'chunk_file': file_paths
    })
    
    # Merge the embedding DataFrame with metadata
    df_combined = pd.concat([df_metadata.reset_index(drop=True), df.reset_index(drop=True)], axis=1)
    df_combined = df_combined.merge(metadata_df, on='chunk_file', how='left')
    
    # Use 'df_combined' for your further analysis and visualization
    df = df_combined  # Replace 'df' with the combined DataFrame

    # Save the DataFrame
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    results_dir = '/workspace/MatrixAutoEncoder/results'
    os.makedirs(results_dir, exist_ok=True)
    df_file = f'{results_dir}/latent_space_data_{model_name}_n{args.n_neighbors}_d{args.min_dist}_{args.metric}.csv'
    df.to_csv(df_file, index=False)
    print(f"💾 DataFrame with all metadata and embeddings saved to '{df_file}'")

    # Create and save plots for both color schemes
    create_and_save_plot(df, 'Condition', args, model_name, results_dir)
    create_and_save_plot(df, 'BaseSubject', args, model_name, results_dir)

    print("\n🎉 Latent space visualization complete! 🎉")

if __name__ == '__main__':
    main()