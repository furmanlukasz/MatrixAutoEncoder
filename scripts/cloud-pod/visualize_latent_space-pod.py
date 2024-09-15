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

# Import utility functions
from utils import ConvLSTMEEGEncoder

# Disable MNE info messages
mne.set_log_level('ERROR')

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
    parser.add_argument('--complexity', type=int, choices=[0, 1, 2], default=0,
                        help='Model complexity (0, 1, or 2)')
    parser.add_argument('--n_subjects_per_group', type=int, default=100,
                        help='Number of subjects per group to process')
    parser.add_argument('--chunk_duration', type=float, default=5.0,
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
        # Collect all file paths and labels
        file_paths = []
        labels = []
        label_mapping = {'AD': 0, 'HID': 1, 'MCI': 2}
        for file_name in os.listdir(output_dir):
            if file_name.endswith('.npy'):
                file_paths.append(os.path.join(output_dir, file_name))
                group_name = file_name.split('_')[0]
                labels.append(label_mapping.get(group_name, -1))  # Default to -1 if group not recognized

        # Filter out any files with unrecognized group names
        valid_indices = [i for i, label in enumerate(labels) if label != -1]
        file_paths = [file_paths[i] for i in valid_indices]
        labels = [labels[i] for i in valid_indices]

    # Create Dataset and DataLoader
    batch_size = args.batch_size
    dataset = EEGDataset(file_paths, labels)
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
    labels = np.array(labels)
    print("✅ Latent representations shape:", latent_representations.shape)
    print("✅ Total samples:", len(labels))

    # Apply UMAP to reduce dimensions to 3D
    print("🔬 Applying UMAP dimensionality reduction...")
    reducer = umap.UMAP(n_components=3,
                        n_neighbors=args.n_neighbors,
                        min_dist=args.min_dist,
                        metric=args.metric)
    embedding = reducer.fit_transform(latent_representations)
    print("✅ UMAP embedding shape:", embedding.shape)

    # Save UMAP embeddings to file
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    results_dir = '/workspace/MatrixAutoEncoder/results'
    os.makedirs(results_dir, exist_ok=True)
    embedding_file = f'{results_dir}/umap_embeddings_{model_name}_n{args.n_neighbors}_d{args.min_dist}_{args.metric}.npy'
    np.save(embedding_file, embedding)
    print(f"💾 UMAP embeddings saved to '{embedding_file}'")

    # Save labels to file
    labels_file = f'{results_dir}/labels_{model_name}.npy'
    np.save(labels_file, labels)
    print(f"💾 Labels saved to '{labels_file}'")

    # Create color mapping
    unique_labels = np.unique(labels)
    label_to_color = {label: idx / len(unique_labels) for idx, label in enumerate(unique_labels)}
    color_values = [label_to_color[label] for label in labels]

    # Create label names
    label_names = {0: 'AD', 1: 'HID', 2: 'MCI'}
    label_texts = [label_names.get(label, 'Unknown') for label in labels]

    print("🎨 Creating interactive 3D scatter plot...")
    # Create an interactive 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=embedding[:, 0],
        y=embedding[:, 1],
        z=embedding[:, 2],
        mode='markers',
        marker=dict(
            size=2.75,
            color=color_values,
            colorscale='Viridis',
            opacity=0.8
        ),
        text=[f"Condition: {label_text}" for label_text in label_texts],
        hoverinfo='text'
    )])

    # Update layout
    fig.update_layout(
        title=f'3D UMAP Projection of Latent Space (n_neighbors={args.n_neighbors}, min_dist={args.min_dist}, metric={args.metric})',
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

    # Save the plot as an HTML file
    html_file = f'{results_dir}/latent_space_umap_{model_name}_n{args.n_neighbors}_d{args.min_dist}_{args.metric}.html'
    fig.write_html(html_file)
    print(f"💾 Interactive plot saved to '{html_file}'")

    # Open in default browser
    import webbrowser
    print(f"🌐 Opening {html_file} in default browser...")
    webbrowser.open('file://' + os.path.abspath(html_file))

    print("\n🎉 Latent space visualization complete! 🎉")

if __name__ == '__main__':
    main()