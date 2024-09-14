# scripts/visualize_latent_space.py

import mne
import numpy as np
import torch
from torch.utils.data import DataLoader
import pathlib
from mne.preprocessing import compute_current_source_density
from sklearn.model_selection import train_test_split
import torch.nn as nn
import umap
import plotly.graph_objects as go
import sys
import os
from tqdm import tqdm
import argparse
import glob

# Import utility functions
from utils import chunk_data, extract_phase, EEGDataset, ConvLSTMEEGEncoder

# Disable MNE info messages
mne.set_log_level('ERROR')

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
    return parser.parse_args()

def load_data_with_labels(group_dirs, n_subjects_per_group=None, chunk_duration=5.0):
    all_data = []
    all_labels = []
    label_mapping = {'AD': 0, 'HID': 1, 'MCI': 2}
    
    total_subjects = sum([len(list(group_dir.glob('*'))) for group_dir in group_dirs.values()])
    if n_subjects_per_group:
        total_subjects = min(total_subjects, n_subjects_per_group * len(group_dirs))
    
    with tqdm(total=total_subjects, desc="Loading subjects") as pbar:
        for group_name, group_dir in group_dirs.items():
            subject_folders = [f for f in group_dir.glob('*') if f.is_dir()]
            if n_subjects_per_group is not None:
                subject_folders = subject_folders[:n_subjects_per_group]
            for subject in subject_folders:
                files = list(subject.glob('**/*_good_*_eeg.fif'))
                for file in files:
                    raw = mne.io.read_raw_fif(file, preload=True, verbose=False)
                    raw = compute_current_source_density(raw)
                    data = raw.get_data()
                    sfreq = raw.info['sfreq']
                    chunks = chunk_data(data, sfreq, chunk_duration=chunk_duration)
                    for chunk in chunks:
                        phase_chunk = extract_phase(chunk)
                        all_data.append(phase_chunk)
                        all_labels.append(label_mapping[group_name])
                pbar.update(1)
    return np.stack(all_data), np.array(all_labels)

def select_model():
    models = glob.glob('models/*.pth')
    if not models:
        print("❌ No models found in the 'models' directory.")
        sys.exit(1)
    
    print("📚 Available models:")
    for i, model in enumerate(models, 1):
        print(f"  {i}. 🤖 {os.path.basename(model)}")
    
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
    data_dir = pathlib.Path('data')
    group_dirs = {
        'AD': data_dir / 'AD',
        'HID': data_dir / 'HID',
        'MCI': data_dir / 'MCI'
    }
    # Number of subjects per group to use (set to None to use all subjects)
    n_subjects_per_group = None  # Adjust as needed

    # Load data and labels
    print("📊 Loading data and labels...")
    all_data, all_labels = load_data_with_labels(
        group_dirs=group_dirs,
        n_subjects_per_group=n_subjects_per_group,
        chunk_duration=5.0
    )

    # Create Dataset and DataLoader
    batch_size = 64
    dataset = EEGDataset(all_data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Device configuration
    device = torch.device('mps')
    print(f"🖥️ Using device: {device}")

    # Model parameters
    n_channels = all_data.shape[1]
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
    print(f"📊 Model structure: {'model_state_dict' in model_state_dict}")
    print(f"🔢 Number of encoder parameters: {len(encoder_state_dict)}")

    # Initialize a list to store latent representations
    latent_representations = []

    print("🔄 Processing data through encoder...")
    # No need to compute gradients during inference
    with torch.no_grad():
        for batch, mask in tqdm(data_loader, desc="Processing batches"):
            batch = batch.to(device)
            lstm_out = encoder(batch)
            latent_vec = lstm_out.mean(dim=1)
            latent_vec = latent_vec.cpu().numpy()
            latent_representations.append(latent_vec)

    # Concatenate all latent representations
    latent_representations = np.concatenate(latent_representations, axis=0)
    print("✅ Latent representations shape:", latent_representations.shape)
    print("✅ Total samples:", len(all_labels))

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
    embedding_file = f'results/umap_embeddings_{model_name}_n{args.n_neighbors}_d{args.min_dist}_{args.metric}.npy'
    np.save(embedding_file, embedding)
    print(f"💾 UMAP embeddings saved to '{embedding_file}'")

    # Save labels to file
    labels_file = 'results/labels.npy'
    np.save(labels_file, all_labels)
    print(f"💾 Labels saved to '{labels_file}'")

    # Create color mapping
    unique_labels = np.unique(all_labels)
    label_to_color = {label: idx / len(unique_labels) for idx, label in enumerate(unique_labels)}
    color_values = [label_to_color[label] for label in all_labels]

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
        text=[f"Condition: {label}" for label in all_labels],
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
        template="plotly_dark"
    )

    # Save the plot as an HTML file
    html_file = f'results/latent_space_umap_{model_name}_n{args.n_neighbors}_d{args.min_dist}_{args.metric}.html'
    fig.write_html(html_file)
    print(f"💾 Interactive plot saved to '{html_file}'")

    # Open in default browser
    import webbrowser
    print(f"🌐 Opening {html_file} in default browser...")
    webbrowser.open('file://' + os.path.abspath(html_file))

    print("\n🎉 Latent space visualization complete! 🎉")

if __name__ == '__main__':
    main()