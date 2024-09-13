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

# Import utility functions
from utils import chunk_data, extract_phase, EEGDataset, ConvLSTMEEGEncoder

# Disable MNE info messages
mne.set_log_level('ERROR')

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

def main():
    print("\n🚀 Welcome to the Latent Space Visualizer! 👩‍💻👨‍💻\n")

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
    device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")

    # Model parameters
    n_channels = all_data.shape[1]
    hidden_size = 64  # Should match the trained model's hidden size

    # Load the trained encoder
    print("🧠 Loading trained encoder...")
    encoder = ConvLSTMEEGEncoder(n_channels=n_channels, hidden_size=hidden_size).to(device)
    model_state_dict = torch.load('models/model.pth', map_location=device)
    encoder_state_dict = {k.replace('encoder.', ''): v for k, v in model_state_dict.items() if 'encoder.' in k}
    encoder.load_state_dict(encoder_state_dict)
    encoder.eval()

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
    reducer = umap.UMAP(n_components=3)
    embedding = reducer.fit_transform(latent_representations)
    print("✅ UMAP embedding shape:", embedding.shape)

    # Save UMAP embeddings to file
    embedding_file = 'results/umap_embeddings.npy'
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
        title='3D UMAP Projection of Latent Space',
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
    html_file = 'results/latent_space_umap.html'
    fig.write_html(html_file)
    print(f"💾 Interactive plot saved to '{html_file}'")

    # Open in default browser
    import webbrowser
    print(f"🌐 Opening {html_file} in default browser...")
    webbrowser.open('file://' + os.path.abspath(html_file))

    print("\n🎉 Latent space visualization complete! 🎉")

if __name__ == '__main__':
    main()