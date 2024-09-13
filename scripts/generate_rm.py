# scripts/generate_rm.py

import mne
import numpy as np
import torch
from torch.utils.data import DataLoader
import pathlib
from mne.preprocessing import compute_current_source_density
from sklearn.model_selection import train_test_split
import torch.nn as nn
import sys
import os
from tqdm import tqdm

# Import utility functions
from utils import chunk_data, extract_phase, EEGDataset, ConvLSTMEEGAutoencoder

# Disable MNE info messages
mne.set_log_level('ERROR')

def load_data(group_folder, chunk_duration=5.0):
    preprocessed_data = []
    # Get list of subject folders
    subject_folders = [f for f in group_folder.glob('*') if f.is_dir()]
    
    with tqdm(total=len(subject_folders), desc="Loading subjects") as pbar:
        for subject in subject_folders:
            # Get all the .fif files for this subject
            files = list(subject.glob('**/*_good_*_eeg.fif'))
            for file in files:
                raw = mne.io.read_raw_fif(file, preload=True, verbose=False)
                # Apply Laplacian referencing
                raw = compute_current_source_density(raw)
                # Get data and sampling frequency
                data = raw.get_data()
                sfreq = raw.info['sfreq']
                # Split data into chunks
                chunks = chunk_data(data, sfreq, chunk_duration=chunk_duration)
                # Process each chunk
                for chunk in chunks:
                    # Extract phase information
                    phase_chunk = extract_phase(chunk)
                    # Append to list
                    preprocessed_data.append(phase_chunk)
            pbar.update(1)
    return np.stack(preprocessed_data)

def main():
    print("\n🚀 Welcome to the Recurrence Matrix Generator! 👩‍💻👨‍💻\n")
    
    # Set the paths
    data_dir = pathlib.Path('data')
    group_dirs = {
        'AD': data_dir / 'AD',
        'HID': data_dir / 'HID',
        'MCI': data_dir / 'MCI'
    }
    # Choose which group to process
    group_to_process = 'HID'  # Change to 'AD' or 'HID' as needed

    # Load data
    print(f"📊 Loading data for {group_to_process}...")
    all_data = load_data(
        group_folder=group_dirs[group_to_process],
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

    # Load the trained model
    print("🧠 Loading trained model...")
    model = ConvLSTMEEGAutoencoder(n_channels=n_channels, hidden_size=hidden_size).to(device)
    model.load_state_dict(torch.load('models/model.pth', map_location=device))
    model.eval()

    # Initialize a list to store recurrence matrices
    recurrence_matrices = []

    # Epsilon value used during training
    epsilon = np.pi / 4  # Adjust if different

    print("🔄 Generating recurrence matrices...")
    # No need to compute gradients during inference
    with torch.no_grad():
        for batch, mask in tqdm(data_loader, desc="Processing batches"):
            batch = batch.to(device)
            mask = mask.to(device)
            # Forward pass through the model
            reconstructed, recurrence_matrix = model(batch, epsilon)
            # Move recurrence_matrix to CPU and convert to numpy
            recurrence_matrix = recurrence_matrix.cpu().numpy()
            # Append to list
            recurrence_matrices.append(recurrence_matrix)

    # Concatenate the recurrence matrices
    recurrence_matrices = np.concatenate(recurrence_matrices, axis=0)
    print(f"✅ {group_to_process} recurrence matrices shape:", recurrence_matrices.shape)

    # Save the recurrence matrices
    output_file = f'data/{group_to_process}_recurrence_matrices.npy'
    np.save(output_file, recurrence_matrices)
    print(f"💾 Recurrence matrices saved to '{output_file}'")

    print("\n🎉 Recurrence matrix generation complete! 🎉")

if __name__ == '__main__':
    main()