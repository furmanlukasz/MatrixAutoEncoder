# scripts/train_model.py

import mne
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pathlib
from mne.preprocessing import compute_current_source_density
from scipy.signal import hilbert
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
from wandb.errors import CommError
import psutil

# Import utility functions
from utils import chunk_data, extract_phase, EEGDataset, ConvLSTMEEGAutoencoder

# Disable MNE info messages
mne.set_log_level('ERROR')

def parse_args():
    args = {
        'n_subjects_per_group': 10,
        'batch_size': 64,  # Increased from 64
        'chunk_duration': 5.0,
        'hidden_size': 64,
        'num_epochs': 90,
        'epsilon': np.pi / 4
    }

    for arg in sys.argv[1:]:
        if '=' in arg:
            key, value = arg.split('=')
            if key in args:
                if key in ['n_subjects_per_group', 'batch_size', 'hidden_size', 'num_epochs']:
                    args[key] = int(value)
                elif key in ['chunk_duration', 'epsilon']:
                    args[key] = float(value)

    return args

def print_memory_usage():
    gpu_memory = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    ram_memory = psutil.virtual_memory().used / 1024**2
    print(f"GPU Memory Usage: {gpu_memory:.2f} MB")
    print(f"RAM Usage: {ram_memory:.2f} MB")

def load_data(group_folders, n_subjects_per_group, chunk_duration=5.0):
    preprocessed_data = []
    total_subjects = n_subjects_per_group * len(group_folders)
    
    with tqdm(total=total_subjects, desc="Loading subjects") as pbar:
        for group in group_folders:
            subject_folders = [f for f in group.glob('*') if f.is_dir()]
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
                        preprocessed_data.append(phase_chunk)
                pbar.update(1)
    return np.stack(preprocessed_data)

def load_or_initialize_model(model, model_path):
    if os.path.exists(model_path):
        print(f"🔍 Existing model found at {model_path}")
        while True:
            choice = input("Do you want to fine-tune the existing model? (yes/no): ").lower()
            if choice in ['yes', 'y']:
                print("🔄 Loading existing model for fine-tuning...")
                model.load_state_dict(torch.load(model_path))
                return model, True
            elif choice in ['no', 'n']:
                print("🆕 Starting training from scratch...")
                return model, False
            else:
                print("❌ Invalid input. Please enter 'yes' or 'no'.")
    else:
        print("🆕 No existing model found. Starting training from scratch...")
        return model, False

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_sample_and_reconstruction(original, reconstructed, recurrence_matrix, channel=0):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Main title
    fig.suptitle(f'Sample and Reconstruction for Channel {channel}')
    # Plot original sample
    ax1.plot(original[channel].cpu().numpy())
    ax1.set_title('Original')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Amplitude')
    
    # Plot reconstructed sample
    ax2.plot(reconstructed[channel].detach().cpu().numpy())
    ax2.set_title('Reconstructed')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Amplitude')
    
    # Plot recurrence matrix
    recurrence_matrix = recurrence_matrix.detach().cpu().numpy()
    if recurrence_matrix.ndim == 1:
        # If 1D, reshape to 2D square matrix
        size = int(np.sqrt(recurrence_matrix.shape[0]))
        recurrence_matrix = recurrence_matrix[:size*size].reshape(size, size)
    elif recurrence_matrix.ndim > 2:
        # If more than 2D, take the first 2D slice
        recurrence_matrix = recurrence_matrix[0]
    
    im = ax3.imshow(recurrence_matrix, cmap='viridis')
    ax3.set_title('Recurrence Matrix for all channels')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Time')
    plt.colorbar(im, ax=ax3)
    
    plt.tight_layout()
    return fig

def main():
    print("\n🚀 Welcome to the EEG Model Trainer! 👩‍💻👨‍💻\n")

    # Initialize wandb
    wandb.login(key="a32849d94af9c711d39d425741579db87e1f192c")
    
    try:
        wandb.init(project="EEG-Autoencoder")
    except UnicodeDecodeError:
        print("⚠️ Warning: Encountered UnicodeDecodeError during wandb initialization. Some logging features may be limited.")
        os.environ['WANDB_DISABLED'] = 'true'

    args = parse_args()
    if 'WANDB_DISABLED' not in os.environ:
        wandb.config.update(args)  # Log hyperparameters

    # Set the paths
    data_dir = pathlib.Path('data')
    print(f"📂 Data directory: {data_dir}")
    group_dirs = {
        'AD': data_dir / 'AD',
        'HID': data_dir / 'HID',
        'MCI': data_dir / 'MCI'
    }

    # Load data
    print("📊 Loading and preprocessing data...")
    all_data = load_data(
        group_folders=[group_dirs['AD'], group_dirs['HID'], group_dirs['MCI']],
        n_subjects_per_group=args['n_subjects_per_group'],
        chunk_duration=args['chunk_duration']
    )

    # Split data into training and testing sets
    print("🔪 Splitting data into train and test sets...")
    x_train, x_test = train_test_split(all_data, test_size=0.2, random_state=42)

    # Create Datasets and DataLoaders
    batch_size = args['batch_size']
    train_dataset = EEGDataset(x_train)
    test_dataset = EEGDataset(x_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"🧮 Batch size: {batch_size}")
    print_memory_usage()

    # Device configuration
    # device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
    device = torch.device('mps')
    print(f"🖥️ Using device: {device}")

    # Model parameters
    n_channels = x_train.shape[1]
    hidden_size = args['hidden_size']

    # Initialize model
    model = ConvLSTMEEGAutoencoder(n_channels=n_channels, hidden_size=hidden_size).to(device)

    # Check for existing model and ask user for fine-tuning
    model_path = 'models/model.pth'
    model, is_fine_tuning = load_or_initialize_model(model, model_path)

    # Print number of trainable parameters
    num_params = count_trainable_parameters(model)
    print(f"🧮 Number of trainable parameters: {num_params:,}")
    if 'WANDB_DISABLED' not in os.environ:
        wandb.log({"num_trainable_parameters": num_params})

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    num_epochs = args['num_epochs']
    epsilon = args['epsilon']

    print(f"🏋️ Starting training for {num_epochs} epochs...")
    for epoch in tqdm(range(num_epochs), desc="Training progress"):
        model.train()
        train_loss = 0
        for batch_idx, (batch, mask) in enumerate(train_loader):
            batch = batch.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            
            # Update epsilon based on the current batch
            model.update_epsilon(model.encoder(batch), max_chunk_size=500)  # Adjust max_chunk_size as needed
            
            # Forward pass
            reconstructed, recurrence_matrix = model(batch)
            
            # Apply mask to handle padding
            masked_reconstructed = reconstructed * mask
            masked_batch = batch * mask
            
            # Compute reconstruction loss
            loss = criterion(masked_reconstructed, masked_batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Log to wandb every 5 steps
            if batch_idx % 5 == 0 and 'WANDB_DISABLED' not in os.environ:
                try:
                    wandb.log({
                        "train_loss": loss.item(),
                        "epoch": epoch,
                        "batch": batch_idx,
                        "epsilon": model.epsilon.item()
                    })

                    # Create and log visualization
                    try:
                        fig = plot_sample_and_reconstruction(batch[0], reconstructed[0], recurrence_matrix[0])
                        wandb.log({"sample_visualization": wandb.Image(fig)})
                        plt.close(fig)
                    except Exception as e:
                        print(f"⚠️ Warning: Failed to create or log visualization: {e}")

                except Exception as e:
                    print(f"⚠️ Warning: Failed to log to wandb: {e}")

            if batch_idx % 5 == 0:
                print_memory_usage()

        avg_train_loss = train_loss / len(train_loader)
        tqdm.write(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.4f}, Epsilon: {model.epsilon.item():.4f}')
        if 'WANDB_DISABLED' not in os.environ:
            try:
                wandb.log({"avg_train_loss": avg_train_loss, "epoch": epoch, "epsilon": model.epsilon.item()})
            except Exception as e:
                print(f"⚠️ Warning: Failed to log average loss to wandb: {e}")

    # Save the trained model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"💾 Model saved to '{model_path}'")

    # Log the final model to wandb
    if 'WANDB_DISABLED' not in os.environ:
        try:
            wandb.save(model_path)
        except Exception as e:
            print(f"⚠️ Warning: Failed to save model to wandb: {e}")

    print("\n🎉 Training complete! 🎉")
    if 'WANDB_DISABLED' not in os.environ:
        wandb.finish()

if __name__ == '__main__':
    main()