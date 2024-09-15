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
import time
from dotenv import load_dotenv

# Import utility functions
from utils import chunk_data, extract_phase, EEGDataset, ConvLSTMEEGAutoencoder

# Disable MNE info messages
mne.set_log_level('ERROR')

def parse_args():
    args = {
        'n_subjects_per_group': 2,
        'batch_size': 64,
        'chunk_duration': 5.0,
        'hidden_size': 64,
        'num_epochs': 100,
        'epsilon': np.pi / 16,
        'complexity': 0,
        'checkpoint_frequency': 200,
        'filter_low': 3.0,
        'filter_high': 40.0,
        'use_threshold': False,
        'wandb_project': "EEG-Autoencoder"
    }

    for arg in sys.argv[1:]:
        if '=' in arg:
            key, value = arg.split('=')
            if key in args:
                if key in ['n_subjects_per_group', 'batch_size', 'hidden_size', 'num_epochs', 'complexity', 'checkpoint_frequency']:
                    args[key] = int(value)
                elif key in ['chunk_duration', 'epsilon', 'filter_low', 'filter_high']:
                    args[key] = float(value)
    return args

def print_memory_usage():
    gpu_memory = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    ram_memory = psutil.virtual_memory().used / 1024**2
    print(f"GPU Memory Usage: {gpu_memory:.2f} MB")
    print(f"RAM Usage: {ram_memory:.2f} MB")

def load_data(group_folders, n_subjects_per_group, chunk_duration=5.0, filter_low=3.0, filter_high=40.0):
    preprocessed_data = []
    total_subjects = n_subjects_per_group * len(group_folders)
    sfreq = None

    with tqdm(total=total_subjects, desc="Loading subjects") as pbar:
        for group in group_folders:
            subject_folders = [f for f in group.glob('*') if f.is_dir()]
            subject_folders = subject_folders[:n_subjects_per_group]
            for subject in subject_folders:
                files = list(subject.glob('**/*_good_*_eeg.fif'))
                for file in files:
                    raw = mne.io.read_raw_fif(file, preload=True, verbose=False)
                    raw = raw.filter(l_freq=filter_low, h_freq=filter_high)
                    raw = compute_current_source_density(raw)
                    data = raw.get_data()
                    sfreq = raw.info['sfreq']
                    chunks = chunk_data(data, sfreq, chunk_duration=chunk_duration)
                    for chunk in chunks:
                        phase_chunk = extract_phase(chunk)
                        preprocessed_data.append(phase_chunk)
                pbar.update(1)
    return np.stack(preprocessed_data), sfreq

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

def plot_sample_and_reconstruction(original, reconstructed, recurrence_matrix, channel=0, sfreq=250):
    fig = plt.figure(figsize=(20, 10))  # Increased figure width
    gs = fig.add_gridspec(2, 3, width_ratios=[2, 1, 1], height_ratios=[1, 1])

    # Calculate time values
    n_samples = original.shape[1]
    time = np.arange(n_samples) / sfreq

    # Plot angular distance matrix
    ax1 = fig.add_subplot(gs[:, 0])
    recurrence_matrix = recurrence_matrix.detach().cpu().numpy()
    if recurrence_matrix.ndim == 1:
        size = int(np.sqrt(recurrence_matrix.shape[0]))
        recurrence_matrix = recurrence_matrix[:size*size].reshape(size, size)
    elif recurrence_matrix.ndim > 2:
        recurrence_matrix = recurrence_matrix[0]
    
    im = ax1.imshow(recurrence_matrix, cmap='viridis', aspect='equal')
    ax1.set_title('Encoded Phase Similarity Matrix\nAngular Distances in Latent Space')
    ax1.set_xlabel('Encoded Sequence')
    ax1.set_ylabel('Encoded Sequence')
    ax1.set_xticks([])  # Remove x-axis ticks
    ax1.set_yticks([])  # Remove y-axis ticks
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Angular Distance')

    # Plot reconstructed sample
    ax2 = fig.add_subplot(gs[0, 1:])  # Span two columns
    ax2.plot(time, reconstructed[channel].detach().cpu().numpy() * np.pi)
    ax2.set_title('Reconstructed Signal')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Phase (Φ)')

    # Plot original sample
    ax3 = fig.add_subplot(gs[1, 1:])  # Span two columns
    ax3.plot(time, original[channel].detach().cpu().numpy() * np.pi)
    ax3.set_title('Original Signal')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Phase (Φ)')

    plt.tight_layout()
    return fig

def compute_batch_epsilon(batch, percentile=95):
    with torch.no_grad():
        batch_flat = batch.view(batch.size(0), -1)
        norm = torch.norm(batch_flat, p=2, dim=1, keepdim=True)
        normalized_batch = batch_flat / norm
        cosine_sim = torch.mm(normalized_batch, normalized_batch.t())
        cosine_sim = cosine_sim.clamp(-1 + 1e-7, 1 - 1e-7)
        angular_distances = torch.acos(cosine_sim)
        epsilon = torch.quantile(angular_distances, q=percentile/100)
    return epsilon.item()

def save_checkpoint(model, optimizer, epoch, loss, args, model_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'args': args
    }
    checkpoint_path = f"{model_path[:-4]}_checkpoint_epoch_{epoch}_{time.strftime('%Y%m%d_%H%M%S')}.pth"
    torch.save(checkpoint, checkpoint_path)
    print(f"💾 Checkpoint saved to '{checkpoint_path}'")

def main():
    print("\n🚀 Welcome to the EEG Model Trainer! 👩‍💻👨‍💻\n")

    # Load environment variables
    load_dotenv()

    # Initialize wandb
    wandb_api_key = os.getenv('WANDB_API_KEY')
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
        try:
            wandb.init(project="eeg-autoencoder")
        except UnicodeDecodeError:
            print("⚠️ Warning: Encountered UnicodeDecodeError during wandb initialization.")
            print("Continuing without wandb logging.")
            os.environ['WANDB_DISABLED'] = 'true'
    else:
        print("⚠️ Warning: WANDB_API_KEY not found in .env file. Weights & Biases logging will be disabled.")
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
    print(f"📊 Loading and preprocessing data... - setting fmin: {args['filter_low']} - fmax: {args['filter_high']}")
    all_data, sfreq = load_data(
        group_folders=[group_dirs['AD'], group_dirs['HID'], group_dirs['MCI']],
        n_subjects_per_group=args['n_subjects_per_group'],
        chunk_duration=args['chunk_duration'],
        filter_low=args['filter_low'],
        filter_high=args['filter_high']
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
    percentile = args.get('percentile', 95)
    initial_epsilon = args['epsilon']
    alpha = 0.025  # Smoothing factor for epsilon updates
    complexity = args['complexity']  # Add this line

    # Initialize model
    model = ConvLSTMEEGAutoencoder(n_channels=n_channels, hidden_size=hidden_size, 
                                   initial_epsilon=initial_epsilon, alpha=alpha,
                                   complexity=complexity).to(device)  # Add complexity parameter

    # Disable thresholding
    model.use_threshold = args['use_threshold']
    print(f"🧮 Use threshold: {model.use_threshold}")

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

    print(f"🏋️ Starting training for {num_epochs} epochs...")
    for epoch in tqdm(range(num_epochs), desc="Training progress"):
        model.train()
        train_loss = 0
        for batch_idx, (batch, mask) in enumerate(train_loader):
            batch = batch.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            
            # Compute new epsilon for this batch
            new_epsilon = compute_batch_epsilon(batch, percentile)
            
            # Update model's epsilon (smoothed update)
            model.update_epsilon(new_epsilon)
            
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

            # Log to wandb every 128 steps
            if batch_idx % 128 == 0: # 
                try:
                    wandb.log({
                        "train_loss": loss.item(),
                        "epoch": epoch,
                        "batch": batch_idx,
                        "epsilon": model.epsilon.item()
                    })

                    # Create and log visualization
                    try:
                        fig = plot_sample_and_reconstruction(batch[0], reconstructed[0], recurrence_matrix[0], sfreq=sfreq)
                        wandb.log({"sample_visualization": wandb.Image(fig)})
                        plt.close(fig)
                    except Exception as e:
                        print(f"⚠️ Warning: Failed to create or log visualization: {e}")
                except Exception as e:
                    print(f"⚠️ Warning: Failed to log to wandb: {e}")

        avg_train_loss = train_loss / len(train_loader)
        tqdm.write(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.4f}, Epsilon: {model.epsilon.item():.4f}')
        if 'WANDB_DISABLED' not in os.environ:
            try:
                wandb.log({"avg_train_loss": avg_train_loss, "epoch": epoch, "epsilon": model.epsilon.item()})
            except Exception as e:
                print(f"⚠️ Warning: Failed to log average loss to wandb: {e}")

        # Save checkpoint
        if (epoch + 1) % args['checkpoint_frequency'] == 0:
            save_checkpoint(model, optimizer, epoch + 1, avg_train_loss, args, model_path)

    # Save the trained model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"💾 Model saved to '{model_path}'")

    print("\n🎉 Training complete! 🎉")


if __name__ == '__main__':
    main()