# scripts/train-model-on-preproc-raw.py

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
import sys
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import psutil
import time
from dotenv import load_dotenv
from torch.cuda.amp import GradScaler, autocast
from multiprocessing import cpu_count

# Import utility functions
from utils import extract_phase, ConvLSTMEEGAutoencoder

class EEGDataset(Dataset):
    def __init__(self, file_paths, filter_low, filter_high, transform=None, max_length=1250):
        self.file_paths = file_paths
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = np.load(self.file_paths[idx])
        # Apply filtering
        filtered_data = mne.filter.filter_data(data, sfreq=250, l_freq=self.filter_low, h_freq=self.filter_high)
        # Extract phase
        phase_data = extract_phase(filtered_data)
        # Pad or truncate to max_length
        if phase_data.shape[1] < self.max_length:
            pad_width = ((0, 0), (0, self.max_length - phase_data.shape[1]))
            phase_data = np.pad(phase_data, pad_width, mode='constant', constant_values=0)
        else:
            phase_data = phase_data[:, :self.max_length]
        if self.transform:
            phase_data = self.transform(phase_data)
        # Scale the phase data from [-π, π] to [-1, 1]
        phase_data = phase_data / np.pi
        return torch.tensor(phase_data, dtype=torch.float32)

# Disable MNE info messages
mne.set_log_level('ERROR')

def parse_args():
    args = {
        'batch_size': 64,
        'hidden_size': 64,
        'num_epochs': 6000,
        'epsilon': np.pi / 16,
        'complexity': 2,
        'checkpoint_frequency': 50,
        'filter_low': 3.0,
        'filter_high': 16.0,
        'use_threshold': False,
        'wandb_project': "EEG-Autoencoder"
    }

    for arg in sys.argv[1:]:
        if '=' in arg:
            key, value = arg.split('=')
            if key in args:
                if key in ['batch_size', 'hidden_size', 'num_epochs', 'complexity', 'checkpoint_frequency']:
                    args[key] = int(value)
                elif key in ['epsilon', 'filter_low', 'filter_high']:
                    args[key] = float(value)
    return args

def print_memory_usage():
    gpu_memory = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    ram_memory = psutil.virtual_memory().used / 1024**2
    print(f"GPU Memory Usage: {gpu_memory:.2f} MB")
    print(f"RAM Usage: {ram_memory:.2f} MB")

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
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 3, width_ratios=[2, 1, 1], height_ratios=[1, 1])

    n_samples = original.shape[1]
    time = np.arange(n_samples) / sfreq

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
    ax1.set_xticks([])
    ax1.set_yticks([])
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Angular Distance')

    ax2 = fig.add_subplot(gs[0, 1:])
    ax2.plot(time, reconstructed[channel].detach().cpu().numpy() * np.pi)
    ax2.set_title('Reconstructed Signal')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Phase (Φ)')

    ax3 = fig.add_subplot(gs[1, 1:])
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

def custom_collate(batch):
    # Find the minimum length in the batch
    min_length = min([item.shape[1] for item in batch])
    # Truncate all items to the minimum length
    batch = [item[:, :min_length] for item in batch]
    # Stack the batch
    return torch.stack(batch)

def main():
    print("\n🚀 Welcome to the EEG Model Trainer! 👩‍💻👨‍💻\n")

    # Load environment variables
    load_dotenv()

    # Initialize wandb
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
        try:
            wandb.init(project="eeg-autoencoder")
        except Exception as e:
            print(f"⚠️ Warning: Encountered an error during wandb initialization: {e}")
            print("Continuing without wandb logging.")
            os.environ['WANDB_DISABLED'] = 'true'
    else:
        print("⚠️ Warning: WANDB_API_KEY not found in .env file. Weights & Biases logging will be disabled.")
        os.environ['WANDB_DISABLED'] = 'true'

    args = parse_args()
    if 'WANDB_DISABLED' not in os.environ:
        wandb.config.update(args)

    # Set the paths
    preprocessed_dir = pathlib.Path('/workspace/MatrixAutoEncoder/preprocessed_data')
    print(f"📂 Preprocessed data directory: {preprocessed_dir}")

    # Collect all file paths
    file_paths = [str(f) for f in preprocessed_dir.glob('*.npy')]
    if not file_paths:
        print("❌ No preprocessed data found. Please run the preprocessing script first.")
        sys.exit(1)

    # Split file paths into training and testing
    train_files, test_files = train_test_split(file_paths, test_size=0.2, random_state=42)

    # Create Datasets and DataLoaders
    batch_size = args['batch_size']
    train_dataset = EEGDataset(train_files, args['filter_low'], args['filter_high'])
    test_dataset = EEGDataset(test_files, args['filter_low'], args['filter_high'])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(8, cpu_count()),
        pin_memory=True,
        collate_fn=custom_collate
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(8, cpu_count()),
        pin_memory=True,
        collate_fn=custom_collate
    )

    print(f"🧮 Batch size: {batch_size}")
    print_memory_usage()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")

    # Model parameters
    n_channels = train_dataset[0].shape[0]
    hidden_size = args['hidden_size']
    percentile = args.get('percentile', 95)
    initial_epsilon = args['epsilon']
    alpha = 0.025
    complexity = args['complexity']

    # Initialize model
    model = ConvLSTMEEGAutoencoder(n_channels=n_channels, hidden_size=hidden_size, 
                                   initial_epsilon=initial_epsilon, alpha=alpha,
                                   complexity=complexity).to(device)

    # Disable thresholding
    model.use_threshold = args['use_threshold']
    print(f"🧮 Use threshold: {model.use_threshold}")

    # Check for existing model and ask user for fine-tuning
    model_path = '/workspace/MatrixAutoEncoder/models/model.pth'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model, is_fine_tuning = load_or_initialize_model(model, model_path)

    # Print number of trainable parameters
    num_params = count_trainable_parameters(model)
    print(f"🧮 Number of trainable parameters: {num_params:,}")
    if 'WANDB_DISABLED' not in os.environ:
        wandb.log({"num_trainable_parameters": num_params})

    # Define loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Initialize the GradScaler
    scaler = GradScaler()

    # Training loop
    num_epochs = args['num_epochs']

    print(f"🏋️ Starting training for {num_epochs} epochs...")
    for epoch in tqdm(range(num_epochs), desc="Training progress"):
        model.train()
        train_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            
            # Compute new epsilon for this batch
            new_epsilon = compute_batch_epsilon(batch, percentile)
            
            # Update model's epsilon (smoothed update)
            model.update_epsilon(new_epsilon)
            
            # Forward pass with autocasting
            with autocast():
                reconstructed, recurrence_matrix = model(batch)
                loss = criterion(reconstructed, batch)
            
            # Backward pass with scaled gradients
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            # Log to wandb every 128 steps
            if batch_idx % 128 == 0:
                if 'WANDB_DISABLED' not in os.environ:
                    try:
                        wandb.log({
                            "train_loss": loss.item(),
                            "epoch": epoch,
                            "batch": batch_idx,
                            "epsilon": model.epsilon.item()
                        })

                        # Create and log visualization
                        try:
                            fig = plot_sample_and_reconstruction(batch[0], reconstructed[0], recurrence_matrix[0], sfreq=250)
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
    torch.save(model.state_dict(), model_path)
    print(f"💾 Model saved to '{model_path}'")

    print("\n🎉 Training complete! 🎉")

if __name__ == '__main__':
    main()