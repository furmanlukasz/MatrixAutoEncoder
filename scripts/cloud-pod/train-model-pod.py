# scripts/train_model.py

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
from mne.preprocessing import compute_current_source_density
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
from utils import chunk_data, extract_phase, ConvLSTMEEGAutoencoder

class EEGDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = np.load(self.file_paths[idx])
        if self.transform:
            data = self.transform(data)
        # Scale the phase data from [-π, π] to [-1, 1]
        data = data / np.pi  # Normalizing phase to [-1, 1]
        return torch.tensor(data, dtype=torch.float32)

# Disable MNE info messages
mne.set_log_level('ERROR')

def parse_args():
    args = {
        'n_subjects_per_group': 100,
        'batch_size': 64,
        'chunk_duration': 5.0,
        'hidden_size': 64,
        'num_epochs': 6000,
        'epsilon': np.pi / 16,
        'complexity': 2,
        'checkpoint_frequency': 50,
        'filter_low': 3.0,
        'filter_high': 7.0,
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

def preprocess_subject(args, metadata_list):
    subject, group_name, output_dir, chunk_duration, filter_low, filter_high = args
    sfreq = None
    file_paths = []
    files = [f for f in subject.glob('**/*_good_*_eeg.fif') if not f.name.startswith('._')]
    print(f"Subject '{subject.name}' has {len(files)} files.")
    if len(files) == 0:
        print(f"⚠️ Warning: No data files found for subject '{subject.name}'.")
    for file in files:
        try:
            raw = mne.io.read_raw_fif(file, preload=True, verbose=False)
            raw = raw.filter(l_freq=filter_low, h_freq=filter_high, n_jobs=1)
            raw = compute_current_source_density(raw)
            data = raw.get_data()
            sfreq = raw.info['sfreq']
            chunks = chunk_data(data, sfreq, chunk_duration=chunk_duration)
            for i, chunk in enumerate(chunks):
                phase_chunk = extract_phase(chunk)
                file_name = f"{group_name}_{subject.name}_{file.stem}_{i}.npy"
                file_path = os.path.join(output_dir, file_name)
                np.save(file_path, phase_chunk)
                file_paths.append(file_path)
                
                # Collect metadata
                metadata_list.append({
                    'chunk_file': file_path,
                    'original_fif_file': str(file),
                    'group': group_name,
                    'subject': subject.name,
                    'file_stem': file.stem,
                    'chunk_index': i
                })
        except Exception as e:
            print(f"⚠️ Warning: Failed to process file {file}: {e}")
    return file_paths, sfreq

def preprocess_and_save_data(group_dirs, n_subjects_per_group, output_dir, chunk_duration=5.0, filter_low=3.0, filter_high=40.0, metadata_list=None):
    os.makedirs(output_dir, exist_ok=True)
    all_file_paths = []
    sfreq = None

    if metadata_list is None:
        metadata_list = []
    
    args_list = []
    for group_name, group_dir in group_dirs.items():
        filt_dir = group_dir / 'FILT'
        if not filt_dir.exists():
            print(f"⚠️ Warning: FILT directory does not exist for group '{group_name}' at path '{filt_dir}'.")
            continue
        subject_folders = [f for f in filt_dir.glob('*') if f.is_dir()]
        print(f"Group '{group_name}' has {len(subject_folders)} subjects.")
        if len(subject_folders) == 0:
            print(f"⚠️ Warning: No subject folders found in {filt_dir}.")
        subject_folders = subject_folders[:n_subjects_per_group]
        for subject in subject_folders:
            args_list.append((subject, group_name, output_dir, chunk_duration, filter_low, filter_high))
    
    total_subjects = len(args_list)
    if total_subjects == 0:
        print("❌ No subjects found to process. Please check your data directories and glob patterns.")
        sys.exit(1)
    
    print(f"Total subjects to process: {total_subjects}")
    
    # Process subjects sequentially
    for args in tqdm(args_list, desc="Processing subjects"):
        file_paths, sfreq = preprocess_subject(args, metadata_list)
        all_file_paths.extend(file_paths)
    
    return all_file_paths, sfreq

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
    
    # Create a more informative checkpoint filename
    checkpoint_name = (f"checkpoint_epoch_{epoch}_"
                       f"loss_{loss:.4f}_"
                       f"lr_{optimizer.param_groups[0]['lr']:.6f}_"
                       f"fl{args['filter_low']}_fh{args['filter_high']}_"
                       f"hs{args['hidden_size']}_"
                       f"eps{args['epsilon']:.4f}_"
                       f"{time.strftime('%Y%m%d_%H%M%S')}.pth")
    
    checkpoint_path = os.path.join(os.path.dirname(model_path), checkpoint_name)
    torch.save(checkpoint, checkpoint_path)
    print(f"💾 Checkpoint saved to '{checkpoint_path}'")
    
    # Log checkpoint information to wandb
    # if 'WANDB_DISABLED' not in os.environ:
    try:
        # Log checkpoint details as a text file
        checkpoint_info = (f"Epoch: {epoch}\n"
                            f"Loss: {loss:.4f}\n"
                            f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}\n"
                            f"Filter Low: {args['filter_low']}\n"
                            f"Filter High: {args['filter_high']}\n"
                            f"Hidden Size: {args['hidden_size']}\n"
                            f"Epsilon: {args['epsilon']:.4f}\n"
                            f"Checkpoint Path: {checkpoint_path}")
        
        wandb.log({"checkpoint_info": wandb.Html(checkpoint_info)})
        
        # Optionally, save the checkpoint as an artifact
        artifact = wandb.Artifact(f"model-checkpoint-epoch-{epoch}", type="model")
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact)
        
        print(f"✅ Checkpoint information logged to wandb")
    except Exception as e:
        print(f"⚠️ Warning: Failed to log checkpoint information to wandb: {e}")


def main():
    print("\n🚀 Welcome to the EEG Model Trainer! 👩‍💻👨‍💻\n")

    # Load environment variables
    load_dotenv()

    # Initialize wandb
    wandb_api_key = "a32849d94af9c711d39d425741579db87e1f192c"
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
    data_dir = pathlib.Path('/workspace/MatrixAutoEncoder/data')
    print(f"📂 Data directory: {data_dir}")
    group_dirs = {
        'AD': data_dir / 'AD',
        'HID': data_dir / 'HID',
        'MCI': data_dir / 'MCI'
    }

    output_dir = '/workspace/MatrixAutoEncoder/preprocessed_data'

    # Check if preprocessed data exists
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        print("🗂️ Preprocessed data already exists. Skipping preprocessing.")
        # Collect all file paths
        file_paths = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.npy')]
        sfreq = 250  # Set your sampling frequency accordingly
    else:
        # Preprocess data and get file paths
        import pandas as pd

        metadata_list = []
        
        print(f"📊 Preprocessing and saving data... - setting fmin: {args['filter_low']} - fmax: {args['filter_high']}")
        file_paths, sfreq = preprocess_and_save_data(
            group_dirs=group_dirs,
            n_subjects_per_group=args['n_subjects_per_group'],
            output_dir=output_dir,
            chunk_duration=args['chunk_duration'],
            filter_low=args['filter_low'],
            filter_high=args['filter_high'],
            metadata_list=metadata_list  # Pass the metadata list to the function
        )
        
        # After preprocessing, create a DataFrame from metadata_list and save it
        metadata_df = pd.DataFrame(metadata_list)
        metadata_csv_path = os.path.join(output_dir, 'chunk_metadata.csv')
        metadata_df.to_csv(metadata_csv_path, index=False)
        print(f"💾 Metadata CSV saved to '{metadata_csv_path}'")

    # Split file paths into training and testing
    train_files, test_files = train_test_split(file_paths, test_size=0.2, random_state=42)

    # Create Datasets and DataLoaders
    batch_size = args['batch_size']
    train_dataset = EEGDataset(train_files)
    test_dataset = EEGDataset(test_files)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(8, cpu_count()),  # Adjust based on your CPU cores
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(8, cpu_count()),
        pin_memory=True
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
    torch.save(model.state_dict(), model_path)
    print(f"💾 Model saved to '{model_path}'")

    print("\n🎉 Training complete! 🎉")

if __name__ == '__main__':
    main()
