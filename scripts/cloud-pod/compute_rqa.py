# scripts/compute_rqa.py

import os
import sys
import pathlib
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse
from multiprocessing import cpu_count
from RQA import get_results
import matplotlib.pyplot as plt

# Import utility functions and models
from utils import ConvLSTMEEGAutoencoder

def parse_args():
    parser = argparse.ArgumentParser(description="Compute RQA features from recurrence matrices")
    parser.add_argument('--input_csv', type=str, required=True,
                        help='Path to the input CSV file containing latent space data with ForFurtherAnalysis flag')
    parser.add_argument('--output_csv', type=str, required=True,
                        help='Path to save the output CSV file with RQA features')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for DataLoader')
    # Add argument for threshold percentile
    parser.add_argument('--threshold_percentile', type=float, default=85.0,
                        help='Percentile for thresholding the recurrence matrix (default: 85.0)')
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
            choice = int(input("\n🔢 Enter the number of the model you want to use: "))
            if 1 <= choice <= len(models):
                return models[choice - 1]
            else:
                print("❌ Invalid choice. Please try again.")
        except ValueError:
            print("❌ Please enter a valid number.")

class EEGDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = np.load(self.file_paths[idx])
        return torch.tensor(data, dtype=torch.float32)

def compute_rqa_features(recurrence_matrix, threshold_percentile=85.0):
    # Apply thresholding based on the specified percentile
    threshold = np.percentile(recurrence_matrix, threshold_percentile)
    recurrence_matrix_thresholded = (recurrence_matrix >= threshold).astype(int)

    # Compute RQA features using custom RQA implementation
    try:
        rqa_results = get_results(recurrence_matrix_thresholded,
                                  minimum_diagonal_line_length=2,
                                  minimum_vertical_line_length=2,
                                  minimum_white_vertical_line_length=2)
        
        features = {
            'RR': rqa_results[0],    # Recurrence Rate
            'DET': rqa_results[1],   # Determinism
            'L': rqa_results[2],     # Average diagonal line length
            'Lmax': rqa_results[3],  # Max diagonal line length
            'DIV': rqa_results[4],   # Divergence
            'ENTR': rqa_results[5],  # Entropy of diagonal lines
            'LAM': rqa_results[7],   # Laminarity
            'TT': rqa_results[15],   # Trapping Time
            'Vmax': rqa_results[9]   # Maximum vertical line length
        }
    except Exception as e:
        print(f"⚠️ Warning: Failed to compute RQA measures: {e}")
        features = {key: np.nan for key in ['RR', 'DET', 'L', 'Lmax', 'DIV', 'ENTR', 'LAM', 'TT', 'Vmax']}
    
    return features

def plot_recurrence_matrix(recurrence_matrix, subject_id, output_dir):
    plt.figure(figsize=(10, 10))
    plt.imshow(recurrence_matrix, cmap='binary', aspect='auto')
    plt.title(f'Recurrence Matrix for Subject {subject_id}')
    plt.colorbar(label='Recurrence')
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'rqa_matrix_subject_{subject_id}.png'))
    plt.close()

def main():
    print("\n🚀 Welcome to the RQA Feature Computation Script! 👩‍💻👨‍💻\n")

    args = parse_args()
    input_csv = args.input_csv
    output_csv = args.output_csv
    batch_size = args.batch_size
    threshold_percentile = args.threshold_percentile

    # Validate threshold_percentile
    if not (0 < threshold_percentile < 100):
        print("❌ 'threshold_percentile' must be between 0 and 100.")
        sys.exit(1)

    # Load the DataFrame
    try:
        print(f"📂 Loading data from '{input_csv}'...")
        df = pd.read_csv(input_csv)
        print(f"✅ Loaded data with {len(df)} samples.")
    except Exception as e:
        print(f"❌ Failed to load CSV file: {e}")
        sys.exit(1)

    # Check for required columns
    required_columns = ['ForFurtherAnalysis', 'chunk_file', 'subject']
    for col in required_columns:
        if col not in df.columns:
            print(f"❌ Required column '{col}' not found in the input CSV.")
            sys.exit(1)

    # Filter DataFrame for 'ForFurtherAnalysis' == True
    df_filtered = df[df['ForFurtherAnalysis'] == True].reset_index(drop=True)
    print(f"✅ {len(df_filtered)} samples marked for further analysis.")

    if len(df_filtered) == 0:
        print("❌ No samples marked for further analysis. Exiting.")
        sys.exit(1)

    # Load the model
    model_path = select_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")

    print("🧠 Loading trained model...")
    # Initialize the model without command-line parameters
    # We will extract them from the checkpoint
    # First, load the checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Extract model parameters from the checkpoint
    try:
        n_channels = 256
        hidden_size = 64
        complexity = 2
        # n_channels = checkpoint['n_channels']
        # hidden_size = checkpoint['hidden_size']
        # complexity = checkpoint['complexity']
        print(f"🔍 Retrieved model parameters from checkpoint:")
        print(f"    - n_channels: {n_channels}")
        print(f"    - hidden_size: {hidden_size}")
        print(f"    - complexity: {complexity}")
    except KeyError as e:
        print(f"❌ Missing parameter in checkpoint: {e}")
        print("🔧 Please ensure that 'n_channels', 'hidden_size', and 'complexity' are saved in the checkpoint.")
        sys.exit(1)

    # Initialize the model with extracted parameters
    model = ConvLSTMEEGAutoencoder(n_channels=n_channels, hidden_size=hidden_size, complexity=complexity).to(device)

    # Load the model state_dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded model state_dict from '{os.path.basename(model_path)}'")
    else:
        print("❌ 'model_state_dict' not found in the checkpoint.")
        sys.exit(1)

    model.eval()
    print("✅ Model is set to evaluation mode.")

    # Prepare Dataset and DataLoader
    file_paths = df_filtered['chunk_file'].tolist()
    dataset = EEGDataset(file_paths)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                             num_workers=min(8, cpu_count()), pin_memory=True)

    # Initialize a list to store RQA features
    rqa_features_list = []

    print("\n🔄 Computing recurrence matrices and RQA features...")
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Processing batches"):
            batch = batch.to(device, non_blocking=True)
            try:
                # Forward pass through model to get recurrence matrices
                _, recurrence_matrices = model(batch)
            except Exception as e:
                print(f"⚠️ Warning: Failed during model inference: {e}")
                # Append NaNs for all features if inference fails
                for _ in range(batch.size(0)):
                    rqa_features_list.append({
                        'RR': np.nan,
                        'DET': np.nan,
                        'L': np.nan,
                        'Lmax': np.nan,
                        'ENTR': np.nan,
                        'LAM': np.nan,
                        'TT': np.nan,
                        'Vmax': np.nan
                    })
                continue

            # Move to CPU and convert to numpy
            recurrence_matrices = recurrence_matrices.cpu().numpy()

            # Compute RQA features for each recurrence matrix
            for rec_mat in recurrence_matrices:
                features = compute_rqa_features(rec_mat, threshold_percentile=threshold_percentile)
                rqa_features_list.append(features)
                
                # Plot one matrix per subject
                subject_id = df_filtered.iloc[len(rqa_features_list) - 1]['subject']
                if len(rqa_features_list) % batch_size == 1:  # First sample of each batch
                    plot_recurrence_matrix(rec_mat, subject_id, 'results/rqa-mat/')

    # Create a DataFrame from RQA features
    rqa_df = pd.DataFrame(rqa_features_list)
    print(f"✅ Computed RQA features for {len(rqa_df)} samples.")

    # Concatenate RQA features with the filtered DataFrame
    df_result = pd.concat([df_filtered.reset_index(drop=True), rqa_df.reset_index(drop=True)], axis=1)

    # Save the updated DataFrame
    try:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    except FileNotFoundError:
        # If output_csv is in the current directory
        pass

    try:
        df_result.to_csv(output_csv, index=False)
        print(f"💾 Updated DataFrame with RQA features saved to '{output_csv}'")
    except Exception as e:
        print(f"❌ Failed to save the output CSV file: {e}")
        sys.exit(1)

    print("\n🎉 RQA feature computation complete! 🎉")

if __name__ == '__main__':
    main()