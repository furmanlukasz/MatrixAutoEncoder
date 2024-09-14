import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.preprocessing import compute_current_source_density
import pathlib
from utils import chunk_data, extract_phase, ConvLSTMEEGAutoencoder
import torch
from tqdm import tqdm
import argparse
import subprocess
import pywt
from skimage.filters import gabor
from skimage import img_as_float

def select_group():
    groups = ['AD', 'HID', 'MCI']
    print("🏥 Available groups:")
    for i, group in enumerate(groups, 1):
        print(f"{i}. {group}")
    while True:
        choice = input("Select a group (1-3): ")
        if choice.isdigit() and 1 <= int(choice) <= 3:
            return groups[int(choice) - 1]
        print("❌ Invalid choice. Please try again.")

def select_subject(group):
    data_dir = pathlib.Path('data') / group
    subjects = list(data_dir.glob('*'))
    print(f"\n👥 Available subjects in {group}:")
    for i, subject in enumerate(subjects, 1):
        print(f"{i}. {subject.name}")
    while True:
        choice = input(f"Select a subject (1-{len(subjects)}): ")
        if choice.isdigit() and 1 <= int(choice) <= len(subjects):
            return subjects[int(choice) - 1]
        print("❌ Invalid choice. Please try again.")

def load_and_preprocess_data(subject_path):
    files = list(subject_path.glob('**/*_good_*_eeg.fif'))
    file = files[0]  # Take the first file for the subject
    
    raw = mne.io.read_raw_fif(file, preload=True, verbose=False)
    raw = compute_current_source_density(raw)
    
    return raw

def generate_angular_matrices(raw, model, window_duration, step_size, device):
    data = raw.get_data()
    sfreq = raw.info['sfreq']
    total_duration = data.shape[1] / sfreq
    print(f" 📊 Total Duration: {total_duration:.2f}s")
    
    window_samples = int(window_duration * sfreq)
    step_samples = int(step_size * sfreq)
    
    angular_matrices = []
    
    for start in tqdm(range(0, data.shape[1] - window_samples, step_samples), desc="🔄 Generating matrices"):
        end = start + window_samples
        window = data[:, start:end]
        
        phase_chunk = extract_phase(window)
        phase_chunk = torch.from_numpy(phase_chunk).unsqueeze(0).float().to(device)
        
        with torch.no_grad():
            _, recurrence_matrix = model(phase_chunk)
        
        angular_matrices.append(recurrence_matrix[0].cpu().numpy())
    
    return angular_matrices, total_duration

def dwt_analysis(angular_matrix):
    coeffs2 = pywt.dwt2(angular_matrix, 'haar')
    LL, (LH, HL, HH) = coeffs2
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(LL, cmap='gray')
    axes[0, 0].set_title('Approximation (LL)')
    axes[0, 1].imshow(LH, cmap='gray')
    axes[0, 1].set_title('Horizontal Details (LH)')
    axes[1, 0].imshow(HL, cmap='gray')
    axes[1, 0].set_title('Vertical Details (HL)')
    axes[1, 1].imshow(HH, cmap='gray')
    axes[1, 1].set_title('Diagonal Details (HH)')
    plt.tight_layout()

def cwt_analysis(angular_matrix):
    signal = np.mean(angular_matrix, axis=0)
    widths = np.arange(1, 31)
    cwt_matrix, freqs = pywt.cwt(signal, widths, 'gaus1')
    
    plt.imshow(cwt_matrix, extent=[0, len(signal), widths[-1], widths[0]], cmap='PRGn', aspect='auto')
    plt.title('Continuous Wavelet Transform')
    plt.xlabel('Time')
    plt.ylabel('Scale')
    plt.colorbar()

def texture_analysis(angular_matrix):
    # Convert to float
    image = img_as_float(angular_matrix)
    
    # Apply Gabor filter
    filt_real, filt_imag = gabor(image, frequency=0.6)
    
    # Compute the magnitude
    magnitude = np.sqrt(filt_real**2 + filt_imag**2)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(image, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2.imshow(magnitude, cmap='hot')
    ax2.set_title('Gabor Magnitude')
    ax2.axis('off')
    
    plt.tight_layout()

def create_movie(angular_matrices, output_file, fps, time_dilation, total_duration, analysis_method):
    plt.figure(figsize=(10, 8))
    
    for i, matrix in enumerate(tqdm(angular_matrices, desc="🎬 Creating movie frames")):
        plt.clf()
        
        if analysis_method == '1':
            plt.imshow(matrix, cmap='viridis', aspect='equal')
            plt.colorbar(label='Angular Distance')
            real_time = i / len(angular_matrices) * total_duration
            plt.title(f"Angular Distance Matrix\nTime: {real_time:.2f}s, Dilation: {time_dilation}x")
            plt.xlabel("Encoded Sequence Index")
            plt.ylabel("Encoded Sequence Index")
        elif analysis_method == '2':
            dwt_analysis(matrix)
        elif analysis_method == '3':
            cwt_analysis(matrix)
        elif analysis_method == '4':
            texture_analysis(matrix)
        
        plt.savefig(f'temp_frame_{i:04d}.png')
        plt.close()
    
    print("🎥 Encoding video...")
    effective_fps = len(angular_matrices) / total_duration * time_dilation
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-framerate', str(effective_fps), 
        '-i', 'temp_frame_%04d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-vf', f'setpts={1/time_dilation}*PTS',
        output_file
    ]
    subprocess.run(ffmpeg_cmd)
    
    # Clean up temporary files
    for i in range(len(angular_matrices)):
        os.remove(f'temp_frame_{i:04d}.png')

def main():
    print("\n🚀 Welcome to the Angular Matrix Movie Generator! 🎬\n")

    parser = argparse.ArgumentParser(description="Generate Angular Matrix Movie")
    parser.add_argument("--window", type=float, default=1.0, help="Window duration in seconds")
    parser.add_argument("--step", type=float, default=0.1, help="Step size in seconds")
    parser.add_argument("--dilation", type=float, default=1.0, help="Time dilation factor")
    args = parser.parse_args()

    group = select_group()
    subject_path = select_subject(group)
    
    print(f"\n📊 Loading and preprocessing data for {subject_path.name}...")
    raw = load_and_preprocess_data(subject_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")
    
    n_channels = len(raw.ch_names)
    hidden_size = 64  # Should match the trained model's hidden size
    
    print("🧠 Loading trained model...")
    model = ConvLSTMEEGAutoencoder(n_channels=n_channels, hidden_size=hidden_size).to(device)
    model.load_state_dict(torch.load('models/model.pth', map_location=device))
    model.eval()
    
    print(f"🔍 Parameters: Window={args.window}s, Step={args.step}s, Dilation={args.dilation}x")
    print("🔄 Generating angular matrices...")
    angular_matrices, total_duration = generate_angular_matrices(raw, model, args.window, args.step, device)
    
    print("\n📊 Choose analysis method:")
    print("1. Original Angular Distance Matrix")
    print("2. Discrete Wavelet Transform (DWT)")
    print("3. Continuous Wavelet Transform (CWT)")
    print("4. Texture Analysis (Gabor Filter)")
    
    while True:
        choice = input("Enter your choice (1-4): ")
        if choice in ['1', '2', '3', '4']:
            break
        print("❌ Invalid choice. Please try again.")
    
    output_file = f'results/angular_matrix_movie_{group}_{subject_path.name}_method{choice}.mp4'
    print(f"🎬 Creating movie: {output_file}")
    create_movie(angular_matrices, output_file, len(angular_matrices)/total_duration, args.dilation, total_duration, choice)
    
    print("\n🎉 Movie creation complete! 🎉")

if __name__ == "__main__":
    main()