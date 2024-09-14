# evaluation_tests/signal_analysis.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.preprocessing import compute_current_source_density
from scipy.signal import hilbert
import pathlib
from utils import chunk_data, extract_phase

def select_group():
    groups = ['AD', 'HID', 'MCI']
    print("Available groups:")
    for i, group in enumerate(groups, 1):
        print(f"{i}. {group}")
    while True:
        choice = input("Select a group (1-3): ")
        if choice.isdigit() and 1 <= int(choice) <= 3:
            return groups[int(choice) - 1]
        print("Invalid choice. Please try again.")

def load_and_preprocess_data(group, apply_laplacian=True):
    data_dir = pathlib.Path('data') / group
    subjects = list(data_dir.glob('*'))
    print(f"Number of subjects in {group}: {len(subjects)}")
    
    subject = np.random.choice(subjects)
    files = list(subject.glob('**/*_good_*_eeg.fif'))
    file = np.random.choice(files)
    
    raw = mne.io.read_raw_fif(file, preload=True, verbose=False)
    if apply_laplacian:
        raw = compute_current_source_density(raw)
    
    return raw

def analyze_signal(raw, channel_name):
    data = raw.get_data(picks=[channel_name])
    sfreq = raw.info['sfreq']
    times = np.arange(data.shape[1]) / sfreq
    
    phase = np.angle(hilbert(data[0]))
    
    print(f"Signal duration: {data.shape[1] / sfreq:.2f} seconds")
    print(f"Phase range: [{phase.min():.2f}, {phase.max():.2f}]")
    
    plt.figure(figsize=(12, 6))
    plt.plot(times, data[0])
    plt.title(f"Raw signal - {channel_name}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.plot(times, phase)
    plt.title(f"Phase - {channel_name}")
    plt.xlabel("Time (s)")
    plt.ylabel("Phase")
    plt.show()

def main():
    group = select_group()
    apply_laplacian = input("Apply Laplacian? (y/n): ").lower() == 'y'
    
    raw = load_and_preprocess_data(group, apply_laplacian)
    
    print("\nAvailable channels:")
    for i, ch in enumerate(raw.ch_names, 1):
        print(f"{i}. {ch}")
    
    while True:
        choice = input("Select a channel number: ")
        if choice.isdigit() and 1 <= int(choice) <= len(raw.ch_names):
            channel = raw.ch_names[int(choice) - 1]
            break
        print("Invalid choice. Please try again.")
    
    analyze_signal(raw, channel)

if __name__ == "__main__":
    main()