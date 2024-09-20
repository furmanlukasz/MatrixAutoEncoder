# Place this at the very top of your script to limit the number of threads
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


import mne
import numpy as np
import os
import pathlib
from tqdm import tqdm
from mne.preprocessing import compute_current_source_density
from multiprocessing import cpu_count

# Disable MNE info messages
mne.set_log_level('ERROR')

def chunk_data(data, sfreq, chunk_duration=5.0):
    chunk_size = int(sfreq * chunk_duration)
    return [data[:, i:i+chunk_size] for i in range(0, data.shape[1], chunk_size)]

def preprocess_subject(args):
    subject, group_name, output_dir, chunk_duration = args
    sfreq = None
    file_paths = []
    files = [f for f in subject.glob('**/*_good_*_eeg.fif') if not f.name.startswith('._')]
    print(f"Subject '{subject.name}' has {len(files)} files.")
    if len(files) == 0:
        print(f"⚠️ Warning: No data files found for subject '{subject.name}'.")
    for file in files:
        try:
            raw = mne.io.read_raw_fif(file, preload=True, verbose=False)
            raw = compute_current_source_density(raw)
            data = raw.get_data()
            sfreq = raw.info['sfreq']
            chunks = chunk_data(data, sfreq, chunk_duration=chunk_duration)
            for i, chunk in enumerate(chunks):
                file_name = f"{group_name}_{subject.name}_{file.stem}_{i}.npy"
                file_path = os.path.join(output_dir, file_name)
                np.save(file_path, chunk)
                file_paths.append(file_path)
        except Exception as e:
            print(f"⚠️ Warning: Failed to process file {file}: {e}")
    return file_paths, sfreq

def preprocess_and_save_data(group_dirs, n_subjects_per_group, output_dir, chunk_duration=5.0):
    os.makedirs(output_dir, exist_ok=True)
    all_file_paths = []
    sfreq = None

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
            args_list.append((subject, group_name, output_dir, chunk_duration))

    total_subjects = len(args_list)
    if total_subjects == 0:
        print("❌ No subjects found to process. Please check your data directories and glob patterns.")
        return [], None

    print(f"Total subjects to process: {total_subjects}")

    # Process subjects in parallel
    for args in tqdm(args_list, total=total_subjects, desc="Processing subjects"):
        file_paths, subject_sfreq = preprocess_subject(args)
        all_file_paths.extend(file_paths)
        if sfreq is None:
            sfreq = subject_sfreq

    return all_file_paths, sfreq

def main():
    print("\n🚀 Welcome to the EEG Data Preprocessor! 👩‍💻👨‍💻\n")

    # Set the paths
    data_dir = pathlib.Path('/workspace/MatrixAutoEncoder/data')
    print(f"📂 Data directory: {data_dir}")
    group_dirs = {
        'AD': data_dir / 'AD',
        'HID': data_dir / 'HID',
        'MCI': data_dir / 'MCI'
    }

    output_dir = '/workspace/MatrixAutoEncoder/preprocessed_data'
    n_subjects_per_group = 100  # You can adjust this or make it a command-line argument
    chunk_duration = 5.0  # You can adjust this or make it a command-line argument

    # Preprocess and save data
    print(f"📊 Preprocessing and saving data...")
    file_paths, sfreq = preprocess_and_save_data(
        group_dirs=group_dirs,
        n_subjects_per_group=n_subjects_per_group,
        output_dir=output_dir,
        chunk_duration=chunk_duration
    )

    print(f"\n✅ Preprocessing complete!")
    print(f"📁 Preprocessed data saved to: {output_dir}")
    print(f"📊 Total files processed: {len(file_paths)}")
    print(f"⏱️ Sampling frequency: {sfreq} Hz")

if __name__ == '__main__':
    main()