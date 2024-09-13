# scripts/utils.py

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn

def chunk_data(data, sfreq, chunk_duration=5.0):
    n_samples = data.shape[1]
    n_chunk_samples = int(chunk_duration * sfreq)
    n_chunks = int(np.ceil(n_samples / n_chunk_samples))
    chunks = []
    for i in range(n_chunks):
        start = i * n_chunk_samples
        end = start + n_chunk_samples
        chunk = data[:, start:end]
        # Pad if the chunk is shorter than n_chunk_samples
        if chunk.shape[1] < n_chunk_samples:
            pad_width = n_chunk_samples - chunk.shape[1]
            chunk = np.pad(chunk, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
        chunks.append(chunk)
    return chunks

def extract_phase(data):
    from scipy.signal import hilbert
    analytic_signal = hilbert(data, axis=-1)
    instantaneous_phase = np.angle(analytic_signal)
    return instantaneous_phase

class EEGDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)
        # Create a mask to handle padding (if necessary)
        self.masks = (self.data != 0).float()  # Mask is 1 where data is not zero

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        mask = self.masks[idx]
        return x, mask

# Model classes
class ConvLSTMEEGEncoder(nn.Module):
    def __init__(self, n_channels, hidden_size):
        super(ConvLSTMEEGEncoder, self).__init__()
        self.conv1 = nn.Conv1d(n_channels, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size, batch_first=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        return lstm_out

class ConvLSTMEEGDecoder(nn.Module):
    def __init__(self, n_channels, hidden_size):
        super(ConvLSTMEEGDecoder, self).__init__()
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=128, batch_first=True)
        self.deconv1 = nn.ConvTranspose1d(128, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.deconv2 = nn.ConvTranspose1d(64, n_channels, kernel_size=3, stride=1, padding=1)
        self.output_activation = nn.Tanh()

    def forward(self, lstm_out):
        lstm_dec_out, _ = self.lstm(lstm_out)
        x = lstm_dec_out.permute(0, 2, 1)
        x = self.deconv1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        reconstructed = self.output_activation(x)
        return reconstructed

class ConvLSTMEEGAutoencoder(nn.Module):
    def __init__(self, n_channels, hidden_size):
        super(ConvLSTMEEGAutoencoder, self).__init__()
        self.encoder = ConvLSTMEEGEncoder(n_channels, hidden_size)
        self.decoder = ConvLSTMEEGDecoder(n_channels, hidden_size)

    def forward(self, x, epsilon):
        lstm_out = self.encoder(x)
        recurrence_matrix = self.compute_recurrence_matrix(lstm_out, epsilon)
        reconstructed = self.decoder(lstm_out)
        return reconstructed, recurrence_matrix

    def compute_recurrence_matrix(self, lstm_out, epsilon):
        batch_size, seq_len, hidden_size = lstm_out.size()
        recurrence_matrices = []
        for i in range(batch_size):
            sample_lstm_out = lstm_out[i]  # Shape: (seq_len, hidden_size)
            # Normalize the vectors
            norm = sample_lstm_out.norm(dim=1, keepdim=True)
            normalized_vectors = sample_lstm_out / norm
            # Compute cosine similarity matrix
            cosine_similarity = torch.mm(normalized_vectors, normalized_vectors.t())
            # Clamp values to the valid range of arccos to avoid numerical errors
            cosine_similarity = cosine_similarity.clamp(-1 + 1e-7, 1 - 1e-7)
            # Compute angular distance matrix
            angular_distance = torch.acos(cosine_similarity)
            # Apply threshold to create binary recurrence matrix
            recurrence_matrix = (angular_distance <= epsilon).float()
            recurrence_matrices.append(recurrence_matrix)
        recurrence_matrices = torch.stack(recurrence_matrices)
        return recurrence_matrices