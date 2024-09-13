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
    def __init__(self, n_channels, hidden_size, complexity=0):
        super(ConvLSTMEEGEncoder, self).__init__()
        self.complexity = complexity
        
        if complexity == 0:
            self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=3, stride=1, padding=1)
            self.relu1 = nn.ReLU()
            self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_size, batch_first=True)
        elif complexity == 1:
            self.conv1 = nn.Conv1d(n_channels, 64, kernel_size=3, stride=1, padding=1)
            self.relu1 = nn.ReLU()
            self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
            self.relu2 = nn.ReLU()
            self.lstm1 = nn.LSTM(input_size=128, hidden_size=hidden_size, batch_first=True)
            self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        else:  # complexity == 2
            self.conv1 = nn.Conv1d(n_channels, 64, kernel_size=3, stride=1, padding=1)
            self.relu1 = nn.ReLU()
            self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
            self.relu2 = nn.ReLU()
            self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
            self.relu3 = nn.ReLU()
            self.lstm1 = nn.LSTM(input_size=256, hidden_size=hidden_size, batch_first=True)
            self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        if self.complexity == 0:
            x = self.conv1(x)
            x = self.relu1(x)
            x = x.permute(0, 2, 1)
            x, _ = self.lstm(x)
        elif self.complexity == 1:
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.conv2(x)
            x = self.relu2(x)
            x = x.permute(0, 2, 1)
            x, _ = self.lstm1(x)
            x, _ = self.lstm2(x)
        else:  # complexity == 2
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.conv2(x)
            x = self.relu2(x)
            x = self.conv3(x)
            x = self.relu3(x)
            x = x.permute(0, 2, 1)
            x, _ = self.lstm1(x)
            x, _ = self.lstm2(x)
            x = self.fc(x)
        return x

class ConvLSTMEEGDecoder(nn.Module):
    def __init__(self, n_channels, hidden_size, complexity=0):
        super(ConvLSTMEEGDecoder, self).__init__()
        self.complexity = complexity
        
        if complexity == 0:
            self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=32, batch_first=True)
            self.deconv = nn.ConvTranspose1d(32, n_channels, kernel_size=3, stride=1, padding=1)
        elif complexity == 1:
            self.lstm1 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
            self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=128, batch_first=True)
            self.deconv1 = nn.ConvTranspose1d(128, 64, kernel_size=3, stride=1, padding=1)
            self.relu1 = nn.ReLU()
            self.deconv2 = nn.ConvTranspose1d(64, n_channels, kernel_size=3, stride=1, padding=1)
        else:  # complexity == 2
            self.fc = nn.Linear(hidden_size, hidden_size)
            self.lstm1 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
            self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=256, batch_first=True)
            self.deconv1 = nn.ConvTranspose1d(256, 128, kernel_size=3, stride=1, padding=1)
            self.relu1 = nn.ReLU()
            self.deconv2 = nn.ConvTranspose1d(128, 64, kernel_size=3, stride=1, padding=1)
            self.relu2 = nn.ReLU()
            self.deconv3 = nn.ConvTranspose1d(64, n_channels, kernel_size=3, stride=1, padding=1)
        
        self.output_activation = nn.Tanh()

    def forward(self, lstm_out):
        if self.complexity == 0:
            x, _ = self.lstm(lstm_out)
            x = x.permute(0, 2, 1)
            x = self.deconv(x)
        elif self.complexity == 1:
            x, _ = self.lstm1(lstm_out)
            x, _ = self.lstm2(x)
            x = x.permute(0, 2, 1)
            x = self.deconv1(x)
            x = self.relu1(x)
            x = self.deconv2(x)
        else:  # complexity == 2
            x = self.fc(lstm_out)
            x, _ = self.lstm1(x)
            x, _ = self.lstm2(x)
            x = x.permute(0, 2, 1)
            x = self.deconv1(x)
            x = self.relu1(x)
            x = self.deconv2(x)
            x = self.relu2(x)
            x = self.deconv3(x)
        reconstructed = self.output_activation(x)
        return reconstructed

class ConvLSTMEEGAutoencoder(nn.Module):
    def __init__(self, n_channels, hidden_size, initial_epsilon=np.pi/4, alpha=0.1, complexity=0):
        super(ConvLSTMEEGAutoencoder, self).__init__()
        self.encoder = ConvLSTMEEGEncoder(n_channels, hidden_size, complexity)
        self.decoder = ConvLSTMEEGDecoder(n_channels, hidden_size, complexity)
        self.epsilon = nn.Parameter(torch.tensor(initial_epsilon), requires_grad=False)
        self.alpha = alpha
        self.use_threshold = True

    def forward(self, x):
        lstm_out = self.encoder(x)
        recurrence_matrix = self.compute_recurrence_matrix(lstm_out)
        reconstructed = self.decoder(lstm_out)
        return reconstructed, recurrence_matrix

    def compute_recurrence_matrix(self, lstm_out):
        batch_size, seq_len, hidden_size = lstm_out.size()
        recurrence_matrices = []
        for i in range(batch_size):
            sample_lstm_out = lstm_out[i]
            norm = sample_lstm_out.norm(dim=1, keepdim=True)
            normalized_vectors = sample_lstm_out / norm
            cosine_similarity = torch.mm(normalized_vectors, normalized_vectors.t())
            cosine_similarity = cosine_similarity.clamp(-1 + 1e-7, 1 - 1e-7)
            angular_distance = torch.acos(cosine_similarity)
            
            if self.use_threshold:
                recurrence_matrix = (angular_distance <= self.epsilon).float()
            else:
                recurrence_matrix = angular_distance
            
            recurrence_matrices.append(recurrence_matrix)
        recurrence_matrices = torch.stack(recurrence_matrices)
        return recurrence_matrices

    def update_epsilon(self, new_epsilon):
        with torch.no_grad():
            self.epsilon.data = self.alpha * new_epsilon + (1 - self.alpha) * self.epsilon.data