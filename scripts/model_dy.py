# scripts/utils.py

import json
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
        # scale the phase data from [-π, π] to [-1, 1]
        x = x / np.pi  # Normalizing phase to [-1, 1]
        mask = self.masks[idx]
        return x, mask


# Model classes
class DynamicConvLSTMEEGEncoder(nn.Module):
    def __init__(self, config, n_channels, hidden_size):
        super(DynamicConvLSTMEEGEncoder, self).__init__()
        self.config = config['encoder']
        self.hidden_size = hidden_size
        self.conv_layers = self._build_conv_layers(self.config['conv_layers'], n_channels)
        self.lstm_layers = self._build_lstm_layers(self.config['lstm_layers'])
        self.fc_layers = self._build_fc_layers(self.config.get('fc_layers', []))

    def _build_conv_layers(self, layer_configs, n_channels):
        layers = []
        in_channels = n_channels
        for layer_cfg in layer_configs:
            out_channels = layer_cfg['out_channels']
            conv = nn.Conv1d(
                in_channels=in_channels if layer_cfg['in_channels'] == 'n_channels' else layer_cfg['in_channels'],
                out_channels=out_channels,
                kernel_size=layer_cfg['kernel_size'],
                stride=layer_cfg['stride'],
                padding=layer_cfg['padding']
            )
            activation = getattr(nn, layer_cfg['activation'])()
            layers.extend([conv, activation])
            in_channels = out_channels
        return nn.Sequential(*layers)

    def _build_lstm_layers(self, layer_configs):
        layers = []
        for idx, layer_cfg in enumerate(layer_configs):
            lstm = nn.LSTM(
                input_size=layer_cfg['input_size'],
                hidden_size=layer_cfg['hidden_size'],
                num_layers=layer_cfg.get('num_layers', 1),
                batch_first=layer_cfg['batch_first']
            )
            layers.append(lstm)
        return nn.ModuleList(layers)

    def _build_fc_layers(self, layer_configs):
        layers = []
        for layer_cfg in layer_configs:
            fc = nn.Linear(
                in_features=layer_cfg['in_features'],
                out_features=layer_cfg['out_features']
            )
            if 'activation' in layer_cfg:
                activation = getattr(nn, layer_cfg['activation'])()
                layers.extend([fc, activation])
            else:
                layers.append(fc)
        return nn.Sequential(*layers) if layers else None

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.permute(0, 2, 1)  # Prepare for LSTM layers
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
        if self.fc_layers:
            x = self.fc_layers(x)
        return x

class DynamicConvLSTMEEGDecoder(nn.Module):
    def __init__(self, config, n_channels, hidden_size):
        super(DynamicConvLSTMEEGDecoder, self).__init__()
        self.config = config['decoder']
        self.hidden_size = hidden_size
        self.fc_layers = self._build_fc_layers(self.config.get('fc_layers', []))
        self.lstm_layers = self._build_lstm_layers(self.config['lstm_layers'])
        self.deconv_layers = self._build_deconv_layers(self.config['deconv_layers'], n_channels)

    def _build_fc_layers(self, layer_configs):
        layers = []
        for layer_cfg in layer_configs:
            fc = nn.Linear(
                in_features=layer_cfg['in_features'],
                out_features=layer_cfg['out_features']
            )
            if 'activation' in layer_cfg:
                activation = getattr(nn, layer_cfg['activation'])()
                layers.extend([fc, activation])
            else:
                layers.append(fc)
        return nn.Sequential(*layers) if layers else None

    def _build_lstm_layers(self, layer_configs):
        layers = []
        for layer_cfg in layer_configs:
            lstm = nn.LSTM(
                input_size=layer_cfg['input_size'],
                hidden_size=layer_cfg['hidden_size'],
                num_layers=layer_cfg.get('num_layers', 1),
                batch_first=layer_cfg['batch_first']
            )
            layers.append(lstm)
        return nn.ModuleList(layers)

    def _build_deconv_layers(self, layer_configs, n_channels):
        layers = []
        in_channels = layer_configs[0]['in_channels']
        for layer_cfg in layer_configs:
            out_channels = n_channels if layer_cfg['out_channels'] == 'n_channels' else layer_cfg['out_channels']
            deconv = nn.ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=layer_cfg['kernel_size'],
                stride=layer_cfg['stride'],
                padding=layer_cfg['padding']
            )
            activation = getattr(nn, layer_cfg['activation'])()
            layers.extend([deconv, activation])
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.fc_layers:
            x = self.fc_layers(x)
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
        x = x.permute(0, 2, 1)  # Prepare for deconvolution layers
        x = self.deconv_layers(x)
        return x

class DynamicConvLSTMEEGAutoencoder(nn.Module):
    def __init__(self, config, n_channels, hidden_size, initial_epsilon=np.pi/4, alpha=0.1):
        super(DynamicConvLSTMEEGAutoencoder, self).__init__()
        self.encoder = DynamicConvLSTMEEGEncoder(config['model_config'], n_channels, hidden_size)
        self.decoder = DynamicConvLSTMEEGDecoder(config['model_config'], n_channels, hidden_size)
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