"""
N.Y. Hendrikx, K. Barhmi, L.R. Visser, T.A. de Bruin, M. Pó, A.A. Salah, W.G.J.H.M. van Sark
All sky imaging-based short-term solar irradiance forecasting with Long Short-Term Memory networks.
https://doi.org/10.1016/j.solener.2024.112463
https://github.com/nielsyh/ASI_playground/blob/master/models/models_ts/lstm_model.py
"""

import torch
import torch.nn as nn


class LSTMPredictor(nn.Module):
    """
    PyTorch version of the Keras LSTM model from:
    https://github.com/nielsyh/ASI_playground/blob/master/models/models_ts/lstm_model.py

    Keras Architecture:
    1. LSTM(50, activation='relu', input_shape=(seq_len, features), return_sequences=True)
    2. LSTM(25, activation='relu')
    3. Dense(10, activation='relu')
    4. Dense(1)
    """

    def __init__(self, input_features, hidden_size1=50, hidden_size2=25, dense_hidden=10):
        super().__init__()

        # --- LSTM Layers ---
        self.lstm1 = nn.LSTM(input_size=input_features, hidden_size=hidden_size1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hidden_size1, hidden_size=hidden_size2, batch_first=True)

        # --- Dense (Fully-Connected) Layers ---
        self.fc1 = nn.Linear(in_features=hidden_size2, out_features=dense_hidden)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(in_features=dense_hidden, out_features=1)

    def forward(self, x):
        # x input shape: (batch_size, sequence_length, input_features)
        # --- LSTM 1 (return_sequences=True) ---
        lstm_out, _ = self.lstm1(x)
        lstm_out = torch.relu(lstm_out)

        # --- LSTM 2 (return_sequences=False) ---
        _, (h_n, c_n) = self.lstm2(lstm_out)

        # h_n shape is (num_layers, batch_size, hidden_size2)
        last_hidden_state = h_n.squeeze(0)  # Shape: (batch_size, hidden_size2)
        last_hidden_state = torch.relu(last_hidden_state)

        # --- Dense Layers ---
        x = self.fc1(last_hidden_state)
        x = self.relu(x)
        return self.fc_out(x)
