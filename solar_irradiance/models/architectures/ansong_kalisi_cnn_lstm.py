"""
PyTorch implementation of the CNN-LSTM model for the Karlsruhe low-cost all-sky imager (KALiSI).
https://github.com/KALiSI4SIFS/KALiSI/blob/main/SIFS_forecast_cnn_lstm_15min.ipynb

Martin Ansong, Gan Huang, Thomas N. Nyangonda, Robinson J. Musembi, Bryce S. Richards
Very short-term solar irradiance forecasting based on open-source low-cost sky imager and hybrid deep-learning techniques.
https://doi.org/10.1016/j.solener.2025.113516
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KALiSI(nn.Module):
    def __init__(
        self,
        image_input_dim,
        numeric_input_dim,
        num_filters=24,
        kernel_size=(3, 3),
        pool_size=(2, 2),
        strides=2,
        lstm_units=32,
        dense_size=1024,
        drop_rate=0.4,
    ) -> None:
        super().__init__()

        # Store input dims for helper function
        self._image_input_dim = image_input_dim

        # --- CNN Branch (for image input) ---
        self.conv1 = nn.Conv2d(
            in_channels=image_input_dim[0], out_channels=num_filters, kernel_size=kernel_size, padding="same"
        )
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.pool1 = nn.MaxPool2d(kernel_size=pool_size, stride=strides)

        self.conv2 = nn.Conv2d(in_channels=num_filters, out_channels=num_filters * 2, kernel_size=kernel_size, padding="same")
        self.bn2 = nn.BatchNorm2d(num_filters * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=pool_size, stride=strides)

        # --- Adaptive Pooling Layer ---
        # Guarantees the output spatial dimension is (16, 16)
        # This size is chosen to match the stable 64x64 input's output
        self.adaptive_pool = nn.AdaptiveMaxPool2d((16, 16))

        # --- Calculate flattened CNN output size ---
        # This will now be constant: 16 * 16 * (num_filters * 2)
        self.cnn_output_size = self._get_conv_output_dim()

        # --- Combined Branch ---
        self.combined_features_dim = self.cnn_output_size + numeric_input_dim

        # --- LSTM Layer ---
        self.lstm = nn.LSTM(input_size=self.combined_features_dim, hidden_size=lstm_units, batch_first=True)

        # --- Fully Connected (Dense) Layers ---
        self.fc1 = nn.Linear(lstm_units, dense_size)
        self.dropout1 = nn.Dropout(drop_rate)
        self.fc2 = nn.Linear(dense_size, dense_size)
        self.dropout2 = nn.Dropout(drop_rate)

        # --- Output Layer ---
        self.fc_out = nn.Linear(dense_size, 1)

    def _get_conv_output_dim(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *self._image_input_dim)
            x = self.pool1(F.relu(self.bn1(self.conv1(dummy_input))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))

            # Apply adaptive pooling in the dummy pass as well
            x = self.adaptive_pool(x)

            flat_size = x.flatten(start_dim=1).shape[1]
        return flat_size

    def forward(self, x_image, x_numeric):
        # --- CNN Branch ---
        x = F.relu(self.bn1(self.conv1(x_image)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        # --- Apply Adaptive Pooling ---
        # This ensures the spatial dimension is fixed, e.g., (Batch, 48, 16, 16)
        x = self.adaptive_pool(x)

        # --- Flatten ---
        # Shape: (Batch, 16 * 16 * 48) = (Batch, 12288)
        x_cnn_flat = torch.flatten(x, start_dim=1)

        # --- Concatenate ---
        # Shape: (Batch, 12288 + numeric_input_dim)
        x_combined = torch.cat((x_cnn_flat, x_numeric), dim=1)

        # --- Prepare for LSTM ---
        # Add sequence dimension (Seq_Len=1)
        # Shape: (Batch, 1, 12288 + numeric_input_dim)
        x_seq = x_combined.unsqueeze(1)

        # --- LSTM ---
        lstm_out, (h_n, c_n) = self.lstm(x_seq)

        # Get the hidden state from the last layer
        x_lstm = h_n[-1]  # Shape: (Batch, lstm_units)

        # --- Fully Connected Layers ---
        x = F.relu(self.fc1(x_lstm))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        # --- Output Layer ---
        y_out = self.fc_out(x)

        return y_out
