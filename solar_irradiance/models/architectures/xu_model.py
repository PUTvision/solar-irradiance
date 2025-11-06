"""
Shijie Xu, Ruiyuan Zhang, Hui Ma, Chandima Ekanayake, Yi Cui
On vision transformer for ultra-short-term forecasting of photovoltaic generation using sky images
https://doi.org/10.1016/j.solener.2023.112203
"""

import torch
import torch.nn as nn


class SpatialProcessingModule(nn.Module):
    """
    Implements the Spatial Processing Module from Fig. 1 & 3.
    This module processes a SINGLE sky image using a ViT.

    Based on parameters from Table 1:
    - Input Image: (B, 3, H, W) - flexible input size
    - ViT:
        - Patch Size: 16x16
        - Embed Dim: 1024
        - Patches: (H/16) * (W/16)
        - Heads: 4
        - Transformer Layers: 4
        - MLP Dim (ViT): 2048
    - Output Layer: 256 (This is the 'MLP Head' in Fig. 2)
    """

    def __init__(self, embed_dim=1024, patch_size=16, num_layers=4, num_heads=4, mlp_dim=2048, output_dim=256, dropout=0.1):
        super().__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # This is the ViT patch embedding layer (Fig 2)
        self.patch_embed = nn.Conv2d(in_channels=3, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)

        # CLS token (Fig 2)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Position Embedding will be created dynamically based on input size
        self.pos_embed = None

        # Transformer Encoder (Fig 3)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=mlp_dim, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # MLP Head to get final spatial features (Fig 2, 3)
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp_head = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        # x shape: (B, 3, H, W) - e.g., (B, 3, 128, 128) or (B, 3, 258, 258)
        B, C, H, W = x.shape

        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        num_patches = x.shape[1]

        # Initialize or update position embeddings if needed
        if self.pos_embed is None or self.pos_embed.shape[1] != num_patches + 1:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim, device=x.device))

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches + 1, embed_dim)

        # Add position embedding
        x = x + self.pos_embed

        # Pass through Transformer Encoder
        x = self.transformer_encoder(x)  # (B, num_patches + 1, embed_dim)

        # Get the CLS token output
        cls_token_out = x[:, 0]  # (B, embed_dim)

        # Pass through MLP Head
        cls_token_out = self.norm(cls_token_out)
        spatial_features = self.mlp_head(cls_token_out)  # (B, output_dim)

        return spatial_features


class SkyImageStream(nn.Module):
    """
    Implements the Sky Image stream from Fig. 1 & 7.
    MODIFIED: This version does NOT take exogenous data.

    Based on parameters from Table 1:
    - GRU Encoder 1: 256 hidden state, 4 layers
    - FC after GRU1: (4*256) -> 256
    - Final FC Layer: 256 -> 1 (This replaces the fc_after_concat layer)
    """

    def __init__(self):
        super().__init__()

        self.spatial_processor = SpatialProcessingModule()

        # GRU Encoder 1
        self.temporal_gru = nn.GRU(
            input_size=256,  # From SpatialProcessingModule output
            hidden_size=256,
            num_layers=4,
            batch_first=True,
        )

        # FC Layer after GRU1
        self.fc_after_gru = nn.Linear(4 * 256, 256)

        # Final FC layer (replaces the fc_after_concat)
        self.final_fc = nn.Linear(256, 1)

        self.relu = nn.ReLU()

    def forward(self, x_images):
        # x_images shape: (B, T, C, H, W), e.g., (8, 15, 3, 258, 258)

        B, T, C, H, W = x_images.shape

        # Process each image in the sequence spatially
        # (B, T, C, H, W) -> (B*T, C, H, W)
        x_images_flat = x_images.view(B * T, C, H, W)
        spatial_features = self.spatial_processor(x_images_flat)  # (B*T, 256)

        # Reshape back to sequence: (B*T, 256) -> (B, T, 256)
        spatial_sequence = spatial_features.view(B, T, -1)

        # Pass sequence through GRU Encoder 1
        # We use the final hidden state h_n
        _, h_n = self.temporal_gru(spatial_sequence)  # h_n shape: (4, B, 256)

        # Flatten the hidden states from all layers
        # (4, B, 256) -> (B, 4, 256) -> (B, 4 * 256)
        h_n_flat = h_n.permute(1, 0, 2).flatten(1)

        # Pass through FC layer
        sky_feat = self.relu(self.fc_after_gru(h_n_flat))  # (B, 256)

        # Pass through the final FC layer
        output = self.final_fc(sky_feat)  # (B, 1)

        return output


class IrradianceHistoryStream(nn.Module):
    """
    Implements the Historical Irradiance stream (formerly PV Generation)
    from Fig. 1 & 7.

    This uses the parameters for 'GRU Encoder 2':
    - GRU Encoder 2: 2 hidden state, 4 layers
    - FC after GRU2: (4*2) -> 1
    """

    def __init__(self):
        super().__init__()

        # GRU Encoder 2
        self.temporal_gru = nn.GRU(
            input_size=1,  # Historical irradiance value
            hidden_size=2,
            num_layers=4,
            batch_first=True,
        )

        # FC Layer after GRU2
        self.fc = nn.Linear(4 * 2, 1)

    def forward(self, x_irradiance):
        # x_irradiance shape: (B, T, 1), e.g., (8, 15, 1)

        # Pass sequence through GRU Encoder 2
        _, h_n = self.temporal_gru(x_irradiance)  # h_n shape: (4, B, 2)

        # Flatten the hidden states
        # (4, B, 2) -> (B, 4, 2) -> (B, 4 * 2)
        h_n_flat = h_n.permute(1, 0, 2).flatten(1)

        # Final FC layer for this stream
        output = self.fc(h_n_flat)  # (B, 1)

        return output


class ViTGRUForecaster(nn.Module):
    """
    The complete forecasting framework from Fig. 1.
    MODIFIED: This version does NOT take exogenous data.

    The 'x_irradiance_history' input is optional.
    """

    def __init__(self):
        super().__init__()
        self.sky_stream = SkyImageStream()
        self.irradiance_stream = IrradianceHistoryStream()

    def forward(self, x_images, x_irradiance_history=None):
        # Get the prediction from the sky image stream
        sky_prediction = self.sky_stream(x_images)

        if x_irradiance_history is not None:
            # Get the prediction from the historical irradiance stream
            irradiance_prediction_stream = self.irradiance_stream(x_irradiance_history)

            # Combine via element-wise addition
            final_prediction = sky_prediction + irradiance_prediction_stream
        else:
            # Use only the sky stream prediction
            final_prediction = sky_prediction

        # The final output is the forecasted solar irradiance
        return final_prediction
