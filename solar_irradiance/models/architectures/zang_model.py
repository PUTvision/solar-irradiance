"""
Haixiang Zang, Dianhao Chen, Jingxuan Liu, Lilin Cheng, Guoqiang Sun, Zhinong Wei
Improving ultra-short-term photovoltaic power forecasting using a novel sky-image-based framework considering spatial-temporal feature interaction
https://doi.org/10.1016/j.energy.2024.130538
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. Helper Modules (Building Blocks) ---


class ConvLSTMCell(nn.Module):
    """
    A single ConvLSTM cell. PyTorch does not have this built-in.
    Based on the paper's description and standard implementations.
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        # Convolution for all gates at once (input-to-hidden)
        self.conv_ii = nn.Conv2d(
            in_channels=self.input_dim,
            out_channels=4 * self.hidden_dim,  # i, f, o, g
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )
        # Convolution for all gates at once (hidden-to-hidden)
        self.conv_hh = nn.Conv2d(
            in_channels=self.hidden_dim,
            out_channels=4 * self.hidden_dim,  # i, f, o, g
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        # Compute all gates
        gates = self.conv_ii(input_tensor) + self.conv_hh(h_cur)

        # Split gates
        i, f, o, g = torch.split(gates, self.hidden_dim, dim=1)

        # Apply activations (Eq. 2)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = (f * c_cur) + (i * g)
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv_ii.weight.device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv_ii.weight.device),
        )


class CausalConv1d(nn.Module):
    """
    A 1D causal convolution layer.
    Pads the input on the left so that the output at time `t` only
    depends on inputs up to time `t`.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super().__init__()
        # Calculate left padding
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        # Remove padding from the right
        return x[:, :, : -self.padding]


class TemporalBlock(nn.Module):
    """
    Residual TCN block as shown in Fig. 7.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()

        # Create CausalConv1d layers first
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation=dilation)

        # Apply WeightNorm to the inner conv layers using the new API
        self.conv1.conv = torch.nn.utils.parametrizations.weight_norm(self.conv1.conv)
        self.conv2.conv = torch.nn.utils.parametrizations.weight_norm(self.conv2.conv)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        self.net = nn.Sequential(self.conv1, self.relu1, self.conv2, self.relu2)

        # 1x1 conv for residual connection if channels don't match
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        res = x if self.residual is None else self.residual(x)
        out = self.net(x)
        return self.relu(out + res)


# --- 2. Core Model Components ---


class SpatialStreamCNN(nn.Module):
    """
    Implements the Spatial Stream CNN (VGG-like). (Section 2.1.1, Fig. 3)
    Processes a sequence of images [B, T, C, H, W] and returns
    a sequence of feature maps [B, T, C_out, H_out, W_out].
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # VGG-like blocks: [Conv -> ReLU -> Conv -> ReLU -> MaxPool]
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x64 -> 32x32
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16
        )
        # 1x1 conv to project to the final `out_channels`
        self.project = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        b, t, c, h, w = x.shape
        # Reshape to (B*T, C, H, W) to process all images in one batch
        x_flat = x.view(b * t, c, h, w)

        out = self.block1(x_flat)
        out = self.block2(out)
        out = self.project(out)

        # Reshape back to (B, T, C_out, H_out, W_out)
        _, c_out, h_out, w_out = out.shape
        out = out.view(b, t, c_out, h_out, w_out)
        return out


class TemporalStreamConvLSTM(nn.Module):
    """
    Implements the Temporal Stream using stacked ConvLSTM. (Section 2.1.2)
    Processes a sequence of optical flow maps [B, T, 2, H, W]
    and returns a sequence of feature maps [B, T, C_out, H_out, W_out].
    (Fig. 2 shows 3 layers)
    """

    def __init__(self, in_channels, out_channels, hidden_dims, kernel_size):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims)

        # Add downsampling to match spatial stream (2 maxpools = 4x reduction)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

        cells = []
        for i in range(self.num_layers):
            cur_input_dim = in_channels if i == 0 else hidden_dims[i - 1]
            cells.append(ConvLSTMCell(input_dim=cur_input_dim, hidden_dim=hidden_dims[i], kernel_size=kernel_size))
        self.cells = nn.ModuleList(cells)

        # 1x1 conv to project to the final `out_channels`
        self.project = nn.Conv2d(hidden_dims[-1], out_channels, kernel_size=1)

    def forward(self, x):
        b, t, c, h, w = x.shape

        # Downsample optical flow to match spatial stream output (128->64->32)
        x_flat = x.view(b * t, c, h, w)
        x_down = self.downsample(x_flat)  # 128 -> 64
        x_down = self.downsample(x_down)  # 64 -> 32
        _, _, h_down, w_down = x_down.shape
        x = x_down.view(b, t, c, h_down, w_down)

        h, w = h_down, w_down

        # Initialize hidden states
        hidden_states = []
        for cell in self.cells:
            hidden_states.append(cell.init_hidden(b, (h, w)))

        layer_outputs = []  # To store the output sequence of the last layer

        for t_step in range(t):
            x_t = x[:, t_step, :, :, :]

            for layer_idx in range(self.num_layers):
                h_state, c_state = hidden_states[layer_idx]
                h_state, c_state = self.cells[layer_idx](x_t, (h_state, c_state))
                hidden_states[layer_idx] = (h_state, c_state)
                x_t = h_state  # Output of this layer is input to the next

            layer_outputs.append(h_state)

        # Stack outputs along the time dimension
        out = torch.stack(layer_outputs, dim=1)  # (B, T, C_hidden, H, W)

        # Project the output
        # Need to reshape for 1x1 Conv
        _, _, c_hidden, h_out, w_out = out.shape
        out_flat = out.view(b * t, self.hidden_dims[-1], h_out, w_out)
        out_proj = self.project(out_flat)
        _, c_out, h_final, w_final = out_proj.shape
        out = out_proj.view(b, t, c_out, h_final, w_final)

        return out


class GateUnit(nn.Module):
    """
    Implements the Gate Unit for feature fusion. (Section 2.1.3, Fig. 5, Eq. 3)
    Uses the *original image* to generate weights W_s and W_t.
    """

    def __init__(self, in_channels, img_h, img_w):
        super().__init__()

        # As per Fig 5, "Dense" is used on the image. This is likely
        # 1x1 Convs to compute scores for each feature map.
        # We compute one score for spatial (e_s) and one for temporal (e_t).
        self.conv_s = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.conv_t = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, spatial_features, temporal_features, original_image):
        b, t, c_img, h_img, w_img = original_image.shape

        # Reshape for 2D conv
        img_flat = original_image.view(b * t, c_img, h_img, w_img)

        # Get scores e_s, e_t (Eq. 3)
        e_s = self.conv_s(img_flat)  # (B*T, 1, H, W)
        e_t = self.conv_t(img_flat)  # (B*T, 1, H, W)

        # Get feature map dimensions
        _, _, c_feat, h_feat, w_feat = spatial_features.shape

        # Resize scores to match feature map dimensions
        e_s = F.interpolate(e_s, size=(h_feat, w_feat), mode="bilinear")
        e_t = F.interpolate(e_t, size=(h_feat, w_feat), mode="bilinear")

        # Stack scores for softmax: (B*T, 2, H_feat, W_feat)
        e = torch.cat([e_s, e_t], dim=1)

        # Softmax over the "channel" (dim 1) to get weights
        weights = F.softmax(e, dim=1)  # (B*T, 2, H_feat, W_feat)

        # Split weights W_s, W_t
        w_s, w_t = torch.chunk(weights, 2, dim=1)  # (B*T, 1, ...), (B*T, 1, ...)

        # Reshape features for fusion
        spatial_flat = spatial_features.view(b * t, c_feat, h_feat, w_feat)
        temporal_flat = temporal_features.view(b * t, c_feat, h_feat, w_feat)

        # Apply weights: y = W_s * m + W_t * h (Eq. 3)
        fused = (w_s * spatial_flat) + (w_t * temporal_flat)

        # Reshape back to (B, T, C_feat, H_feat, W_feat)
        out = fused.view(b, t, c_feat, h_feat, w_feat)
        return out


class IrradianceGuidedAttention(nn.Module):
    """
    Implements the PV-guided (Irradiance-guided) Attention. (Section 2.2)
    This is interpreted as an attention mechanism where Q (query) comes
    from irradiance and K (key)/V (value) come from the fused feature map.
    This condenses the spatial-temporal features into a 1D time series.
    """

    def __init__(self, feature_channels, irradiance_dim, attn_dim, val_dim):
        super().__init__()
        self.attn_dim = attn_dim

        # W_q: Projects irradiance (Q)
        self.W_q = nn.Linear(irradiance_dim, attn_dim)

        # W_k, W_v: 1x1 Convs for feature map (K, V)
        self.W_k = nn.Conv2d(feature_channels, attn_dim, kernel_size=1)
        self.W_v = nn.Conv2d(feature_channels, val_dim, kernel_size=1)

    def forward(self, fused_features, irradiance):
        # fused_features: (B, T, C, H, W)
        # irradiance: (B, T, 1)
        b, t, c, h, w = fused_features.shape

        # 1. Get Q
        Q = self.W_q(irradiance)  # (B, T, attn_dim)
        Q = Q.unsqueeze(-2)  # (B, T, 1, attn_dim)

        # Reshape features for 1x1 conv
        feat_flat = fused_features.view(b * t, c, h, w)

        # 2. Get K
        K = self.W_k(feat_flat)  # (B*T, attn_dim, H, W)
        K = K.view(b, t, self.attn_dim, h * w)  # (B, T, attn_dim, H*W)

        # 3. Get V
        V = self.W_v(feat_flat)  # (B*T, val_dim, H, W)
        V = V.view(b, t, -1, h * w)  # (B, T, val_dim, H*W)

        # 4. Attention scores (Eq. 5)
        # (B, T, 1, attn_dim) @ (B, T, attn_dim, H*W) -> (B, T, 1, H*W)
        attn_scores = torch.matmul(Q, K) / (self.attn_dim**0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, T, 1, H*W)

        # 5. Apply weights to V
        # (B, T, 1, H*W) @ (B, T, H*W, val_dim) -> (B, T, 1, val_dim)
        # Note the transpose on V
        out = torch.matmul(attn_weights, V.transpose(-1, -2))

        # Squeeze to get the final "Processed Feature"
        out = out.squeeze(-2)  # (B, T, val_dim)
        return out


class FusionModule(nn.Module):
    """
    Implements the fusion block from Fig. 9.
    Fuses the TCN hidden state (from irradiance) with the image features.
    """

    def __init__(self, tcn_channels, image_channels, out_channels):
        super().__init__()

        # Dense layer for image feature (Fig. 9)
        self.dense_img = nn.Linear(image_channels, tcn_channels)
        self.cat_dim = tcn_channels * 2  # tcn_state + dense_img

        # Dense layer to compute sigmoid weights
        self.gate_dense = nn.Sequential(nn.Linear(self.cat_dim, tcn_channels), nn.Sigmoid())

        # Final projection
        self.project = nn.Linear(self.cat_dim, out_channels)

    def forward(self, tcn_state, image_feature):
        # tcn_state: (B, T, C_tcn)
        # image_feature: (B, T, C_img)

        # Project image feature (Fig. 9 "Dense")
        img_proj = self.dense_img(image_feature)  # (B, T, C_tcn)

        # Concatenate (Fig. 9 "Concatenate" before sigmoid)
        cat_feat = torch.cat([tcn_state, img_proj], dim=-1)  # (B, T, C_tcn*2)

        # Get weights (Fig. 9 "Sigmoid")
        w_power = self.gate_dense(cat_feat)  # (B, T, C_tcn)
        w_image = 1.0 - w_power

        # Apply weights and concatenate (Fig. 9 "Concatenate" at end)
        fused = torch.cat([w_power * tcn_state, w_image * img_proj], dim=-1)

        # Project to out_channels for the next TCN block
        out = self.project(fused)
        return out


class ProgressiveTCN(nn.Module):
    """
    Implements the Time Series Inference Module. (Section 2.3.2, Fig. 8a)
    Fuses image features *between* TCN layers.
    """

    def __init__(self, irradiance_dim, image_dim, tcn_channels, kernel_size, forecast_horizon):
        super().__init__()
        self.tcn_channels = tcn_channels
        self.image_dim = image_dim

        # Initial projection for irradiance
        self.irradiance_proj = nn.Conv1d(irradiance_dim, tcn_channels[0], 1)

        self.tcn_blocks = nn.ModuleList()
        self.fusion_blocks = nn.ModuleList()

        num_layers = len(tcn_channels)

        for i in range(num_layers):
            dilation = 2**i
            in_c = tcn_channels[i - 1] if i > 0 else tcn_channels[0]
            out_c = tcn_channels[i]

            self.tcn_blocks.append(TemporalBlock(in_c, out_c, kernel_size, dilation))

            self.fusion_blocks.append(FusionModule(out_c, image_dim, out_c))

        # Final layers to produce the forecast
        self.final_conv = nn.Conv1d(tcn_channels[-1], forecast_horizon, 1)

    def forward(self, irradiance, image_features):
        # irradiance: (B, T, 1)
        # image_features: (B, T, C_img)

        # Reshape irradiance for Conv1d: (B, 1, T)
        x = irradiance.transpose(1, 2)
        x = self.irradiance_proj(x)

        for tcn, fusion in zip(self.tcn_blocks, self.fusion_blocks, strict=True):
            # 1. Pass through TCN block
            x_hidden = tcn(x)  # (B, C_tcn, T)

            # 2. Transpose for fusion
            x_hidden_t = x_hidden.transpose(1, 2)  # (B, T, C_tcn)

            # 3. Fuse with image features (Fig. 8a)
            x_fused = fusion(x_hidden_t, image_features)  # (B, T, C_tcn)

            # 4. Transpose back for next TCN block
            x = x_fused.transpose(1, 2)  # (B, C_tcn, T)

        # Final output projection
        out = self.final_conv(x)  # (B, forecast_horizon, T)

        # Take the last time step's output
        out = out[:, :, -1]  # (B, forecast_horizon)
        return out


# --- 3. Main Model Class ---


class ZangModel(nn.Module):
    """
    The complete, combined model. (Fig. 1)
    """

    def __init__(
        self,
        img_c,
        img_h,
        img_w,
        seq_len,
        forecast_horizon,
        fused_channels=64,
        tcn_channels=(32, 32, 32),
        attn_dim=64,
        val_dim=32,
    ):
        super().__init__()

        self.img_h = img_h
        self.img_w = img_w

        # 1. Image Feature Extraction Module
        self.spatial_stream = SpatialStreamCNN(in_channels=img_c, out_channels=fused_channels)
        # Calculate spatial stream output size
        self.h_feat, self.w_feat = self._get_feat_size(img_h, img_w)

        self.temporal_stream = TemporalStreamConvLSTM(
            in_channels=2,  # Optical flow (u, v)
            out_channels=fused_channels,
            hidden_dims=[32, 64, fused_channels],
            kernel_size=(3, 3),
        )

        # 2. Gate Unit
        self.gate_unit = GateUnit(in_channels=img_c, img_h=img_h, img_w=img_w)

        # 3. Irradiance-Guided Attention
        self.attention = IrradianceGuidedAttention(
            feature_channels=fused_channels,
            irradiance_dim=1,
            attn_dim=attn_dim,
            val_dim=val_dim,  # This is the image_dim for the TCN
        )

        # 4. Time Series Inference Module
        self.progressive_tcn = ProgressiveTCN(
            irradiance_dim=1, image_dim=val_dim, tcn_channels=tcn_channels, kernel_size=3, forecast_horizon=forecast_horizon
        )

    def _get_feat_size(self, h, w):
        # Calculate output size of spatial stream
        # 2x MaxPool with stride 2
        return h // 4, w // 4

    def forward(self, image_sequence, optical_flow, irradiance_history):
        """
        Full forward pass of the model.

        Args:
            image_sequence (torch.Tensor): (B, T, C, H, W)
            optical_flow (torch.Tensor): (B, T, 2, H, W)
            irradiance_history (torch.Tensor): (B, T)
        """
        # Ensure irradiance has channel dim
        irradiance_history = irradiance_history.unsqueeze(-1)  # (B, T, 1)

        # 1. Feature Extraction
        spatial_features = self.spatial_stream(image_sequence)
        temporal_features = self.temporal_stream(optical_flow)

        # 2. Gate Unit Fusion
        fused_features = self.gate_unit(spatial_features, temporal_features, image_sequence)

        # 3. Attention
        processed_features = self.attention(fused_features, irradiance_history)

        # 4. Progressive TCN
        forecast = self.progressive_tcn(irradiance_history, processed_features)

        return forecast
