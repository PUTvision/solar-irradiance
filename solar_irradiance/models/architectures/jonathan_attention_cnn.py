"""
Anto Leoba Jonathan, Dongsheng Cai, Chiagoziem C. Ukwuoma, Nkou Joseph Junior Nkou, Qi Huang, Olusola Bamisile
A radiant shift: Attention-embedded CNNs for accurate solar irradiance forecasting and prediction from sky images
https://doi.org/10.1016/j.renene.2024.121133
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel-Wise Attention component."""

    def __init__(self, in_channels: int, reduction_ratio: int = 16) -> None:
        """
        Initializes the ChannelAttention module.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        reduction_ratio : int, optional
            Factor by which to reduce the channels in the bottleneck (hidden) layer.
            Article doesn't specify, so applied default one, 16.
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Calculate the number of channels in the hidden layer
        hidden_channels = max(1, in_channels // reduction_ratio)

        self.fc = nn.Sequential(
            # Squeeze: 1x1 Conv
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            # Excite: 1x1 Conv
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the attention mechanism.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map. Shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Attended feature map. Shape (B, C, H, W).
        """
        b, c, _, _ = x.size()

        # Squeeze: Global Average Pooling
        y = self.avg_pool(x)

        # Excitation: Pass through FC layers (implemented as 1x1 Convs)
        y = self.fc(y)

        # Scale the original input 'x' by the attention weights 'y'
        return x * y.expand_as(x)


class AttentionCNN(nn.Module):
    """
    Attention-embedded Convolutional Neural Network (AttentionCNN).

    This model consists of 5 CNN blocks followed by a Channel-Wise Attention layer and a final regressor head.
    """

    def __init__(self, in_channels: int, num_classes: int = 3) -> None:
        """
        Initializes the AttentionCNN model.

        Parameters
        ----------
        in_channels : int
            Number of input channels. As per the paper, this is (sequence_length * 3) for RGB images. E.g., for a sequence of 4 images, in_channels = 12.
        num_classes : int, optional
            Number of output regression targets. As per the paper, this is 3 (GHI, DNI, DHI). Default is 3.
        """
        super().__init__()

        # --- CNN Component ---
        # Block 1: in_channels -> 32
        self.block1 = self._make_block(in_channels, 32)

        # Block 2: 32 -> 64
        self.block2 = self._make_block(32, 64)

        # Block 3: 64 -> 128
        self.block3 = self._make_block(64, 128)

        # Block 4: 128 -> 256
        self.block4 = self._make_block(128, 256)

        # Block 5: 256 -> 512
        self.block5 = self._make_block(256, 512)

        # --- Attention Mechanism ---
        self.attention = ChannelAttention(in_channels=512)

        # --- Regressor Head ---

        # Calculate the flattened feature size after 5 max-pooling layers.
        # Assuming 128x128 input image:
        # 128 -> 64 (pool1) -> 32 (pool2) -> 16 (pool3) -> 8 (pool4) -> 4 (pool5)
        # So, the feature map size is 4x4.
        # flat_features = channels * height * width
        flat_features = 512 * 4 * 4

        # We infer the hidden size of the dense layer (e.g., 1024 or 4096)
        # as it's not specified in the diagram. Let's use 1024.
        hidden_dim = 1024

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(flat_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, num_classes),  # 3 outputs (GHI, DNI, DHI)
        )

    def _make_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """
        Helper function to create one CNN block.
        (Conv -> ReLU -> Conv -> ReLU -> MaxPool)

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.

        Returns
        -------
        nn.Sequential
            A sequential container with the CNN block layers.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the AttentionCNN model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor. Shape (B, sequence_length * 3, 128, 128).

        Returns
        -------
        torch.Tensor
            Model output. Shape (B, 3).
        """
        # Pass through the 5 CNN blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        # Apply channel-wise attention
        x = self.attention(x)

        # Pass through the regressor head
        x = self.regressor(x)

        return x


# --- Example Usage ---
if __name__ == "__main__":
    # --- Model Configuration based on the paper ---
    # Paper uses sequence lengths of 1, 2, 4, or 8. Let's use 1.
    sequence_length = 1

    # Each image is RGB (3 channels)
    image_channels = 3

    # Total input channels = sequence_length * image_channels
    # This assumes images are stacked on the channel dimension.
    input_channels = sequence_length * image_channels

    # Output is 3 values (GHI, DNI, DHI)
    num_outputs = 1

    # Image size is 128x128
    img_size = 128

    # Batch size (e.g., 2)
    batch_size = 2

    # 1. Instantiate the model
    print("Initializing AttentionCNN model with:")
    print(f"  Sequence Length: {sequence_length}")
    print(f"  Input Channels: {input_channels}")
    print(f"  Output Classes: {num_outputs}\n")

    model = AttentionCNN(in_channels=input_channels, num_classes=num_outputs)

    # 2. Create a dummy input tensor
    # Shape: (batch_size, input_channels, height, width)
    dummy_input = torch.randn(batch_size, input_channels, img_size, img_size)

    print(f"Created a dummy input tensor of shape: {dummy_input.shape}")

    # 3. Perform a forward pass
    try:
        output = model(dummy_input)
        print("Successfully performed a forward pass.")
        print(f"Output tensor shape: {output.shape}")

        # 4. Verify the output shape
        assert output.shape == (batch_size, num_outputs)
        print("Output shape is correct (Batch Size, Num Classes).")

        print("\n--- Model Architecture ---")
        print(model)

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTotal trainable parameters: {total_params:,}")

    except Exception as e:
        print("\nAn error occurred during the forward pass:")
        print(e)
        print("Please check the model architecture and input dimensions.")
