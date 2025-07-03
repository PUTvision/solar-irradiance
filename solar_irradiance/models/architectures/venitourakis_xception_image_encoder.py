"""
Venitourakis, G., Vasilakis, C., Tsagkaropoulos, A., Amrou, T., Konstantoulakis, G., Golemis, P., & Reisis, D. (2023).
Neural Network-Based Solar Irradiance Forecast for Edge Computing Devices. Information, 14(11), 617.
https://doi.org/10.3390/info14110617
"""

import torch
import torch.nn as nn


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()

        self.depthwise_separable_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=padding, groups=in_channels),
            nn.LeakyReLU(negative_slope=0.1125, inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1125, inplace=True),
        )

    def forward(self, x):
        return self.depthwise_separable_conv(x)


class XceptionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise_separable_conv_k3 = SeparableConv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.depthwise_separable_conv_k5 = SeparableConv2d(in_channels, in_channels, kernel_size=5, padding=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(in_channels * 4, out_channels, kernel_size=1), nn.LeakyReLU(negative_slope=0.1125, inplace=True)
        )

    def forward(self, x):
        y_dsc_k3 = self.depthwise_separable_conv_k3(x)
        y_dsc_k5 = self.depthwise_separable_conv_k5(x)
        y_maxpool = self.maxpool(x)
        y = torch.cat([x, y_dsc_k3, y_dsc_k5, y_maxpool], dim=1)
        return self.pointwise_conv(y)


class XceptionImageEncoder(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.entry_flow = nn.Sequential(
            XceptionLayer(in_channels=in_channels, out_channels=64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            XceptionLayer(in_channels=64, out_channels=128),
        )

        self.skip_connection_flow_1 = nn.Sequential(
            XceptionLayer(in_channels=128, out_channels=128),
            XceptionLayer(in_channels=128, out_channels=128),
        )

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.skip_connection_flow_2 = nn.Sequential(
            XceptionLayer(in_channels=128, out_channels=128),
            XceptionLayer(in_channels=128, out_channels=256),
        )
        self.pointwise_conv = nn.Conv2d(128, 256, kernel_size=1)

        self.exit_flow = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            XceptionLayer(in_channels=256, out_channels=128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            XceptionLayer(in_channels=128, out_channels=128),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )

    def forward(self, x):
        entry_flow_y = self.entry_flow(x)

        skip_connection_1_y = self.skip_connection_flow_1(entry_flow_y)
        skip_connection_1_y = torch.add(entry_flow_y, skip_connection_1_y)

        maxpool_y = self.maxpool(skip_connection_1_y)

        skip_connection_2_y = self.skip_connection_flow_2(maxpool_y)
        pointwise_conv_y = self.pointwise_conv(maxpool_y)
        middle_flow_y = torch.add(pointwise_conv_y, skip_connection_2_y)

        exit_flow_y = self.exit_flow(middle_flow_y)
        y = exit_flow_y.view(x.size(0), -1)
        return y
