"""
Thomas M. Mercier, Amin Sabet, Tasmiat Rahman
Vision transformer models to measure solar irradiance using sky images in temperate climates
https://doi.org/10.1016/j.apenergy.2024.122967
https://github.com/Gittingthehubbing/Solar_Irradiance_ViT/blob/master/models/pretrained_timm.py
"""

import torch
import torch.nn as nn


class MercierViT(nn.Module):
    def __init__(
        self,
        inmodel,
        numeric_input_size: int,
        number_of_linear_layers: int,
        drop_out_lin: float,
        intermediate_linear_layer_shape: int,
        linear_activation_func: str,
        sigmoid_on: bool,
        y_shape: tuple[int, ...],
    ) -> None:
        super().__init__()

        self.model_name = "mercier_vit"
        self.number_of_linear_layers = number_of_linear_layers
        self.drop_out_lin = drop_out_lin
        self.intermediate_linear_layer_shape = intermediate_linear_layer_shape
        self.linear_activation_func = linear_activation_func
        self.sigmoid_on = sigmoid_on
        if "test_input_size" in inmodel.pretrained_cfg:
            with torch.no_grad():
                a = torch.rand(inmodel.pretrained_cfg["test_input_size"]).unsqueeze(0)
                out_shape = inmodel(a).shape
            self.lin_in_shape = out_shape[1]
        else:
            self.lin_in_shape = inmodel.norm.normalized_shape[0]

        if len(y_shape) > 1:
            self.out_shape = y_shape[1]
        else:
            self.out_shape = y_shape[0]

        self.vit_main = inmodel

        linear_layers = []
        layer_in_shape = self.lin_in_shape + numeric_input_size  # numeric inputs concatenated
        for lin_layer in range(1, number_of_linear_layers):
            linear_layers.append(nn.Dropout(drop_out_lin))
            linear_layers.append(nn.Linear(layer_in_shape, intermediate_linear_layer_shape))
            layer_in_shape = intermediate_linear_layer_shape
            if lin_layer < number_of_linear_layers and linear_activation_func != "Linear":
                linear_layers.append(getattr(nn, linear_activation_func)())
        linear_layers.append(nn.Dropout(drop_out_lin))
        linear_layers.append(nn.Linear(layer_in_shape, self.out_shape))

        if sigmoid_on:
            linear_layers.append(nn.Sigmoid())

        self.lin_model = nn.Sequential(*linear_layers)

    def forward(self, x, irradiance_history):
        # mostly from https://github.com/rwightman/pytorch-image-models/blob/7c67d6aca992f039eece0af5f7c29a43d48c00e4/timm/models/vision_transformer.py
        x = self.vit_main(x)

        lin_out = self.lin_model(torch.cat([x, irradiance_history], dim=1))
        return lin_out

    def remove_final_linear(self):
        self.lin_model = nn.Identity()
