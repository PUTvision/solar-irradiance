from pathlib import Path
from typing import Any

from albumentations import ReplayCompose
import cv2
import numpy as np
import pandas as pd
import pytz
import torch
from torch.utils.data import Dataset

MAX_IRRADIANCE = 1466.0  # max irradiance in the dataset
# MAX_IRRADIANCE = 1600.0     # max irradiance from Hukseflux pyranometer


class FolsomForecastingDataset(Dataset):
    latitude = 38.642
    longitude = -121.148
    altitude = 68  # Altitude in meters
    camera_orientation_compensation = 165
    focal_length = 0.48

    us_pacific = pytz.timezone("US/Pacific")
    utc = pytz.utc

    def __init__(
        self,
        data_root: Path,
        periods: list[dict[str, Any]],
        transforms: ReplayCompose,
        add_sun_mask: bool,
        add_irradiance_channel: bool,
        optical_flow: None | str,
        cloud_mask_method: None | str,
        image_size: tuple[int],
        image_mean: tuple[float],
        image_std: tuple[float],
    ):
        self._data_root = data_root
        self._periods = periods
        self._image_size = image_size

        self._crop_mask = cv2.circle(
            np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8),
            (image_size[1] // 2, image_size[0] // 2),
            image_size[0] // 2,
            color=(1, 1, 1),
            thickness=-1,
        )
        # Load precomputed features DataFrame
        features_path = self._data_root / "hendrikx_features_normalized.csv"
        self._features_df = pd.read_csv(features_path, index_col="image_name")

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        period = self._periods[index]

        features_list = []

        for history_item in period["history"]:
            image_name = history_item["image_name"]

            row = self._features_df.loc[image_name]
            features_list.append(
                [
                    row["irradiance"],
                    row["brightness"],
                    row["cloud_pixels"],
                    row["edge_count"],
                    row["corner_count"],
                    row["csi"],
                    row["zenith"],
                    row["azimuth"],
                    row["apparent_elevation"],
                ]
            )

        target_irradiance = period["target_irradiance"]  # / MAX_IRRADIANCE

        return (torch.Tensor(features_list), torch.Tensor(features_list), torch.Tensor([target_irradiance]))

    def __len__(self) -> int:
        return len(self._periods)
