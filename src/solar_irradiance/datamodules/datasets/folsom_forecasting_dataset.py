from pathlib import Path
from typing import Tuple, List, Dict, Any

import numpy as np
import torch
from PIL import Image
from albumentations import ReplayCompose
from torch.utils.data import Dataset

from solar_irradiance.datamodules.sun_mask import SunMask

MAX_IRRADIANCE = 1366.0  # max irradiance in the dataset
IRRADIANCE_MEAN = 419.1655  # mean irradiance in the dataset
IRRADIANCE_STD = 301.2625  # std irradiance in the dataset

# MAX_IRRADIANCE = 1600.0 # max irradiance from Hukseflux pyranometer


class FolsomForecastingDataset(Dataset):
    latitude = 38.642,
    longitude = -121.148
    camera_orientation_compensation = 165
    focal_length = 0.48

    def __init__(
            self,
            data_root: Path,
            periods: List[Dict[str, Any]],
            transforms: ReplayCompose,
            sun_mask: bool,
            blur_mask: bool,
    ):
        self._data_root = data_root
        self._periods = periods
        self._transforms = transforms
        self._sun_mask_enabled = sun_mask
        self._sun_mask = SunMask(self.latitude, self.longitude, self.camera_orientation_compensation, self.focal_length, blur_mask=blur_mask)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        period = self._periods[index]

        source_images = []
        source_irradiances = []
        replay_data = None
        for history_item in period['history']:
            image_path = self._data_root / 'images' / history_item['image_name']
            irradiance = history_item['irradiance'] / MAX_IRRADIANCE
            image = np.asarray(Image.open(image_path))

            if replay_data is None:
                transformed = self._transforms(image=image)
                replay_data = transformed['replay']
            else:
                transformed = self._transforms.replay(replay_data, image=image)

            image = transformed['image']
            if self._sun_mask_enabled:
                image = self._sun_mask(image=image, timestamp=image_path.name[:15])

            source_images.append(torch.from_numpy(image).permute(2, 0, 1))
            source_irradiances.append(irradiance)

        target_irradiance = period['target_irradiance'] / MAX_IRRADIANCE

        return (torch.stack(source_images).permute(1, 0, 2, 3),
                torch.Tensor(source_irradiances),
                torch.Tensor(target_irradiance))

    def __len__(self) -> int:
        return len(self._periods)
