from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from albumentations import Compose
from torch.utils.data import Dataset

from solar_irradiance.datamodules.sun_mask import SunMask


MAX_IRRADIANCE = 1466.0 # max irradiance in the dataset
# MAX_IRRADIANCE = 1600.0 # max irradiance from Hukseflux pyranometer


class FolsomDataset(Dataset):
    latitude = 38.642,
    longitude = -121.148
    camera_orientation_compensation = 165
    focal_length = 0.48

    def __init__(
            self,
            data_root: Path,
            images_list: List[Path],
            augmentations: Compose,
            add_sun_mask: bool,
        ):
        self._data_root = data_root
        self._images_list = images_list
        self._augmentations = augmentations
        self._df = pd.read_csv(self._data_root / 'irradiance.csv', dtype={'date': str, 'irradiance': float}, index_col='date')
        self._add_sun_mask = add_sun_mask
        self._sun_mask = SunMask(self.latitude, self.longitude, self.camera_orientation_compensation, self.focal_length)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self._images_list[index]
        image, irradiance = self._load_data(image_path)

        transformed = self._augmentations(image=image)
        image = transformed['image']

        if self._add_sun_mask:
            image = self._sun_mask(image=image, timestamp=image_path.name[:15])

        irradiance /= MAX_IRRADIANCE
        irradiance = irradiance if irradiance >= 0.0 else 0.0

        return torch.from_numpy(image.transpose(2, 0, 1)), torch.Tensor([irradiance])

    def _load_data(self, image_path: Path) -> Tuple[np.ndarray, float]:
        frame = np.asarray(Image.open(image_path))
        row_name = ''.join([image_path.name[:12], str(round(float(image_path.name[12:15])/100)), '00'])
        irradiance = self._df.loc[row_name]['irradiance']

        return frame, irradiance

    def __len__(self) -> int:
        return len(self._images_list)
