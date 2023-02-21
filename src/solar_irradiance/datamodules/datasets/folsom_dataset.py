from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from albumentations import Compose
from torch.utils.data import Dataset


MAX_IRRADIANCE = 1466.0 # max irradiance in the dataset
# MAX_IRRADIANCE = 1600.0 # max irradiance from Hukseflux pyranometer


class FolsomDataset(Dataset):
    def __init__(self,
                 data_root: Path,
                 images_list: List[Path],
                 augmentations: Compose
                 ):
        self._data_root = data_root
        self._images_list = images_list
        self._augmentations = augmentations
        self._df = pd.read_csv(self._data_root / 'folsom_global_irradiance.csv', dtype={'date': str, 'irradiance': float}, index_col='date')

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, irradiance = self._load_data(index)

        transformed = self._augmentations(image=image)
        image = transformed['image']

        irradiance /= MAX_IRRADIANCE
        irradiance = irradiance if irradiance >= 0.0 else 0.0

        return image, torch.Tensor([irradiance])

    def _load_data(self, index: int) -> Tuple[np.ndarray, float]:
        image_path = self._images_list[index]

        frame = np.asarray(Image.open(image_path))
        row_name = ''.join([image_path.name[:12], str(round(float(image_path.name[12:15])/100)), '00'])
        irradiance = self._df.loc[row_name]['irradiance']

        return frame, irradiance

    def __len__(self) -> int:
        return len(self._images_list)
