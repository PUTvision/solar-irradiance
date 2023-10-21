from pathlib import Path
from typing import Tuple, List, Dict, Any, Union

import numpy as np
import pandas as pd
import pytz
import torch
from PIL import Image
from albumentations import ReplayCompose
from torch.utils.data import Dataset

from solar_irradiance.datamodules.sun_mask import SunMask
from solar_irradiance.datamodules.cloud_mask import CloudMask

MAX_IRRADIANCE = 1466.0     # max irradiance in the dataset
# MAX_IRRADIANCE = 1600.0     # max irradiance from Hukseflux pyranometer


class FolsomForecastingDataset2D(Dataset):
    latitude = 38.642,
    longitude = -121.148
    camera_orientation_compensation = 165
    focal_length = 0.48

    us_pacific = pytz.timezone('US/Pacific')
    utc = pytz.utc

    def __init__(
            self,
            data_root: Path,
            periods: List[Dict[str, Any]],
            transforms: ReplayCompose,
            add_sun_mask: bool,
            add_irradiance_channel: bool,
            optical_flow: Union[None, str],
            cloud_mask_method: Union[None, str],
            image_size: Tuple[int, int],
    ):
        self._data_root = data_root
        self._periods = periods
        self._transforms = transforms
        self._add_sun_mask = add_sun_mask
        self._sun_mask = SunMask(self.latitude, self.longitude, self.camera_orientation_compensation, self.focal_length)
        self._add_irradiance_channel = add_irradiance_channel
        self._optical_flow = optical_flow
        self._cloud_mask_method = cloud_mask_method
        self._image_size = image_size
        if self._cloud_mask_method is not None:
            self._cloud_mask = CloudMask(shape=self._image_size, method=cloud_mask_method)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        period = self._periods[index]

        history_item = period['history'][-1]
        image_path = self._data_root / 'images' / history_item['image_name']
        irradiance = history_item['irradiance'] / MAX_IRRADIANCE
        image = np.asarray(Image.open(image_path))

        transformed = self._transforms(image=image)

        image = transformed['image']
        torch_image = torch.from_numpy(image).permute(2, 0, 1)

        if self._add_sun_mask:
            date = pd.to_datetime(image_path.name[:15], format='%Y%m%d_%H%M%S')
            us_pacific_date = self.us_pacific.localize(date)
            utc_date = us_pacific_date.astimezone(self.utc).strftime('%Y%m%d_%H%M%S')

            sun_mask = self._sun_mask(image_shape=image.shape, timestamp=utc_date)
            torch_image = torch.cat([torch_image, torch.from_numpy(sun_mask).permute(2, 0, 1)], dim=0)

        if self._add_irradiance_channel:
            if self._add_sun_mask:
                torch_image[-1] *= irradiance
            else:
                torch_image = torch.cat([torch_image, torch.full((1, *torch_image.shape[1:]), irradiance)], dim=0)

        if self._cloud_mask_method is not None:
            cloud_mask = self._cloud_mask(image=image)
            torch_image = torch.cat([torch_image, torch.from_numpy(cloud_mask).permute(2, 0, 1)], dim=0)

        if self._optical_flow is not None:
            flow_x_path = self._data_root.parent / 'flows' / self._optical_flow / history_item['image_name'].replace('.jpg', '_x.tiff')
            flow_y_path = self._data_root.parent / 'flows' / self._optical_flow / history_item['image_name'].replace('.jpg', '_y.tiff')

            flow_x = np.asarray(Image.open(flow_x_path).resize(self._image_size))
            flow_y = np.asarray(Image.open(flow_y_path).resize(self._image_size))

            torch_image = torch.cat([torch_image, torch.from_numpy(flow_x).unsqueeze(0), torch.from_numpy(flow_y).unsqueeze(0)], dim=0)

        source_irradiances = [h['irradiance'] / MAX_IRRADIANCE for h in period['history']]
        target_irradiance = period['target_irradiance'] / MAX_IRRADIANCE

        return (
            torch_image,
            torch.Tensor(source_irradiances),
            torch.Tensor([target_irradiance])
        )

    def __len__(self) -> int:
        return len(self._periods)
