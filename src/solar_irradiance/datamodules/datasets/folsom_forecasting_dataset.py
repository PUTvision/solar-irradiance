from pathlib import Path
from typing import Tuple, List, Dict, Any, Union

import cv2
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


OPTICAL_FLOWS = {
    'dis': cv2.DISOpticalFlow_create(preset=cv2.DISOPTICAL_FLOW_PRESET_FAST),
    'farneback': cv2.optflow.createOptFlow_Farneback(),
    'deep_flow': cv2.optflow.createOptFlow_DeepFlow(),
    'pca_flow': cv2.optflow.createOptFlow_PCAFlow(),
    'dual_tvl1': cv2.optflow.createOptFlow_DualTVL1(),
    'dense_rlof': cv2.optflow.createOptFlow_DenseRLOF(),    # requires RGB input
}


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
            add_sun_mask: bool,
            add_irradiance_channel: bool,
            optical_flow: Union[None, str],
    ):
        self._data_root = data_root
        self._periods = periods
        self._transforms = transforms
        self._add_sun_mask = add_sun_mask
        self._sun_mask = SunMask(self.latitude, self.longitude, self.camera_orientation_compensation, self.focal_length)
        self._add_irradiance_channel = add_irradiance_channel
        self._of = OPTICAL_FLOWS.get(optical_flow)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        period = self._periods[index]

        source_images = []
        source_irradiances = []
        replay_data = None
        flow = None
        prev_image_gray = None

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
            torch_image = torch.from_numpy(image).permute(2, 0, 1)

            if self._add_sun_mask:
                sun_mask = self._sun_mask(image=image, timestamp=image_path.name[:15])
                torch_image = torch.cat([torch_image, torch.from_numpy(sun_mask).permute(2, 0, 1)], dim=0)

            if self._add_irradiance_channel:
                if self._add_sun_mask:
                    torch_image[-1] *= irradiance
                else:
                    torch_image = torch.cat([torch_image, torch.full((1, *torch_image.shape[1:]), irradiance)], dim=0)

            if self._of is not None:
                image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                if prev_image_gray is None:
                    prev_image_gray = image_gray.copy()

                try:
                    flow = self._of.calc(prev_image_gray, image_gray, flow)
                except:
                    flow = np.zeros((image.shape[0], image.shape[1], 2), dtype=np.uint8)

                prev_image_gray = image_gray

                torch_image = torch.cat([torch_image, torch.from_numpy(flow).permute(2, 0, 1)], dim=0)

            source_images.append(torch_image)
            source_irradiances.append(irradiance)

        target_irradiance = period['target_irradiance'] / MAX_IRRADIANCE

        return (
            torch.stack(source_images).permute(1, 0, 2, 3),
            # source_images[-1],
            torch.Tensor(source_irradiances),
            torch.Tensor([target_irradiance])
        )

    def __len__(self) -> int:
        return len(self._periods)
