from pathlib import Path
from typing import Any

from albumentations import ReplayCompose
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import pytz
import torch
from torch.utils.data import Dataset

from solar_irradiance.datamodules.cloud_mask import CloudMask
from solar_irradiance.datamodules.sun_mask import SunMask

MAX_IRRADIANCE = 1466.0  # max irradiance in the dataset
# MAX_IRRADIANCE = 1600.0     # max irradiance from Hukseflux pyranometer


OPTICAL_FLOWS = {
    "dis": cv2.DISOpticalFlow_create(preset=cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST),
    "farneback": cv2.optflow.createOptFlow_Farneback(),
    "deep_flow": cv2.optflow.createOptFlow_DeepFlow(),
    "pca_flow": cv2.optflow.createOptFlow_PCAFlow(),
    "dual_tvl1": cv2.optflow.createOptFlow_DualTVL1(),
    "dense_rlof": cv2.optflow.createOptFlow_DenseRLOF(),  # requires RGB input
}


class FolsomForecastingDataset2D(Dataset):
    latitude = 38.642
    longitude = -121.148
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
        self._transforms = transforms
        self._add_sun_mask = add_sun_mask
        self._sun_mask = SunMask(self.latitude, self.longitude, self.camera_orientation_compensation, self.focal_length)
        self._add_irradiance_channel = add_irradiance_channel
        self._of = OPTICAL_FLOWS.get(optical_flow)
        self._optical_flow = optical_flow
        self._cloud_mask_method = cloud_mask_method
        self._image_size = image_size
        self._image_mean = image_mean
        self._image_std = image_std

        self._crop_mask = cv2.circle(
            np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8),
            (image_size[1] // 2, image_size[0] // 2),
            image_size[0] // 2,
            color=(1, 1, 1),
            thickness=-1,
        )

        if self._cloud_mask_method is not None:
            self._cloud_mask = CloudMask(shape=self._image_size, method=cloud_mask_method)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        period = self._periods[index]

        source_images = []
        source_irradiances = []
        replay_data = None
        flow = None
        prev_image = None

        for history_item in period["history"]:
            image_path = self._data_root / "images" / history_item["image_name"]
            irradiance = history_item["irradiance"]  # / MAX_IRRADIANCE
            image = np.asarray(Image.open(image_path))

            if replay_data is None:
                transformed = self._transforms(image=image)
                replay_data = transformed["replay"]
            else:
                transformed = self._transforms.replay(replay_data, image=image)

            image = transformed["image"]
            norm_image = (image / 255.0 - self._image_mean) / self._image_std
            cropped_image = np.where(self._crop_mask, norm_image, 0.0).astype(np.float32)
            torch_image = torch.from_numpy(cropped_image).permute(2, 0, 1)

            if self._add_sun_mask:
                date = pd.to_datetime(image_path.name[:15], format="%Y%m%d_%H%M%S")
                us_pacific_date = self.us_pacific.localize(date)
                utc_date = us_pacific_date.astimezone(self.utc).strftime("%Y%m%d_%H%M%S")

                sun_mask = self._sun_mask(image_shape=image.shape, timestamp=utc_date)
                sun_mask = self._transforms.replay(replay_data, image=sun_mask)["image"]
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
                image_for_flow = image if self._optical_flow == "dense_rlof" else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

                if prev_image is None:
                    prev_image = image_for_flow.copy()

                flow = self._of.calc(prev_image, image_for_flow, flow)
                prev_image = image_for_flow
                torch_image = torch.cat([torch_image, torch.from_numpy(flow).permute(2, 0, 1)], dim=0)

            source_images.append(torch_image)
            source_irradiances.append(irradiance)

        target_irradiance = period["target_irradiance"]  # / MAX_IRRADIANCE

        image_input = source_images[-1]

        return (image_input, torch.Tensor(source_irradiances), torch.Tensor([target_irradiance]))

    def __len__(self) -> int:
        return len(self._periods)
