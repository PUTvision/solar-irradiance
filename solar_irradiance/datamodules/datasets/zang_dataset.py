from pathlib import Path
from typing import Any

from albumentations import ReplayCompose
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

MAX_IRRADIANCE = 1466.0  # max irradiance in the dataset


class ZangFolsomForecastingDataset(Dataset):
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

    @staticmethod
    def _calculate_optical_flow(image_sequence_np):
        """
        Calculates Farneback optical flow.

        Parameters
        ----------
        image_sequence_np : np.array
            Shape [T, H, W, C] (uint8)

        Returns
        -------
        np.array
            Optical flow maps, shape [T, H, W, 2] (float32)

        Notes
        -----
        - The first frame (index 0) has zero flow (all values are zero).
        - For each subsequent frame at index j (j > 0), the flow is computed relative to the previous frame (j-1).
        - The last dimension of size 2 contains the horizontal (x) and vertical (y) flow components, respectively.
        """
        t, h, w, c = image_sequence_np.shape
        flow_maps = np.zeros((t, h, w, 2), dtype=np.float32)

        # Convert first frame to gray
        prev_gray = cv2.cvtColor(image_sequence_np[0], cv2.COLOR_BGR2GRAY)
        for j in range(1, t):
            gray = cv2.cvtColor(image_sequence_np[j], cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow_maps[j] = flow
            prev_gray = gray

        return flow_maps

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        period = self._periods[index]

        source_images = []
        images_for_optical_flows = []
        source_irradiances = []
        replay_data = None

        for history_item in period["history"]:
            image_path = self._data_root / "images" / history_item["image_name"]
            irradiance = history_item["irradiance"]
            image = np.asarray(Image.open(image_path))

            if replay_data is None:
                transformed = self._transforms(image=image)
                replay_data = transformed["replay"]
            else:
                transformed = self._transforms.replay(replay_data, image=image)

            image = transformed["image"]
            norm_image = (image / 255.0 - self._image_mean) / self._image_std
            cropped_image = np.where(self._crop_mask, norm_image, 0.0).astype(np.float32)

            source_images.append(cropped_image)
            images_for_optical_flows.append(image)
            source_irradiances.append(irradiance)

        target_irradiance = period["target_irradiance"]
        optical_flows = torch.from_numpy(self._calculate_optical_flow(np.array(images_for_optical_flows))).permute(0, 3, 1, 2)
        torch_images = torch.from_numpy(np.array(source_images)).permute(0, 3, 1, 2)

        return (torch_images, optical_flows, torch.Tensor(source_irradiances), torch.Tensor([target_irradiance]))

    def __len__(self) -> int:
        return len(self._periods)
