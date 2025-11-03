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


class FolsomForecastingDataset(Dataset):
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
            
    def detect_sun_in_image(self, rgb_image, red_threshold=240):
        """
        Detects the sun's position based on a red channel threshold.
        doi: 10.1109/IGARSS.2016.7730949
        """
        # --- 1. Isolate the Red Channel ---
        R = rgb_image[:, :, 0] # Assumes RGB order (R=0, G=1, B=2)

        # --- 2. Apply Thresholding ---
        # Create a binary mask where Red > threshold
        ret, sun_mask = cv2.threshold(R, red_threshold, 255, cv2.THRESH_BINARY)
        
        # --- 3. Find the Largest Area (Contour) ---
        contours, _ = cv2.findContours(
            sun_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            # Sun is obscured or not bright enough
            return False
        return True
    
    def calculate_cloud_cover(self, rgb_image, fixed_thresh_value=128):
        """
        Calculates the cloud cover based on the provided flowchart.
        https://doi.org/10.1016/j.solener.2025.113516
        
        Returns:
            float: The calculated cloud cover.
        """
        
        # --- 1. Pre-processing ---
        # Convert RGB to grayscale
        gray_image_8bit = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        
        # Normalize image to [0.0, 1.0] for intensity check
        normalized_image = gray_image_8bit / 255.0
        
        # Get total number of pixels for CC calculation
        total_pixels = gray_image_8bit.size
        
        # --- 2. Detect sun in image ---
        sun_present = self.detect_sun_in_image(rgb_image)
            
        # --- 3. Sun in image? (Decision) ---
        if not sun_present:
            # Fixed thresholding: Pixel < threshold = cloud
            # We use THRESH_BINARY_INV, which makes pixels *below* the
            # threshold white (255), and pixels above it black (0).
            ret, binary_image = cv2.threshold(
                gray_image_8bit, fixed_thresh_value, 255, cv2.THRESH_BINARY_INV
            )
            
            # Count white pixels (255), which represent clouds
            num_cloud_pixels = np.count_nonzero(binary_image)
            
        else:
            # Average (Avg.) pixel intensity
            avg_intensity = np.mean(normalized_image)
            
            # --- 4. Avg. intensity > 0.51? (Decision) ---
            if avg_intensity <= 0.51:
                # Identified as clear sky
                num_cloud_pixels = 0
                
            else:
                # Otsu adaptive thresholding
                # white pixels (255) = sky, black pixel (0) = cloud
                
                # cv2.THRESH_OTSU finds the optimal threshold.
                # cv2.THRESH_BINARY applies it:
                #   - Pixels above threshold become 255 (white)
                #   - Pixels below threshold become 0 (black)
                # This matches the flowchart's logic (white=sky, black=cloud).
                ret_otsu, otsu_image = cv2.threshold(
                    gray_image_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                
                # We need to count the black pixels (clouds)
                # This is total pixels minus white pixels (sky)
                num_sky_pixels = np.count_nonzero(otsu_image)
                num_cloud_pixels = total_pixels - num_sky_pixels

        # --- 5. Calculate cloud cover (CC) ---
        cloud_cover = (num_cloud_pixels / total_pixels)
        return cloud_cover


    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        period = self._periods[index]

        source_images = []
        source_irradiances = []
        cloud_covers = []
        replay_data = None
        flow = None
        prev_image = None

        for history_item in period["history"]:
            image_path = self._data_root / "images" / history_item["image_name"]
            irradiance = history_item["irradiance"]  # / MAX_IRRADIANCE
            image = np.asarray(Image.open(image_path))
            cloud_cover = self.calculate_cloud_cover(image)

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
            cloud_covers.append(cloud_cover)

        target_irradiance = period["target_irradiance"]  # / MAX_IRRADIANCE

        image_input = source_images[-1]
        source_irradiances.append(cloud_covers[-1])

        return (image_input, torch.Tensor(source_irradiances), torch.Tensor([target_irradiance]))

    def __len__(self) -> int:
        return len(self._periods)
