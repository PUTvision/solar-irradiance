from pathlib import Path
from typing import Any

from albumentations import ReplayCompose
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import pvlib
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

    def get_brightness(self, image):
        """
        Calculates the mean brightness (intensity) of the image.

        The paper (Section 3.1.2) defines this as the "mean grayscale of an image."
        This function converts the image to grayscale and returns its mean pixel value.

        Args:
            image (np.ndarray): The input BGR image.

        Returns:
            float: The mean brightness value (0-255).
        """
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate and return the mean value of the grayscale image
        brightness = np.mean(gray_image)
        return brightness

    def get_cloud_pixels(self, image, rbr_threshold=0.8):
        """
        Calculates the number of cloud pixels using the Red-Blue Ratio (RBR).

        The paper (Section 3.1.2) specifies using the RBR algorithm with a
        fixed threshold (Tf) of 0.8. A pixel is classified as a cloud if (R/B) > 0.8.

        Args:
            image (np.ndarray): The input BGR image.
            rbr_threshold (float, optional): The threshold for R/B. Defaults to 0.8.

        Returns:
            int: The total count of pixels classified as cloud.
        """
        # Split the BGR image into its respective channels
        b_channel, g_channel, r_channel = cv2.split(image)

        # Convert to float to for safe division and to avoid type errors
        r_float = r_channel.astype(float)
        b_float = b_channel.astype(float)

        # Calculate RBR. Add a small epsilon (1e-6) to the denominator
        # to prevent division by zero in case of pure black pixels (b_channel=0).
        rbr = r_float / (b_float + 1e-6)

        # Create a boolean mask where RBR > threshold
        cloud_mask = rbr > rbr_threshold

        # Count the number of True values (cloud pixels) in the mask
        cloud_pixel_count = np.sum(cloud_mask)

        return int(cloud_pixel_count)

    def get_edge_count(self, image, canny_threshold1=100, canny_threshold2=200):
        """
        Calculates the number of edge pixels using the Canny edge detector.

        The paper (Section 3.1.2) mentions using the Canny algorithm but does
        not specify the thresholds. We use common default values (100, 200).

        Args:
            image (np.ndarray): The input BGR image.
            canny_threshold1 (int, optional): The first threshold for Canny.
            canny_threshold2 (int, optional): The second threshold for Canny.

        Returns:
            int: The total count of pixels classified as edges.
        """
        # Canny edge detection works on grayscale images
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply the Canny edge detector
        edges = cv2.Canny(gray_image, canny_threshold1, canny_threshold2)

        # Count the number of non-zero pixels (which represent edges)
        edge_count = np.count_nonzero(edges)

        return int(edge_count)

    def get_corner_count(self, image, harris_threshold_ratio=0.01):
        """
        Calculates the number of corners using the Harris algorithm.

        The paper (Section 3.1.2) mentions using the Harris algorithm.
        This algorithm returns a "cornerness" score for each pixel. To get a
        "count," we must threshold this result. A common method is to
        threshold based on a ratio of the maximum cornerness score.

        Args:
            image (np.ndarray): The input BGR image.
            harris_threshold_ratio (float, optional): Ratio of the max corner
                score to use as a threshold.

        Returns:
            int: The total count of pixels classified as corners.
        """
        # Harris corner detection requires a grayscale float32 image
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_float = np.float32(gray_image)

        # Apply Harris corner detection
        # Parameters: (image, blockSize, ksize, k)
        # These are standard parameters for the algorithm.
        corner_scores = cv2.cornerHarris(gray_float, 2, 3, 0.04)

        # Threshold the corner scores to get "strong" corners
        # We define a corner as any pixel with a score > 1% of the max score
        threshold = harris_threshold_ratio * corner_scores.max()
        corner_mask = corner_scores > threshold

        # Count the number of True values (corners)
        corner_count = np.sum(corner_mask)

        return int(corner_count)

    def get_clear_sky_values(self, timestamp):
        """
        Calculates the Clear Sky GHI using the pvlib library.

        This function uses the Ineichen-Perez model, which is a standard
        and referenced in the paper's context.

        Args:
            timestamp (pd.Timestamp): A pandas Timestamp.

        Returns:
            float: The clear-sky GHI value.
        """

        # 1. Define the location
        location = pvlib.location.Location(latitude=self.latitude, longitude=self.longitude, altitude=self.altitude)

        # 2. Convert single Timestamp to DatetimeIndex (pvlib requires this)
        if isinstance(timestamp, pd.Timestamp):
            timestamp = pd.DatetimeIndex([timestamp])

        # 3. Get the clear-sky irradiance
        clearsky = location.get_clearsky(timestamp)

        # 4. Return the scalar GHI value (first element since we only have one timestamp)
        return clearsky["ghi"].iloc[0]

    def calculate_csi(self, measured_ghi, clear_sky_ghi):
        """
        Calculates the Clear Sky Index (CSI).

        CSI = GHI_measured / GHI_clear_sky

        Args:
            measured_ghi (pd.Series): Your actual sensor measurements.
            clear_sky_ghi (pd.Series): The modeled clear-sky GHI.

        Returns:
            pd.Series: The calculated Clear Sky Index.
        """
        if clear_sky_ghi == 0:
            return 0.0

        # Create a copy to avoid modifying the original series
        csi = measured_ghi / clear_sky_ghi

        # Cap CSI at 1.2 for edge cases
        csi = min(csi, 1.2)

        if np.isinf(csi):
            return 0.0

        return csi

    def get_solar_features(self, timestamp):
        """
        Calculates the primary solar features for a given location and time.

        Args:
            timestamp (pd.Timestamp): The timestamp for which to calculate solar features.

        Returns:
            tuple: (zenith, azimuth, sun_earth_distance)
        """
        # Convert single Timestamp to DatetimeIndex
        if isinstance(timestamp, pd.Timestamp):
            timestamp = pd.DatetimeIndex([timestamp])

        # Get solar position
        solar_pos = pvlib.solarposition.get_solarposition(
            time=timestamp, latitude=self.latitude, longitude=self.longitude, altitude=self.altitude
        )

        return solar_pos["zenith"].iloc[0], solar_pos["azimuth"].iloc[0], solar_pos["apparent_elevation"].iloc[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        period = self._periods[index]

        features_list = []

        for history_item in period["history"]:
            image_path = self._data_root / "images" / history_item["image_name"]
            irradiance = history_item["irradiance"]  # / MAX_IRRADIANCE
            image = np.asarray(Image.open(image_path))

            brightness = self.get_brightness(image)
            cloud_pixels = self.get_cloud_pixels(image)
            edge_count = self.get_edge_count(image)
            corner_count = self.get_corner_count(image)
            # Convert filename from yyyymmdd_HHMMSS format to datetime
            timestamp = pd.to_datetime(image_path.stem, format="%Y%m%d_%H%M%S")
            timestamp = timestamp.tz_localize(self.us_pacific)
            clear_sky_values = self.get_clear_sky_values(timestamp=timestamp)
            csi = self.calculate_csi(irradiance, clear_sky_values)
            zenith, azimuth, sun_earth_distance = self.get_solar_features(timestamp=timestamp)

            features_list.append(
                [irradiance, brightness, cloud_pixels, edge_count, corner_count, csi, zenith, azimuth, sun_earth_distance]
            )

        target_irradiance = period["target_irradiance"]  # / MAX_IRRADIANCE

        return (torch.Tensor(features_list), torch.Tensor(features_list), torch.Tensor([target_irradiance]))

    def __len__(self) -> int:
        return len(self._periods)
