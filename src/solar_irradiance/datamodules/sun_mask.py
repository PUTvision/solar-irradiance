from typing import Tuple

import cv2
import numpy as np
import pandas as pd
import pvlib
from scipy.ndimage import gaussian_filter


# Folsom dataset
# LATITUDE = 38.642
# LONGITUDE = -121.148
# CAMERA_ORIENTATION_COMPENSATION = 165
# FOCAL_LENGTH = 0.48


class SunMask:
    def __init__(
            self,
            latitude: float,
            longitude: float,
            camera_orientation_compensation: int,
            focal_length: float,
            blur_mask: bool = False,
        ):
        self._latitude = latitude
        self._longitude = longitude
        self._camera_orientation_compensation = camera_orientation_compensation
        self._focal_length = focal_length
        self._blur_mask = blur_mask

    def __call__(self, image: np.ndarray, timestamp: str) -> np.ndarray:
        x, y = self._calculate_sun_center_in_image(timestamp, image.shape[:2])

        mask_shape = (image.shape[0], image.shape[1], 1)
        mask = np.zeros(mask_shape, np.uint8)
        cv2.circle(mask, (x, y), 100, 255, -1)

        if self._blur_mask:
            mask = gaussian_filter(mask, sigma=9)

        return np.concatenate([image, mask], axis=2)

    def _calculate_sun_center_in_image(
            self,
            timestamp: str,
            image_size: Tuple[int, int],
        ) -> Tuple[int]:

        solarposition = pvlib.solarposition.get_solarposition(
            time=pd.to_datetime(timestamp, format='%Y%m%d_%H%M%S'),
            latitude=self._latitude,
            longitude=self._longitude
        )
        zenith = solarposition['zenith'][0]
        azimuth = solarposition['azimuth'][0]

        # compensation of camera orientation
        azimuth -= self._camera_orientation_compensation 
        
        R = 2 * self._focal_length * np.tan(np.deg2rad(zenith) / 2)

        x_sc = R * np.sin(np.deg2rad(azimuth))
        y_sc = R * np.cos(np.deg2rad(azimuth))

        # convert to the image plane
        x_scp = image_size[1] / 2 + x_sc * image_size[1]/2
        y_scp = image_size[0] / 2 + y_sc * image_size[0]/2
        
        return int(x_scp), int(y_scp)
    