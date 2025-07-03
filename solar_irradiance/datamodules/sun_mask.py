import cv2
import numpy as np
import pandas as pd
import pvlib

# Folsom dataset
# LATITUDE = 38.642
# LONGITUDE = -121.148
# CAMERA_ORIENTATION_COMPENSATION = 165
# FOCAL_LENGTH = 0.48


class SunMask:
    """Creates sun mask based on camera location and orientation."""

    def __init__(
        self,
        latitude: float,
        longitude: float,
        camera_orientation_compensation: int,
        focal_length: float,
    ):
        self._latitude = latitude
        self._longitude = longitude
        self._camera_orientation_compensation = camera_orientation_compensation
        self._focal_length = focal_length

    def __call__(self, image_shape: np.ndarray, timestamp: str) -> np.ndarray:
        """_summary_

        Parameters
        ----------
        image_shape
            Shape of the image.
        timestamp
            Image timestamp in UTC format.

        Returns
        -------
            Sun mask with the same shape as provided with image_shape param and with 1 channel.
        """
        x, y = self._calculate_sun_center_in_image(timestamp, image_shape[:2])

        mask_shape = (image_shape[0], image_shape[1], 1)
        mask = np.zeros(mask_shape, np.uint8)
        cv2.circle(mask, (x, y), image_shape[0] // 16, 255, -1)

        mask = (mask / 255).astype(np.float32)

        return mask

    def _calculate_sun_center_in_image(self, timestamp: str, image_shape: tuple[int, int]) -> tuple[int]:
        """Calculates coordinates of sun center in image for provided timestamp.

        Parameters
        ----------
        timestamp
            Image timestamp in UTC format.
        image_shape
            Shape of the image.

        Returns
        -------
            Sun center coordinates in image.
        """
        solarposition = pvlib.solarposition.get_solarposition(
            time=pd.to_datetime(timestamp, format="%Y%m%d_%H%M%S"), latitude=self._latitude, longitude=self._longitude
        )
        zenith = solarposition["zenith"][0]
        azimuth = solarposition["azimuth"][0]

        # compensation of camera orientation
        azimuth -= self._camera_orientation_compensation

        radius = 2 * self._focal_length * np.tan(np.deg2rad(zenith) / 2)

        x_sc = radius * np.sin(np.deg2rad(azimuth))
        y_sc = radius * np.cos(np.deg2rad(azimuth))

        # convert to the image plane
        x_scp = image_shape[1] / 2 + x_sc * image_shape[1] / 2
        y_scp = image_shape[0] / 2 + y_sc * image_shape[0] / 2

        return int(x_scp), int(y_scp)
