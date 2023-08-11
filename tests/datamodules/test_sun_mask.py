import cv2
import numpy as np
import pytest

from solar_irradiance.datamodules.sun_mask import SunMask


# Folsom dataset
LATITUDE = 38.642
LONGITUDE = -121.148
CAMERA_ORIENTATION_COMPENSATION = 165
FOCAL_LENGTH = 0.48

sun_mask_gen = SunMask(LATITUDE, LONGITUDE, CAMERA_ORIENTATION_COMPENSATION, FOCAL_LENGTH)


@pytest.mark.parametrize("timestamp, image_shape, x_true, y_true", [
    ("20141114_201400", [1536, 1536], 919, 1141),
    ("20150114_161300", [1536, 1536], 357, 1263),
    ("20150214_151459", [768, 768], 91, 580),
    ("20150214_201359", [768, 768], 425, 557),
    ("20160614_201400", [384, 384], 201, 215),
    ("20160814_201400", [192, 192], 102, 115),
])
def test_class_call(timestamp, image_shape, x_true, y_true):
    true_mask = np.zeros((*image_shape, 1), np.uint8)
    cv2.circle(true_mask, (x_true, y_true), 100, 255, -1)
    true_mask = (true_mask / 255).astype(np.float32)

    sun_mask = sun_mask_gen(image_shape=image_shape, timestamp=timestamp)

    assert np.array_equal(true_mask, sun_mask)


@pytest.mark.parametrize("timestamp, image_shape, x_true, y_true", [
    ("20141114_201400", [1536, 1536], 919, 1141),
    ("20150114_161300", [1536, 1536], 357, 1263),
    ("20150214_151459", [768, 768], 91, 580),
    ("20150214_201359", [768, 768], 425, 557),
    ("20160614_201400", [384, 384], 201, 215),
    ("20160814_201400", [192, 192], 102, 115),
])
def test_calculate_sun_center_in_image(timestamp, image_shape, x_true, y_true):
    x, y = sun_mask_gen._calculate_sun_center_in_image(timestamp, image_shape)

    assert x_true == x
    assert y_true == y
