import numpy as np
import pytest

from solar_irradiance.datamodules.cloud_mask import CloudMask


@pytest.mark.parametrize("image_shape, x_idx, y_idx, l2_dist_true", [
    ([1536, 1536], 919, 1150, 410.76),
    ([768, 768], 425, 138, 249.39),
    ([384, 384], 201, 300, 108.37),
    ([192, 192], 102, 103, 9.22),
    ([96, 96], 31, 42, 18.03),
])
def test_l2_distance(image_shape, x_idx, y_idx, l2_dist_true):
    cloud_mask_gen = CloudMask(shape=image_shape, method="red_blue_ratio")

    l2_dist = cloud_mask_gen.l2_distance(x_idx, y_idx)

    assert np.isclose(l2_dist_true, l2_dist, rtol=1e-2, atol=1e-3)


@pytest.mark.parametrize("image_shape, method", [
    ([1536, 1536], "red_blue_ratio"),
    ([1536, 1536], "red_blue_difference"),
    ([768, 768], "normalized_blue_red_ratio"),
    ([768, 768], "red_blue_ratio"),
    ([384, 384], "red_blue_difference"),
    ([192, 192], "normalized_blue_red_ratio"),
])
def test_class_call(image_shape, method):
    cloud_mask_gen = CloudMask(shape=image_shape, method=method)

    cloud_mask = cloud_mask_gen(np.zeros((*image_shape, 3)))

    assert cloud_mask.shape == (*image_shape, 1)


@pytest.mark.parametrize("image, true_mask", [
    (np.ones((768, 768, 3), dtype=np.uint8) * [255, 0, 0], np.zeros((768, 768, 1), dtype=np.float32)),
    (np.ones((384, 384, 3), dtype=np.uint8) * [0, 0, 255], np.zeros((384, 384, 1), dtype=np.float32)),
    (np.ones((192, 192, 3), dtype=np.uint8) * [0, 0, 0], np.zeros((192, 192, 1), dtype=np.float32)),
])
def test_blue_red_ratio(image, true_mask):
    cloud_mask_gen = CloudMask(shape=image.shape[:2], method="red_blue_ratio")
    cloud_mask = cloud_mask_gen(image)

    # assert np.array_equal(true_mask, cloud_mask)
    assert true_mask.dtype == cloud_mask.dtype


@pytest.mark.parametrize("image, true_mask", [
    (np.zeros((768, 768, 3), dtype=np.uint8) * 255, np.zeros((768, 768, 1), dtype=np.float32)),
    (np.zeros((384, 384, 3), dtype=np.uint8), np.zeros((384, 384, 1), dtype=np.float32) / 255),
])
def test_blue_red_difference(image, true_mask):
    cloud_mask_gen = CloudMask(shape=image.shape[:2], method="red_blue_difference")
    cloud_mask = cloud_mask_gen(image)

    # assert np.array_equal(true_mask, cloud_mask)
    assert true_mask.dtype == cloud_mask.dtype


@pytest.mark.parametrize("image, true_mask", [
    (np.ones((768, 768, 3), dtype=np.uint8) * [255, 0, 0], np.zeros((768, 768, 1), dtype=np.float32)),
    (np.ones((384, 384, 3), dtype=np.uint8) * [0, 0, 255], np.zeros((384, 384, 1), dtype=np.float32)),
    (np.ones((192, 192, 3), dtype=np.uint8) * [0, 0, 0], np.zeros((192, 192, 1), dtype=np.float32)),
])
def test_normalized_blue_red_ratio(image, true_mask):
    cloud_mask_gen = CloudMask(shape=image.shape[:2], method="normalized_blue_red_ratio")

    cloud_mask = cloud_mask_gen(image)

    # assert np.array_equal(true_mask, cloud_mask)
    assert true_mask.dtype == cloud_mask.dtype
