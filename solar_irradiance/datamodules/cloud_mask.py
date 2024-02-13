from typing import Tuple

import cv2
import numpy as np


class CloudMask:
    def __init__(self, shape: Tuple[int, int], method: str):
        self.shape = shape
        self.method = method
        if self.method == 'red_blue_ratio':
            self.segmentation_func = self.red_blue_ratio
        elif self.method == 'red_blue_difference':
            self.segmentation_func = self.red_blue_difference
        elif self.method == 'normalized_blue_red_ratio':
            self.segmentation_func = self.normalized_blue_red_ratio

        self.dist_mask = np.zeros(shape, dtype=float)

        for x_idx in range(shape[1]):
            for y_idx in range(shape[0]):
                l2_dist = self.l2_distance(x_idx, y_idx)
                self.dist_mask[y_idx][x_idx] = l2_dist

    def __call__(self, image: np.ndarray) -> np.ndarray:
        mask = self.segmentation_func(image)

        return mask[..., np.newaxis]

    def l2_distance(self, x_idx, y_idx) -> float:
        return np.sqrt((self.shape[1] // 2 - x_idx)**2 + (self.shape[0] // 2 - y_idx)**2)

    def red_blue_ratio(self, img) -> np.ndarray:
        # R2B https://journals.ametsoc.org/view/journals/atot/23/5/jtech1875_1.xml
        img = img.astype(np.float32)
        img[..., 0] = np.where(img[..., 2] == 0, img[..., 0] + 1, img[..., 0])
        img[..., 2] = np.where(img[..., 2] == 0, img[..., 2] + 1, img[..., 2])
        mask = np.divide(img[..., 0], img[..., 2])
        mask = np.where(self.dist_mask > self.shape[0] // 2, 0, mask)
        return (mask / 256.).astype(np.float32) # scale to 0-1 range

    def red_blue_difference(self, img) -> np.ndarray:
        # https://amt.copernicus.org/articles/3/557/2010/amt-3-557-2010.html
        mask = np.where(self.dist_mask > self.shape[0] // 2, 0, img[..., 0] - img[..., 2])
        return ((mask + 255.) / 510.).astype(np.float32) # scale to 0-1 range

    def normalized_blue_red_ratio(self, img) -> np.ndarray:
        # https://journals.ametsoc.org/view/journals/atot/28/10/jtech-d-11-00009_1.xml
        img = img.astype(np.float32)
        img[..., 2] = np.where(img[..., 0] == 0, img[..., 2] + 1, img[..., 2])
        img[..., 0] = np.where(img[..., 0] == 0, img[..., 0] + 1, img[..., 0])
        br_ratio = np.divide(img[..., 2], img[..., 0])

        mask = (br_ratio - 1) / (br_ratio + 1)
        mask = np.where(self.dist_mask > self.shape[0] // 2, 0, mask)
        return ((mask + 1) * 257. / 512.).astype(np.float32)  # scale to 0-1 range
