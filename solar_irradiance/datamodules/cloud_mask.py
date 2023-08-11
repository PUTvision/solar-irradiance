from typing import Tuple

import cv2
import numpy as np


class CloudMask:
    def __init__(self, shape: Tuple[int, int], method: str):
        self.shape = shape
        self.method = method
        if self.method == 'blue_red_ratio':
            self.segmentation_func = self.blue_red_ratio
        elif self.method == 'blue_red_difference':
            self.segmentation_func = self.blue_red_difference
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
        return np.sqrt((self.shape[1]//2 - x_idx)**2 + (self.shape[0]//2 - y_idx)**2)
    
    def blue_red_ratio(self, img) -> np.ndarray:
        mask = np.where(self.dist_mask > self.shape[0]//2, 0, img[..., 2] / np.add(img[..., 0], [1]))
        return (mask / 255).astype(np.float32)

    def blue_red_difference(self, img) -> np.ndarray:
        mask = np.where(self.dist_mask > self.shape[0]//2, 0, img[..., 2] - img[..., 0])
        return ((mask + 255) / 510).astype(np.float32)

    def normalized_blue_red_ratio(self, img) -> np.ndarray:
        # https://journals.ametsoc.org/view/journals/atot/28/10/jtech-d-11-00009_1.xml
        br_ratio = img[..., 2] / np.add(img[..., 0], [1])
        mask = np.where(self.dist_mask > self.shape[0]//2, 1, (br_ratio - 1) / (br_ratio + 1))
        mask = np.where((-0.03 < mask) & (mask < 0.03), 255, 0).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=np.ones((5,5), np.uint8), iterations=3)
        return (mask / 255).astype(np.float32)
