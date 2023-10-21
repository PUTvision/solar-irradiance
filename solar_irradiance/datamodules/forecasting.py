from pathlib import Path
from typing import Tuple, Union

import albumentations as A
import numpy as np
import pandas as pd
import torch.utils.data
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split

from solar_irradiance.datamodules.datasets import FolsomForecastingDataset2D, FolsomForecastingDataset3D


class ForecastingDataModule(LightningDataModule):
    def __init__(
            self,
            root_data_path: Path,
            periods_path: Path,
            augment: bool,
            train_val_set_size: float,
            image_size: Tuple[int, int],
            image_mean: Tuple[float, float, float],
            image_std: Tuple[float, float, float],
            batch_size: int,
            workers: int,
            add_sun_mask: bool,
            add_irradiance_channel: bool,
            optical_flow: Union[None, str],
            cloud_mask_method: Union[None, str],
            model_2D: bool,
            seed: int,
    ):
        super().__init__()

        self._data_root = Path(root_data_path)
        self._periods_path = Path(periods_path)
        self._dataset_name = self._data_root.name
        self._augment = augment
        self._train_val_set_size = train_val_set_size
        self._image_size = image_size
        self._batch_size = batch_size
        self._workers = workers
        self._add_sun_mask = add_sun_mask
        self._add_irradiance_channel = add_irradiance_channel
        self._optical_flow = optical_flow
        self._cloud_mask_method = cloud_mask_method
        self._seed = seed

        self._dataset = FolsomForecastingDataset2D if model_2D else FolsomForecastingDataset3D

        self._transforms = A.ReplayCompose([
            A.Resize(image_size[1], image_size[0]),
            A.Normalize(mean=image_mean, std=image_std),
        ])
        self._augmentations = A.ReplayCompose([
            # geometry augmentations
            A.Affine(rotate=(-10, 10), translate_px=(-10, 10), scale=(0.9, 1.1)),
            A.Flip(),
            # transforms
            A.Resize(image_size[1], image_size[0]),
            A.Normalize(mean=image_mean, std=image_std),
        ])

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

    def setup(self, stage: str) -> None:
        with self._periods_path.open('rb') as f:
            periods = pd.read_pickle(f)

        test_periods = list(filter(lambda p: p['history'][-1]['image_name'].startswith('2014'), periods))
        train_val_periods = list(filter(lambda p: not p['history'][-1]['image_name'].startswith('2014'), periods))

        size = 170  # number of days for validatation dataset to get 80-20 ratio of train-val datasets
        np.random.seed(self._seed)
        val_dates = [str(y) + str(m).zfill(2) + str(d).zfill(2) for y, m, d in zip(
            np.random.randint(2015, 2017, size=size),
            np.random.randint(1, 13, size=size),
            np.random.randint(1, 29, size=size),
        )]
        val_periods = list(filter(lambda p: p['history'][-1]['image_name'][:8] in val_dates, train_val_periods))
        train_periods = list(filter(lambda p: p['history'][-1]['image_name'][:8] not in val_dates, train_val_periods))

        if self._train_val_set_size < 1:
            val_periods, __ = train_test_split(val_periods, train_size=self._train_val_set_size, random_state=self._seed)
            train_periods, __ = train_test_split(train_periods, train_size=self._train_val_set_size, random_state=self._seed)

        self._train_dataset = self._dataset(
            data_root=self._data_root,
            periods=train_periods,
            transforms=self._augmentations if self._augment else self._transforms,
            add_sun_mask=self._add_sun_mask,
            add_irradiance_channel=self._add_irradiance_channel,
            optical_flow=self._optical_flow,
            cloud_mask_method=self._cloud_mask_method,
            image_size=self._image_size,
        )
        self._val_dataset = self._dataset(
            data_root=self._data_root,
            periods=val_periods,
            transforms=self._transforms,
            add_sun_mask=self._add_sun_mask,
            add_irradiance_channel=self._add_irradiance_channel,
            optical_flow=self._optical_flow,
            cloud_mask_method=self._cloud_mask_method,
            image_size=self._image_size,
        )
        self._test_dataset = self._dataset(
            data_root=self._data_root,
            periods=test_periods,
            transforms=self._transforms,
            add_sun_mask=self._add_sun_mask,
            add_irradiance_channel=self._add_irradiance_channel,
            optical_flow=self._optical_flow,
            cloud_mask_method=self._cloud_mask_method,
            image_size=self._image_size,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self._train_dataset, batch_size=self._batch_size, num_workers=self._workers,
            pin_memory=True, drop_last=True, shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self._val_dataset, batch_size=self._batch_size, num_workers=self._workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self._test_dataset, batch_size=self._batch_size, num_workers=self._workers,
            pin_memory=True
        )
