from pathlib import Path
from typing import Tuple

import albumentations as A
import pandas as pd
import torch.utils.data
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split

from solar_irradiance.datamodules.datasets.folsom_forecasting_dataset import FolsomForecastingDataset


class ForecastingDataModule(LightningDataModule):
    def __init__(
            self,
            root_data_path: Path,
            periods_path: Path,
            augment: bool,
            image_size: Tuple[int, int],
            image_mean: Tuple[float, float, float],
            image_std: Tuple[float, float, float],
            batch_size: int,
            workers: int,
            sun_mask: bool,
            blur_mask: bool,
            add_irradiance_channel: bool,
            seed: int,
    ):
        super().__init__()

        self._data_root = Path(root_data_path)
        self._periods_path = Path(periods_path)
        self._dataset_name = self._data_root.name
        self._augment = augment
        self._batch_size = batch_size
        self._workers = workers
        self._sun_mask = sun_mask
        self._blur_mask = blur_mask
        self._add_irradiance_channel = add_irradiance_channel
        self._seed = seed

        self._transforms = A.ReplayCompose([
            A.CenterCrop(image_size[1], image_size[0]),
            A.Normalize(mean=image_mean, std=image_std),
        ])
        self._augmentations = A.ReplayCompose([
            # geometry augmentations
            A.Affine(rotate=(-10, 10), translate_px=(-10, 10), scale=(0.9, 1.1)),
            A.Flip(),
            # transforms
            A.RandomCrop(image_size[1], image_size[0]),
            A.Normalize(mean=image_mean, std=image_std),
        ])

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

    def setup(self, stage: str) -> None:
        with self._periods_path.open('rb') as f:
            periods = pd.read_pickle(f)

        train_periods, val_periods = train_test_split(periods, test_size=0.2, random_state=self._seed)
        val_periods, test_periods = train_test_split(val_periods, test_size=0.5, random_state=self._seed)

        self._train_dataset = FolsomForecastingDataset(
            data_root=self._data_root,
            periods=train_periods,
            transforms=self._augmentations if self._augment else self._transforms,
            sun_mask=self._sun_mask,
            blur_mask=self._blur_mask,
            add_irradiance_channel=self._add_irradiance_channel,
        )
        self._val_dataset = FolsomForecastingDataset(
            data_root=self._data_root,
            periods=val_periods,
            transforms=self._transforms,
            sun_mask=self._sun_mask,
            blur_mask=self._blur_mask,
            add_irradiance_channel=self._add_irradiance_channel,
        )
        self._test_dataset = FolsomForecastingDataset(
            data_root=self._data_root,
            periods=test_periods,
            transforms=self._transforms,
            sun_mask=self._sun_mask,
            blur_mask=self._blur_mask,
            add_irradiance_channel=self._add_irradiance_channel,
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


