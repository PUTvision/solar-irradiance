import itertools
from collections import deque
from pathlib import Path
from random import Random
from typing import Optional, List, Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import hydra
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from solar_irradiance.datamodules.datasets.folsom_dataset import FolsomDataset
from solar_irradiance.datamodules.datasets.sirta_dataset import SIRTADataset


class RegressionDataModule(LightningDataModule):
    def __init__(
            self,
            root_data_path: Path,
            augment: bool,
            image_size: Tuple[int, int],
            padded_image_size: Tuple[int, int],
            image_mean: Tuple[float, float, float],
            image_std: Tuple[float, float, float],
            batch_size: int,
            workers: int,
            number_of_splits: int,
            current_split: int,
        ):
        super().__init__()

        self._data_root = Path(root_data_path)
        self._dataset_name = self._data_root.name
        self._augment = augment
        self._batch_size = batch_size
        self._workers = workers
        self._number_of_splits = number_of_splits
        self._current_split = current_split

        if self._dataset_name == 'Folsom':
            self._dataset = FolsomDataset
        elif self._dataset_name == 'SIRTA':
            self._dataset = SIRTADataset
        else:
            raise ValueError(f'Dataset "{self._dataset_name}" not supported.')

        self._train_dataset = None
        self._valid_dataset = None
        self._test_dataset = None

        self._transforms = A.Compose([
            A.CenterCrop(image_size[1], image_size[0]),
            A.PadIfNeeded(padded_image_size[1], padded_image_size[0], border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Normalize(mean=image_mean, std=image_std),
            ToTensorV2()
        ])

        self._augmentations = A.Compose([
            # rgb augmentations
            # A.RandomSunFlare(),
            # A.RandomGamma(gamma_limit=(80, 120)),
            # A.ColorJitter(brightness=0.1, contrast=0.1, hue=0.01, saturation=0.5),
            # A.ISONoise(color_shift=(0.01, 0.1)),
            # geometry augmentations
            A.Affine(rotate=(-10, 10), translate_px=(-10, 10), scale=(0.9, 1.1)),
            A.HorizontalFlip(),
            # transforms
            A.RandomCrop(image_size[1], image_size[0]),
            A.PadIfNeeded(padded_image_size[1], padded_image_size[0], border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Normalize(mean=image_mean, std=image_std),
            ToTensorV2()
        ])

    def prepare_splits(self) -> List[List[str]]:
        with open(self._data_root / 'skip_images.txt', 'r') as f:
            skip_image_list = f.read().splitlines()

        sequences_names = sorted([path for path in (self._data_root / 'images').rglob('*_01.jpg') if path.name not in skip_image_list])

        if 'train' in sequences_names or 'test' in sequences_names or 'valid' in sequences_names:
            sequences_names = sorted([cat.name + '/' + sequence_path.name for cat in self._data_root.glob('*')
                                      for sequence_path in cat.glob('*')
                                      if not sequence_path.name.startswith('.')])

        splits = self.partition_sequences(sequences_names, self._number_of_splits)
        return splits

    @staticmethod
    def partition_sequences(sequences: List[str], n: int) -> List[List[str]]:
        sequences = sequences.copy()
        Random(42).shuffle(sequences)
        return [sequences[i::n] for i in range(n)]

    @staticmethod
    def get_train_valid_test(splits: List[List[str]], current_split: int):
        splits = deque(splits)
        splits.rotate(current_split)
        splits = list(splits)

        return list(itertools.chain.from_iterable(splits[:-2])), splits[-2], splits[-1]

    def setup(self, stage: Optional[str] = None):
        splits = self.prepare_splits()

        train_split, valid_split, test_split = self.get_train_valid_test(splits, self._current_split)
       
        self._train_dataset = self._dataset(
            data_root=self._data_root,
            images_list=train_split,
            augmentations=self._augmentations if self._augment else self._transforms,
        )

        self._valid_dataset = self._dataset(
            data_root=self._data_root,
            images_list=valid_split,
            augmentations=self._transforms,
        )

        self._train_dataset = self._dataset(
            data_root=self._data_root,
            images_list=test_split,
            augmentations=self._transforms,
        )

    def train_dataloader(self):
        return DataLoader(
            self._train_dataset, batch_size=self._batch_size, num_workers=self._workers,
            pin_memory=True, drop_last=True, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self._valid_dataset, batch_size=self._batch_size, num_workers=self._workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self._test_dataset, batch_size=self._batch_size, num_workers=self._workers,
            pin_memory=True
        )

