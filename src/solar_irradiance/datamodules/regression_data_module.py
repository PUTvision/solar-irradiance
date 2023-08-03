import itertools
from collections import deque
from pathlib import Path
from random import Random
from typing import Optional, List, Tuple

import albumentations as A
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from solar_irradiance.datamodules.datasets.folsom_dataset import FolsomDataset
from solar_irradiance.utils import utils

log = utils.get_logger(__name__)


class RegressionDataModule(LightningDataModule):
    def __init__(
            self,
            root_data_path: Path,
            augment: bool,
            image_size: Tuple[int, int],
            image_mean: Tuple[float, float, float],
            image_std: Tuple[float, float, float],
            batch_size: int,
            workers: int,
            number_of_splits: int,
            current_split: int,
            sun_mask: bool,
            seed: int,
    ):
        super().__init__()

        self._data_root = Path(root_data_path)
        self._dataset_name = self._data_root.name
        self._augment = augment
        self._batch_size = batch_size
        self._workers = workers
        self._number_of_splits = number_of_splits
        self._current_split = current_split
        self._sun_mask = sun_mask
        self._seed = seed

        if self._dataset_name == 'Folsom':
            self._dataset = FolsomDataset
        else:
            raise ValueError(f'Dataset "{self._dataset_name}" not supported.')

        log.info(f'Using {self._dataset_name} dataset and data from {self._data_root} directory.')

        self._train_dataset = None
        self._valid_dataset = None
        self._test_dataset = None

        self._transforms = A.Compose([
            A.CenterCrop(image_size[1], image_size[0]),
            A.Normalize(mean=image_mean, std=image_std),
        ])

        self._augmentations = A.Compose([
            # geometry augmentations
            A.Affine(rotate=(-10, 10), translate_px=(-10, 10), scale=(0.9, 1.1)),
            A.HorizontalFlip(),
            # transforms
            A.RandomCrop(image_size[1], image_size[0]),
            A.Normalize(mean=image_mean, std=image_std),
        ])

    def prepare_splits(self) -> List[List[str]]:
        with open(self._data_root / 'skip_images.txt', 'r') as f:
            skip_image_list = f.read().splitlines()

        sequences_names = sorted(
            [path for path in (self._data_root / 'images/2015').rglob('*.jpg') if path.name not in skip_image_list])

        if 'train' in sequences_names or 'test' in sequences_names or 'valid' in sequences_names:
            sequences_names = sorted([cat.name + '/' + sequence_path.name for cat in self._data_root.glob('*')
                                      for sequence_path in cat.glob('*')
                                      if not sequence_path.name.startswith('.')])

        splits = self.partition_sequences(sequences_names, self._number_of_splits, self._seed)
        return splits

    @staticmethod
    def partition_sequences(sequences: List[str], n: int, seed: int) -> List[List[str]]:
        sequences = sequences.copy()
        Random(seed).shuffle(sequences)
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

        log.info(f'Training samples: {len(train_split)}')
        log.info(f'Validation samples: {len(valid_split)}')
        log.info(f'Test samples: {len(test_split)}')

        self._train_dataset = self._dataset(
            data_root=self._data_root,
            images_list=train_split,
            augmentations=self._augmentations if self._augment else self._transforms,
            sun_mask=self._sun_mask,
        )

        self._valid_dataset = self._dataset(
            data_root=self._data_root,
            images_list=valid_split,
            augmentations=self._transforms,
            sun_mask=self._sun_mask,
        )

        self._test_dataset = self._dataset(
            data_root=self._data_root,
            images_list=test_split,
            augmentations=self._transforms,
            sun_mask=self._sun_mask,
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
