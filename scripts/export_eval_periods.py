import pickle
import shutil
from pathlib import Path

import click
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


@click.command()
@click.option('--periods_path', help='Path to data frame with periods', type=click.Path(exists=True, file_okay=True))
@click.option('--data_root', help='Path to dataset image directory', type=click.Path(exists=True, dir_okay=True))
@click.option('--output_path', help='Path to output evaluation periods', type=click.Path())
def export_eval_periods(periods_path, data_root, output_path):
    with Path(periods_path).open('rb') as f:
        periods = pd.read_pickle(f)

    _, val_periods = train_test_split(periods, test_size=0.2, random_state=42, shuffle=True)
    _, test_periods = train_test_split(val_periods, test_size=0.5, random_state=42, shuffle=True)
    _, eval_periods = train_test_split(test_periods, test_size=0.1, random_state=42, shuffle=False)

    print(f'Number of periods: {len(eval_periods)}')
    output_path = Path(output_path)
    (output_path / 'images').mkdir(parents=True, exist_ok=True)
    with open(output_path / 'eval_periods.pickle', 'wb') as f:
        pickle.dump(eval_periods, f)

    img_names = [p['image_name'] for period in eval_periods for p in period['history']]
    print(f'Number of unique images: {len(set(img_names))}')

    for img_name in tqdm(set(img_names)):
        src_path = Path(data_root, 'images', img_name)
        dst_path = Path(output_path, 'images', img_name)
        shutil.copyfile(src_path, dst_path)


if __name__ == '__main__':
    export_eval_periods()
