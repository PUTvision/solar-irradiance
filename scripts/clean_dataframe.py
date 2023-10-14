from pathlib import Path

import click
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


@click.command()
@click.option('--data-raw', type=click.Path(exists=True, path_type=Path), required=True)
@click.option('--images-path', type=click.Path(exists=True, path_type=Path), required=True)
@click.option('--output-path', type=click.Path(path_type=Path), required=True)
def clean_dataframe(data_raw: Path, images_path: Path, output_path: Path):
    df = pd.read_csv(
        data_raw / 'Folsom_irradiance.csv',
        parse_dates={'datetime': ['timeStamp']},
        index_col='datetime',
        date_format='%Y-%m-%d %H:%M:%S',
    ).tz_localize('UTC').tz_convert('US/Pacific')
    existing_images = {
        pd.to_datetime(image_path.stem, format='%Y%m%d_%H%M%S').tz_localize('US/Pacific') for image_path in images_path.rglob('*.jpg')
    }
    print(f'Existing images: {len(existing_images)}')

    filtered_images = set()
    for image_name in tqdm(existing_images):
        image_path = Path(images_path, image_name.strftime('%Y%m%d_%H%M%S') + '.jpg')
        try:
            _ = np.asarray(Image.open(image_path))
        except OSError:
            print(f'Truncated image: {image_path}')
            continue

        filtered_images.add(image_name)

    print(f'Filtered images: {len(filtered_images)}')
    df = df.loc[df.index.isin(filtered_images)]
    df['image_name'] = df.index.strftime('%Y%m%d_%H%M%S') + '.jpg'
    print(f'Data samples: {len(df)}')
    df.to_csv(output_path)


if __name__ == '__main__':
    clean_dataframe()
