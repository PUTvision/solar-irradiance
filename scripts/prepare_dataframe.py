from datetime import datetime
from pathlib import Path

import click
import pandas as pd


@click.command()
@click.option('--data-root', type=click.Path(exists=True, path_type=Path), required=True)
@click.option('--output-path', type=click.Path(path_type=Path), required=True)
def prepare_dataframe(data_root: Path, output_path: Path):
    df = pd.read_csv(data_root / 'irradiance.csv', parse_dates={'datetime': ['date']},
                     index_col='datetime', date_format='%Y%m%d_%H%M%S')
    existing_images = {datetime.strptime(image_path.stem, '%Y%m%d_%H%M%S') for image_path in data_root.rglob('*.jpg')}
    print(f'Existing images: {len(existing_images)}')
    df = df.loc[df.index.isin(existing_images)]
    df['image_name'] = df.index.strftime('%Y%m%d_%H%M%S') + '.jpg'
    df.to_csv(output_path)


if __name__ == '__main__':
    prepare_dataframe()
