from pathlib import Path
import click
import pandas as pd
import pytz
from tqdm import tqdm


@click.command()
@click.option('--data-root', type=click.Path(exists=True, path_type=Path), required=True)
def convert_timestamps(data_root: Path):
    output_dir = data_root / 'images'
    output_dir.mkdir(exist_ok=True)

    us_pacific = pytz.timezone('US/Pacific')
    utc = pytz.utc

    image_list = sorted(data_root.rglob('*.jpg'))

    for image_path in tqdm(image_list):
        filename = image_path.name
        date = pd.to_datetime(filename.replace('.jpg', ''), format='%Y%m%d_%H%M%S').round('1min')
        utc_date = utc.localize(date)
        us_pacific_date = utc_date.astimezone(us_pacific)
        filename = us_pacific_date.strftime('%Y%m%d_%H%M%S') + '.jpg'

        image_path.rename(output_dir / filename)

    print(f'Images: {len(image_list)}')


if __name__ == '__main__':
    convert_timestamps()
