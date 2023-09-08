from pathlib import Path

import click
import pandas as pd
import pytz
from tqdm import tqdm


@click.command()
@click.argument('data-root', type=click.Path(exists=True, path_type=Path))
def prepare_images(data_root: Path):
    output_dir = data_root / 'images'
    output_dir.mkdir(exist_ok=True)

    us_pacific = pytz.timezone('US/Pacific')
    utc = pytz.utc

    for image_path in tqdm(sorted(data_root.rglob('*.jpg'))):
        filename = image_path.name
        date = pd.to_datetime(filename.replace('.jpg', ''), format='%Y%m%d_%H%M%S').round('1min')
        utc_date= utc.localize(date)
        us_pacific_date = utc_date.astimezone(us_pacific)
        filename = us_pacific_date.strftime('%Y%m%d_%H%M%S') + '.jpg'

        image_path.rename(output_dir / filename)


if __name__ == '__main__':
    prepare_images()
