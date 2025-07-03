from pathlib import Path

import click
import pandas as pd
import pytz
from tqdm import tqdm


@click.command()
@click.option("--data-raw", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output-path", type=click.Path(path_type=Path), required=True)
def convert_timestamps(data_raw: Path, output_path: Path):
    output_path.mkdir(exist_ok=True, parents=True)

    us_pacific = pytz.timezone("US/Pacific")
    utc = pytz.utc

    for image_path in tqdm(sorted(data_raw.rglob("*.jpg"))):
        filename = image_path.name
        date = pd.to_datetime(filename.replace(".jpg", ""), format="%Y%m%d_%H%M%S").round("1min")
        utc_date = utc.localize(date)
        us_pacific_date = utc_date.astimezone(us_pacific)
        filename = us_pacific_date.strftime("%Y%m%d_%H%M%S") + ".jpg"

        image_path.rename(output_path / filename)


if __name__ == "__main__":
    convert_timestamps()
