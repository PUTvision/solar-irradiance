from pathlib import Path
import pickle
import shutil

import click
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


@click.command()
@click.option("--data-prepared-path", help="Path to dataset image directory", type=click.Path(exists=True, path_type=Path))
@click.option("--output-path", help="Path to output evaluation periods", type=click.Path(path_type=Path))
def export_eval_periods(data_prepared_path: Path, output_path: Path):
    with Path(data_prepared_path / "periods.pickle").open("rb") as f:
        periods = pd.read_pickle(f)

    test_size = min(1000, len(periods))
    _, eval_periods = train_test_split(periods, test_size=test_size, random_state=42, shuffle=True)

    print(f"Number of periods: {len(eval_periods)}")
    output_path = Path(output_path)
    (output_path / "images").mkdir(parents=True, exist_ok=True)
    with Path(output_path, "eval_periods.pickle").open("wb") as f:
        pickle.dump(eval_periods, f)

    img_names = [p["image_name"] for period in eval_periods for p in period["history"]]
    print(f"Number of unique images: {len(set(img_names))}")

    for img_name in tqdm(set(img_names)):
        src_path = Path(data_prepared_path, "images", img_name)
        dst_path = Path(output_path, "images", img_name)
        shutil.copyfile(src_path, dst_path)


if __name__ == "__main__":
    export_eval_periods()
