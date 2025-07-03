from pathlib import Path

import click
import pandas as pd
import torch
from torchmetrics.functional.regression import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from tqdm import tqdm

MAX_IRRADIANCE = 1466.0  # max irradiance in the dataset


@click.command()
@click.option(
    "--periods-path",
    help="Data frame with evaluation periods",
    type=click.Path(exists=True, file_okay=True, path_type=Path),
    default="data/prepared/periods.pickle",
)
def evaluate_persistent_model(periods_path: Path):
    with periods_path.open("rb") as f:
        periods = pd.read_pickle(f)

    test_periods = list(filter(lambda p: p["history"][-1]["image_name"].startswith("2014"), periods))

    target = []
    preds = []

    for p in tqdm(test_periods):
        source_irradiances = []

        for history_item in p["history"]:
            irradiance = history_item["irradiance"] / MAX_IRRADIANCE
            source_irradiances.append(irradiance)

        target_irradiance = p["target_irradiance"] / MAX_IRRADIANCE

        target.append(target_irradiance)
        preds.append(source_irradiances[-1])

    preds = torch.as_tensor(preds)
    target = torch.as_tensor(target)

    mape = mean_absolute_percentage_error(preds, target)
    mae = mean_absolute_error(preds, target)
    mse = mean_squared_error(preds, target)
    rmse = mean_squared_error(preds, target, squared=False)
    print(f"MAPE [%]: {mape*100:.2f}")
    print(f"MAE [W/m^2]: {mae*MAX_IRRADIANCE}")
    print(f"MSE (normalized): {mse}")
    print(f"RMSE [W/m^2]: {rmse*MAX_IRRADIANCE}")


if __name__ == "__main__":
    evaluate_persistent_model()
