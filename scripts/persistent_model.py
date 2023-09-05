import pandas as pd
import click
import torch
from torchmetrics.functional.regression import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from tqdm import tqdm


MAX_IRRADIANCE = 1466.0     # max irradiance in the dataset
IRRADIANCE_MEAN = 412.7034  # mean irradiance in the dataset
IRRADIANCE_STD = 295.5182   # std irradiance in the dataset


@click.command()
@click.option("--eval_periods_path", help="Data frame with evaluation periods", type=click.Path(exists=True, file_okay=True), default="data/Eval/eval_periods.pickle")
def evaluate_persistent_model(eval_periods_path):
    with open(eval_periods_path, "rb") as f:
        eval_periods = pd.read_pickle(f)

    target = []
    preds = []

    for p in tqdm(eval_periods):
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
    print(f"MAE [W/m^2]: {mae*MAX_IRRADIANCE:.4f}")
    print(f"MSE (normalized): {mse:.4f}")
    print(f"RMSE [W/m^2]: {rmse*MAX_IRRADIANCE:.4f}")


if __name__ == '__main__':
    evaluate_persistent_model()
