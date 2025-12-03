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
@click.option(
    "--clear-sky-path",
    help="Path to clear sky irradiance data",
    type=click.Path(exists=True, file_okay=True, path_type=Path),
    default="data/prepared/clear_sky_ineichen.csv",
)
@click.option(
    "--forecasting-horizon",
    help="Forecasting horizon in minutes",
    type=int,
    default=15,
)
def evaluate_smart_persistence_model(periods_path: Path, clear_sky_path: Path, forecasting_horizon: int):
    # Load clear sky data
    clear_sky_df = pd.read_csv(clear_sky_path)
    clear_sky_df["datetime"] = pd.to_datetime(clear_sky_df["datetime"], utc=False)
    clear_sky_dict = dict(zip(clear_sky_df["datetime"], clear_sky_df["ghi_clear"], strict=False))

    with periods_path.open("rb") as f:
        periods = pd.read_pickle(f)

    test_periods = list(filter(lambda p: p["history"][-1]["image_name"].startswith("2014"), periods))

    target = []
    preds = []
    naive_count = 0

    for p in tqdm(test_periods):
        # Get the last observed irradiance and its clear sky value
        last_history = p["history"][-1]
        last_irradiance = last_history["irradiance"]
        # Extract timestamp from image_name (format: yyyymmdd_HHMMSS.jpg)
        image_name = last_history["image_name"]
        date_time_str = image_name.replace(".jpg", "")
        last_timestamp = pd.to_datetime(date_time_str, format="%Y%m%d_%H%M%S", utc=False)

        # Get clear sky irradiance for last observation
        last_clear_sky = clear_sky_dict.get(last_timestamp)

        # Get clear sky irradiance for target time
        target_timestamp = last_timestamp + pd.Timedelta(minutes=forecasting_horizon)
        target_clear_sky = clear_sky_dict.get(target_timestamp)

        if last_clear_sky is not None and target_clear_sky is not None and last_clear_sky > 0:
            # Calculate clear sky index from last observation
            kc_last = last_irradiance / last_clear_sky
            kc_last = min(max(kc_last, 0.0), 2.0)  # clamp to [0, 2]

            # Predict target irradiance using smart persistence
            # (assume clear sky index persists, multiply by target clear sky)
            pred_irradiance = kc_last * target_clear_sky
        else:
            # Fallback to naive persistence if clear sky data is missing
            pred_irradiance = last_irradiance
            naive_count += 1

        target_irradiance = p["target_irradiance"]

        target.append(target_irradiance)
        preds.append(pred_irradiance)

    print(f"Used naive persistence for {naive_count} out of {len(test_periods)} periods.")

    preds = torch.as_tensor(preds)
    target = torch.as_tensor(target)

    mape = mean_absolute_percentage_error(preds, target)
    mae = mean_absolute_error(preds, target)
    mse = mean_squared_error(preds, target)
    rmse = mean_squared_error(preds, target, squared=False)
    print("--- Smart Persistence Model evaluation ---")
    print(f"Forecasting horizon: {forecasting_horizon} minutes")
    print(f"MAPE [%]: {mape*100:.2f}")
    print(f"MAE [W/m^2]: {mae}")
    print(f"MSE (normalized): {mse}")
    print(f"RMSE [W/m^2]: {rmse}")


if __name__ == "__main__":
    evaluate_smart_persistence_model()
