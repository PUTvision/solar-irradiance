import pandas as pd
import click
from sklearn.metrics import mean_absolute_percentage_error
from tqdm import tqdm


MAX_IRRADIANCE = 1466.0     # max irradiance in the dataset
IRRADIANCE_MEAN = 412.7034  # mean irradiance in the dataset
IRRADIANCE_STD = 295.5182   # std irradiance in the dataset


@click.command()
@click.option("--eval_periods_path", help="Data frame with evaluation periods", type=click.Path(exists=True, file_okay=True), default="data/Eval/eval_periods.pickle")
def evaluate_persistent_model(eval_periods_path):
    with open(eval_periods_path, "rb") as f:
        eval_periods = pd.read_pickle(f)

    target_irradiances = []
    outputs = []

    for p in tqdm(eval_periods):
        source_irradiances = []

        for history_item in p["history"]:
            irradiance = history_item["irradiance"] / MAX_IRRADIANCE
            source_irradiances.append(irradiance)

        target_irradiance = p["target_irradiance"] / MAX_IRRADIANCE

        target_irradiances.append(target_irradiance)
        outputs.append(source_irradiances[-1])

    mape = mean_absolute_percentage_error(target_irradiances, outputs)
    print(f"MAPE [%]: {mape*100:.2f}")


if __name__ == '__main__':
    evaluate_persistent_model()
