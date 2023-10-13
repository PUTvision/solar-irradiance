import pickle
from pathlib import Path

import click
import pandas as pd
from tqdm import tqdm


@click.command()
@click.option('--cleaned-dataframe-path', type=click.Path(exists=True, path_type=Path), required=True)
@click.option('--history-size', type=int, default=4)
@click.option('--time-window', type=int, default=15)
@click.option('--output-path', type=click.Path(path_type=Path), required=True)
def export_periods(cleaned_dataframe_path: Path, history_size: int, time_window: int, output_path: Path):
    history_size -= 1   # history samples without sample from t time
    period = time_window // history_size

    periods = []

    df = pd.read_csv(cleaned_dataframe_path, parse_dates=['datetime'],
                     index_col='datetime')

    print(f'Min irradiance: {df["ghi"].min()}')
    print(f'Max irradiance: {df["ghi"].max()}')
    print(f'Mean irradiance: {df["ghi"].mean()}')
    print(f'Irradiance std: {df["ghi"].std()}')

    for t, _ in tqdm(df.iterrows(), total=len(df)):
        t = pd.Timestamp(t)
        t_history = [t - pd.Timedelta(minutes=period * i) for i in range(history_size, 0, -1)]
        t_target = t + pd.Timedelta(minutes=time_window)

        if all(map(lambda x: x in df.index, [*t_history, t_target])):
            periods.append({
                'history': [
                    {
                        'image_name': df.loc[_t]['image_name'],
                        'irradiance': df.loc[_t]['ghi']
                    } for _t in sorted([*t_history, t])
                ],
                'target_irradiance': df.loc[t_target]['ghi'],
            })

    print(f'Number of periods: {len(periods)}')
    with output_path.open('wb') as f:
        pickle.dump(periods, f)


if __name__ == '__main__':
    export_periods()
