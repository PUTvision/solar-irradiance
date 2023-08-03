import pickle
from pathlib import Path

import click
import pandas as pd
from tqdm import tqdm


@click.command()
@click.option('--cleaned-dataframe-path', type=click.Path(exists=True, path_type=Path), required=True)
@click.option('--history-size', type=int, required=True)
@click.option('--time-window', type=int, required=True)
@click.option('--output-path', type=click.Path(path_type=Path), required=True)
def export_folsom(cleaned_dataframe_path: Path, history_size: int, time_window: int, output_path: Path):
    time_window += history_size

    periods = []

    df = pd.read_csv(cleaned_dataframe_path, parse_dates=['datetime'],
                     index_col='datetime')

    print(f'Min irradiance: {df["irradiance"].min()}')
    print(f'Max irradiance: {df["irradiance"].max()}')
    print(f'Mean irradiance: {df["irradiance"].mean()}')
    print(f'Irradiance std: {df["irradiance"].std()}')

    for index, row in tqdm(df.iterrows(), total=len(df)):
        history = df.loc[index:index + pd.Timedelta(minutes=history_size - 1)]
        if len(history) < history_size:
            continue

        target_moment_index = df.index[df.index.get_indexer([index + pd.Timedelta(minutes=time_window - 1)],
                                                            method='nearest')]
        time_difference = target_moment_index - index
        if pd.Timedelta(minutes=time_window - 1, seconds=30) < time_difference < pd.Timedelta(
                minutes=time_window, seconds=30):
            periods.append({
                'history': [
                    {
                        'image_name': row['image_name'],
                        'irradiance': row['irradiance']
                    } for index, row in history.iterrows()
                ],
                'target_irradiance': df.loc[target_moment_index]['irradiance'],
            })

    print(f'Number of periods: {len(periods)}')
    with output_path.open('wb') as f:
        pickle.dump(periods, f)


if __name__ == '__main__':
    export_folsom()
