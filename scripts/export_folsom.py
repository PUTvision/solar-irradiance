import pickle

import pandas as pd
from tqdm import tqdm


def main():
    history_size = 3
    time_window = 5 + history_size

    periods = []

    df = pd.read_csv('/home/dpieczynski/Datasets/Folsom/cleaned_irradiance.csv', parse_dates=['datetime'],
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
    with open('/home/dpieczynski/Datasets/Folsom/periods.pickle', 'wb') as f:
        pickle.dump(periods, f)


if __name__ == '__main__':
    main()
