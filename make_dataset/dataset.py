import click
import dask.dataframe as dd
import numpy as np
import pandas as pd 
from distributed import Client
from pathlib import Path
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack

def _save_datasets(train, test, outdir: Path):
    """Save data sets into nice directory structure and write SUCCESS flag."""
    out_train = outdir / 'train.parquet/'
    out_test = outdir / 'test.parquet/'
    flag = outdir / '.SUCCESS'

    train.to_parquet(str(out_train))
    test.to_parquet(str(out_test))

    flag.touch()


@click.command()
@click.option('--in-csv')
@click.option('--out-dir')
def make_datasets(in_csv, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    init_data = pd.read_csv(in_csv, index_col= 0)

    selected_data = init_data[['country', 'description', 'points', 'price', 
        'province', 'title', 'variety','winery']]

    cat_features = [
        'country',
        'province',
        'variety',
        'winery',
    ]

    num_features = [
        'price',
        'description_length'
    ]

    labels = ['points']

    deduped_data = selected_data[~selected_data.duplicated()]

    data = deduped_data.dropna()

    X = data[num_features]
    y = data[labels]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=101
    )

    train = hstack([X_train, y_train])
    test = hstack([X_test, y_test])

    _save_datasets(train, test, out_dir)


if __name__ == '__main__':
    make_datasets()
