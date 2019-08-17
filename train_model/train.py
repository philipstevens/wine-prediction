import click
import logging
import pickle

import numpy as np
import pandas as pd

from sklearn.linear_model import RidgeCV
from pathlib import Path


@click.command()
@click.option('--in-data')
@click.option('--out-dir')
@click.option('--name')
def train_model(in_data, out_dir, name):
    """Train a model and save it to local disk.

    Parameters
    ----------
    in-data: str
        name of the parquet file on local disk that contains training data
    out_dir:
        directory where output model should be saved to.
    name:
        name of output model when saved
    Returns
    -------
    None
    """
    log = logging.getLogger('train-model')

    log.info('Training model.')

    out_path = Path(out_dir) / f'{name}.pickle'

    data = pd.read_parquet(in_data)

    train = data[data.istest == False].drop('istest', axis=1)

    label = ['points']

    X = train.drop(label, axis=1)

    y = train[label]

    model = RidgeCV(alphas=np.logspace(-6, 6, 13))

    model.fit(X, y)

    pickle.dump(model, open(out_path, 'wb'))

    log.info('Success! Model saved to {out_path}')

    flag = Path(out_dir) / '.SUCCESS'

    flag.touch()


if __name__ == '__main__':

    train_model()
