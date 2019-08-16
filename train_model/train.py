import click
import logging
import pickle
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path

@click.command()
@click.option('--in-data')
@click.option('--out-dir')
@click.option('--name')
def train_model(in_data, out_dir, name):

    log = logging.getLogger('train-model')

    log.info('Training model.')

    out_path = Path(out_dir) / f'{name}.pickle'

    data = pd.read_parquet(in_data)

    train = data[data.istest == False].drop('istest', axis=1)

    X = train.iloc[:,:-1]
    y = train.iloc[:,-1]

    model = RandomForestRegressor()
    model.fit(X, y)

    pickle.dump(model, open(out_path, 'wb'))

    log.info('Success! Model saved to {out_path}')

    flag = Path(out_dir) / '.SUCCESS'
    flag.touch()


if __name__ == '__main__':
    train_model()
