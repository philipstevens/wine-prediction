import click
import logging
import pickle
import pweave
import pandas as pd
import sklearn.metrics as metrics
from pathlib import Path


@click.command()
@click.option('--model')
@click.option('--in-data')
@click.option('--out-dir')
@click.option('--name')
def evaluate_model(model, in_data, out_dir, name):

    log = logging.getLogger('evaluate_model')

    log.info('Evaluating model.')

    out_path = Path(out_dir) / f'{name}.html'

    data = pd.read_parquet(in_data)

    test = data[data.istest == True].drop('istest', axis=1)

    X = test.iloc[:, :-1]
    y = test.iloc[:, -1]

    loaded_model = pickle.load(open(model, 'rb'))

    predictions = loaded_model.predict(X)

    frame = {'labels': y, 'predictions': predictions}

    preds = pd.DataFrame(frame)

    preds_path = Path(out_dir) / 'predictions.parquet.gzip'

    preds.to_parquet(str(preds_path), compression='gzip')

    pweave.weave('report.pmd', doctype="md2html", output=out_path)

    log.info('Success! Report saved to {out_path}')

    flag = Path(out_dir) / '.SUCCESS'
    flag.touch()


if __name__ == '__main__':
    evaluate_model()
