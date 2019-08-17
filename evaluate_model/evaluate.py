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
    """Evaluate model and save html report to disc

    Parameters
    ----------
    model: 
        pickle file that contains model to be evaluated
    in-data: str
        name of the parquet file on local disk that contains test data
    out_dir:
        directory where output html should be saved to.
    name: str
        name of output report when saved
    Returns
    -------
    None
    """
    log = logging.getLogger('evaluate_model')

    log.info('Evaluating model.')

    out_path = Path(out_dir) / f'{name}.html'

    data = pd.read_parquet(in_data)

    test = data[data.istest == True].drop('istest', axis=1)

    label = ['points']

    X = test.drop(label, axis=1)

    y = test[label]

    loaded_model = pickle.load(open(model, 'rb'))

    predictions = loaded_model.predict(X)

    y['predictions'] = predictions

    preds_path = Path(out_dir) / 'predictions.parquet.gzip'

    y.to_parquet(str(preds_path), compression='gzip')

    pweave.weave('report.pmd', doctype="md2html", output=out_path)

    log.info('Success! Report saved to {out_path}')

    flag = Path(out_dir) / '.SUCCESS'

    flag.touch()


if __name__ == '__main__':

    evaluate_model()
