import click
import logging
import pickle
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

    out_path = Path(out_dir) / f'{name}.pdf'

    data = pd.read_parquet(in_data)

    test = data[data.istest == True].drop('istest', axis=1)

    X = test.iloc[:,:-1]
    y = test.iloc[:,-1]

    loaded_model = pickle.load(open(model, 'rb'))

    predictions = loaded_model.predict(X)

    log.info("Results of random regressor on price and description length only:" )
    log.info("explained_variance_score: ", metrics.explained_variance_score(y, predictions)) #Explained variance regression score function
    log.info("mean absolute error: ", metrics.mean_absolute_error(y, predictions)) #Mean absolute error regression loss
    log.info("mean squared error: ", metrics.mean_squared_error(y, predictions)) #Mean squared error regression loss
    log.info("mean squared log error: ", metrics.mean_squared_log_error(y, predictions)) #Mean squared logarithmic error regression loss
    log.info("median absolute error: ", metrics.median_absolute_error(y, predictions)) #Median absolute error regression loss
    log.info("R2 score: ", metrics.r2_score(y, predictions)) #R^2 (coefficient of determination) regression score function.


    log.info('Success! Report saved to {out_path}')

    flag = Path(out_dir) / '.SUCCESS'
    flag.touch()


if __name__ == '__main__':
    evaluate_model()