import click

import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


def _save_datasets(dataset, outdir: Path):
    """Save data set into nice directory structure and write SUCCESS flag."""
    out_dataset = outdir / 'data.parquet.gzip/'

    flag = outdir / '.SUCCESS'

    dataset.to_parquet(str(out_dataset), compression='gzip')

    flag.touch()


@click.command()
@click.option('--in-csv')
@click.option('--out-dir')
def make_datasets(in_csv, out_dir):

    out_dir = Path(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(in_csv, index_col=0)

    # Select initial dataset (removed columns with many missing values)

    features = [
        'country',
        'province',
        'variety',
        'winery',
        'price',
        'description'
    ]

    label = ['points']

    data = data[features + label]

    # Remove duplicates and handle missing values

    data = data[~data.duplicated()]

    data = data.fillna(data.mean())

    data = data.dropna()

    # Derive useful features from descriptions (bag of words and description length)

    data = data.assign(description_length=data['description'].apply(len))

    descriptions = data['description']

    vectorizer = CountVectorizer(
        max_features=500, stop_words='english', max_df=0.5)

    vectorizer.fit(descriptions)

    word_counts = vectorizer.transform(descriptions)

    data[['word_' + s for s in vectorizer.get_feature_names()]
         ] = pd.DataFrame(list(word_counts.toarray()), index=data.index)

    data = data.drop('description', axis=1)

    # Encode remaining categorical features

    data = pd.get_dummies(data)

    # Limit number of features

    X = data.drop(label, axis=1)

    feature_value_counts = X.astype(bool).sum(
        axis=0).sort_values(ascending=False, inplace=False)

    X = X[feature_value_counts.index[:2000]]

    y = data[label]

    # Mark data for training and testing

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=101
    )

    train = X_train.join(y_train)

    test = X_test.join(y_test)

    train['istest'] = np.array(False * len(train), dtype='bool')

    test['istest'] = np.array(True * len(test), dtype='bool')

    # Dump data

    final_dataset = train.append(test)

    _save_datasets(final_dataset, out_dir)


if __name__ == '__main__':

    make_datasets()
