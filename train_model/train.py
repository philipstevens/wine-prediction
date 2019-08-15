import pandas as pd 
import pickle
from sklearn.ensemble import RandomForestRegressor

def _save_model(model, outdir: Path):
    """Save model into output directory and write SUCCESS flag."""
    out_model = outdir / 'model.save'
    flag = outdir / '.SUCCESS'

    pickle.dump(model, open(out_model, 'wb'))

    flag.touch()

@click.command()
@click.option('--in-data')
@click.option('--out-dir')
def train_model(in_data, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(in_data)

	X = train.iloc[:,:-1]

	y = train.iloc[:,-1]

	model = RandomForestRegressor()
	model.fit(X, y)

	_save_model(model, out_dir)



if __name__ == '__main__':
    train_model()
