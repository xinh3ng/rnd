"""

USAGE
  Invoke virtual environment

"""
from pdb import set_trace as debug
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pydsutils.generic import create_logger
from pymlearn.cv_ts import TimeSeriesSplitter, ts_cross_validate

from g_research.config import ycol, xcols

logger = create_logger(__name__, level="info")


def get_train_data(data_dir):
    raw = pd.read_csv(data_dir + "/train.csv.zip", header=0, index_col=0)
    logger.info("Successfully loaded all train data. No of rows = %d" % len(raw))
    logger.info("No. of stocks = %d" % (len(set(raw["Stock"]))))
    logger.info("No. of unique days = %d" % (len(set(raw["Day"]))))
    return raw


def gen_perf_scores(y_true: np.array, y_pred: np.array) -> pd.DataFrame:
    """Classification performance scores
    Args:
        y_true:
        y_pred:
        average: String of averaging schemes
    """
    # A single row of several performance metrics
    perf_row = pd.DataFrame([{"mae": mean_absolute_error(y_true, y_pred), "mse": mean_squared_error(y_true, y_pred)}])
    return perf_row


class LR(object):
    def __init__(self, xcols, ycol):
        self.xcols = xcols
        self.ycol = ycol
        self.model = LinearRegression()

    def process_data(self, data):
        return data

    def get_ycol(self):
        return self.ycol

    def get_xcols(self):
        return self.xcols

    def fit(self, data):
        X, y = data[self.xcols], data[self.ycol]
        self.model.fit(X, y)
        return self

    def predict(self, data):
        X = data[self.xcols]
        y_pred = self.model.predict(X)
        return y_pred


def backtest_model(train_data, estimator, train_size, n_ahead, verbose=0):
    ts_splitter = TimeSeriesSplitter(train_size=train_size, n_ahead=n_ahead)
    perf_df = ts_cross_validate(train_data, estimator, ts_splitter, perf_score_fn=gen_perf_scores, verbose=1)

    if verbose >= 1:
        logger.info("Performance metrics (ts cv): \n%s" % perf_df.to_string(line_width=120))
    return perf_df


def main(data_dir, train_size=100, n_ahead=2):
    train_data = get_train_data(data_dir=data_dir)
    train_data = train_data[train_data["Stock"] == 942].reset_index(drop=True)

    estimator = LR(xcols=xcols, ycol=ycol)
    logger.info("List of feature columns: %s\n", estimator.get_xcols())

    cv_report = backtest_model(train_data, estimator, train_size, n_ahead=n_ahead, verbose=1)
    return cv_report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="~/data/g_research")
    args = parser.parse_args()

    cv_report = main(**vars(args))
    logger.info("ALL DONE\n")
