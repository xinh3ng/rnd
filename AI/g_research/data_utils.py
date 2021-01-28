"""Data utility functions

"""
from pdb import set_trace as debug
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

plt.style.use("ggplot")

from pydsutils.generic import create_logger
from pymlearn.sampling import split_train_validate

from lighthorse_ml.volatility_trend_labeler.data_conf import (
    ClassStrToNum,
    TrendStrToNum,
    VolatilityStrToNum,
    get_class_descr_table,
)

from lighthorse_ml.volatility_trend_labeler.volatility.volatility_utils import gen_volatility_features
from lighthorse_ml.volatility_trend_labeler.trend.trend_utils import gen_trend_features
from lighthorse_ml.volatility_trend_labeler.models.model_conf import ts_col

logger = create_logger(__name__, level="info")


# Util functions that are more generic
def impute_nan(x, lookback=1):
    return x


################################################3


def get_train_validate_data(tickers, data_dir, train_size=0.9, head_or_tail="tail"):
    """Load the entire raw data set and process it. End result will include all candidate features"""

    full_data = pd.DataFrame()
    for ticker in tickers:
        raw_data = process_raw_data(get_raw_train_data(data_dir, ticker=ticker))
        data = gen_features(raw_data)
        # explore_data(data)
        # Append data together
        full_data = pd.concat([full_data, data]).reset_index(drop=True)

    if head_or_tail == "tail":  # Take tail portion of entire data set
        dataset = split_train_validate(full_data, train_size=train_size, shuffle=False)  # MUST not shuffle
        full_train_data, full_val_data = dataset["train"], dataset["validate"]

    elif head_or_tail == "head":  # Take head portion of entire data set
        dataset = split_train_validate(full_data, train_size=(1 - train_size), shuffle=False)  # MUST not shuffle
        full_val_data, full_train_data = dataset["train"], dataset["validate"]

    return full_train_data, full_val_data


def get_data_dir(data_dir):
    """Get source data's directory

    :return:
    """
    src_dir = "%s/true_labels" % data_dir
    return src_dir


def get_raw_train_data(data_dir, ticker, ts_format="%m/%d/%Y"):
    file = "%s/True Label Sample %s.csv" % (get_data_dir(data_dir), ticker)
    data = pd.read_csv(file)
    data["ticker"] = ticker
    data[ts_col] = pd.to_datetime(data[ts_col], format=ts_format)  # add date time
    data["class"] = data["Class"].apply(ClassStrToNum().str2num)  # string to numerical
    data["trend"] = data["Trend"].apply(TrendStrToNum().str2num)
    data["volatility"] = data["Volatility"].apply(VolatilityStrToNum().str2num)

    logger.debug(
        "Class description:\n%s"
        % data[["Class", "Trend", "Volatility"]].drop_duplicates().sort_values(by="Class").to_string(line_width=144)
    )
    return data


def get_raw_test_data(data_dir, ticker, ts_format="%Y-%m-%d"):
    file = "%s/Test Sample %s.csv" % (get_data_dir(data_dir), ticker)
    data = pd.read_csv(file)
    data["ticker"] = ticker
    data[ts_col] = pd.to_datetime(data[ts_col], format=ts_format)  # add date time
    return data


def process_raw_data(data):
    """Process raw train or test data"""
    # Take a log()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        data[col] = impute_nan(data[col], lookback=1)
        data[col.lower()] = np.log(data[col])  # log-price
    return data


def gen_final_class_labels(data):
    """Generate final class labels"""

    data["Volatility"] = data["volatility"].apply(VolatilityStrToNum().num2str)
    data["Trend"] = data["trend"].apply(TrendStrToNum().num2str)
    data = data.merge(
        get_class_descr_table(), how="inner", left_on=["Volatility", "Trend"], right_on=["Volatility", "Trend"]
    )
    return data


def gen_features(data):
    """Generate predictor features

    There are many lagged features. tm1 means "t minus 1"
    """
    data = gen_trend_features(data)
    data = gen_volatility_features(data)
    data.dropna(inplace=True)  # Drop all the NA rows

    logger.debug("After gen_features(), column names: %s\n" % data.columns)
    logger.debug("Data head:\n%s\n" % data.head(20).to_string(line_width=144))
    return data


##############################################
# Util function to explore data
##############################################
def explore_data(data):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.plot(data[ts_col], data["class"])
    for col in ["Close", "ratio_Close_tm1_tm150", "volatility_gk_close_tm10", "volatility_gk_close_tm20"]:
        # 5 is the range of the class label
        ax.plot(data[ts_col], (data[col] - data[col].min()) * 5 / (data[col].max() - data[col].min()))

    ax.legend(loc=2)
    plt.show()
    return
