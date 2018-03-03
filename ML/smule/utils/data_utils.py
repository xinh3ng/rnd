# -*- coding: utf-8 -*-
from pdb import set_trace as debug
from itertools import product
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from smule.config import data_config as DC
from smule.utils.generic import create_logger
from smule.utils.pandas_utils import random_split_by_user, ensure_sparsity

logger = create_logger(__name__)


def load_process_raw(filepath):
    """Load and process raw data

    Args:
        filepath: File path to raw data
    """
    raw  = pd.read_csv(filepath, sep='\t', header=None,
                       skipinitialspace=True)
    raw.columns = ['user_id', 'item', 'unix_ts']
    raw['date'] = pd.to_datetime(raw['unix_ts'], unit='s')  # Convert to datetime

    # Assign integer item id to music item
    unique_items = pd.unique(raw['item'])
    music_items = pd.DataFrame.from_dict({'item': unique_items,
                                          'item_id': range(len(unique_items))})
    raw = raw.merge(music_items, how='inner', on='item')
    logger.info('Successfully loaded raw data and add proper index and datetime columns')
    logger.info("No. of users: %d, No. of items: %d" % (len(pd.unique(raw[DC.user_col])),
                                                        len(pd.unique(raw[DC.item_col]))))
    if raw.isnull().values.any():
        raise ValueError('raw has NaN, which is not allowed')
    return raw


def gen_mf_data(data, user_col=DC.user_col,
                item_col=DC.item_col,
                verbose=0):
    """Generate data useful for Spark ML's  matrix factorization (mf)
    Args
        data: Raw behavior data about who used which item at what time
    """
    newdata  = data.groupby([user_col, item_col]).size().reset_index()
    newdata.rename(columns={0: 'count'}, inplace=True)  # rename count column to count
    newdata['count'] = newdata['count'].astype(float)
    newdata['log_count'] = np.log(1.0 + newdata['count'])

    if verbose >= 1:
        summary_stats(data, 'full MF data')
        logger.info('Summary stats on newdata:\n%s' %
                    newdata.describe().to_string())
    return newdata


def summary_stats(data, src_name=''):
    logger.info('%s: no. of users who repeat one item twice on more: %d' % (src_name,
                sum(data[[DC.user_col, DC.item_col]].duplicated())))
    logger.info('%s: summary stats on items per user:\n%s' % (src_name,
                data.groupby([DC.user_col]).size().describe().to_string()))
    logger.info('%s: summary stats on users per item:\n%s' % (src_name,
                data.groupby([DC.item_col]).size().describe().to_string()))
    return


def gen_train_val_data(data, total_size, val_size,
                       user_col=DC.user_col, item_col=DC.item_col, rating_col=DC.rating_col,
                       sparsity=0.99):
    """Get training and validation data sets
    Args:
        total_size: Shrink data by this factor. It speeds up calculation
        val_size: % of data that belongs to validation
    """
    assert total_size > 0 and total_size <= 1
    assert val_size >= 0 and val_size < 1
    data = data.copy()
    data, _ = train_test_split(data, test_size=1.-total_size)  # Throw away some to speed up training

    # Split train and validation data
    train_data, val_data = random_split_by_user(data, val_size, user_col=user_col)
    summary_stats(train_data, 'train_data')
    summary_stats(val_data, 'val_data')
    if val_size > 0:
        logger.info('Start to generate validation data...')
        # Validation data set has 0's and 1's only
        val_data[rating_col] = 1.0  # validation data's count column must be all 1's
        val_data = add_zero_observations(val_data, user_col, item_col,
                                         rating_col)  # Add 0 observations
        val_data = ensure_sparsity(val_data, sparsity=sparsity)
        logger.info('Successfully gen train and val data. Validation set has %d rows' % len(val_data))
    return train_data, val_data


def add_zero_observations(data, user_col, item_col, rating_col):
    """data has no 0 observations, thus adding them
    """
    data = data.copy(deep=True)

    # Find combinations of all users and items
    newdata = pd.DataFrame(list(product(set(data[user_col]), set(data[item_col]))),
                           columns=[user_col, item_col])
    newdata = newdata.merge(data[[user_col, item_col, rating_col]], how='left', on=[user_col, item_col])\
        .fillna(0.0)
    logger.info('After add_zero_observations, sparsity of data: %.4f pct' %\
                (100. * (1. -  newdata[rating_col].sum() / len(newdata))))
    return newdata
