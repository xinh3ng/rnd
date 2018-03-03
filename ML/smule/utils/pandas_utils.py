# -*- coding: utf-8 -*-
from pdb import set_trace as debug
import pandas as pd
from smule.config import data_config as DC
from smule.utils.generic import create_logger

logger = create_logger(__name__)


def random_split_by_user(data, val_size, user_col='user_id', item_col='item_id',
                         rating_col='count', seed=None):
    """Randomly split the sample by user_col.

    Goal is make sure all users that appear in train data are also in validation.
    Args:
        data:
        val_size:
        user_col:
        seed:
    """
    train_size = 1 - val_size
    train_df = data.groupby(user_col).apply(
        lambda x: x[[item_col, rating_col]].sample(frac=train_size, replace=False, random_state=seed))\
        .reset_index()
    train_df = train_df[[user_col, item_col, rating_col]]

    # Create val data as the diff of data - train_df
    merged = data[[user_col, item_col, rating_col]].merge(train_df, indicator=True, how='outer')
    val_df= merged[merged['_merge'] == 'left_only']  # only exist in data but not train_df
    logger.info('Fraction of validation data is: %.2f percent' %(100. * len(val_df) / len(data)))
    return train_df, val_df


def ensure_sparsity(data, sparsity, user_col=DC.user_col,
                    item_col=DC.item_col, rating_col=DC.rating_col):
    num_non_zeros = sum(data[rating_col] > 0)
    num_zeros = int(sparsity / (1. - sparsity) * num_non_zeros)  # number of zeros needed

    non_zeros = data[data[rating_col] > 0].copy()
    zeros_df = data[data[rating_col] == 0].copy()
    zeros_df = zeros_df.sample(n=num_zeros, replace=False)  # Randomly select zeros rows

    newdata = pd.concat([non_zeros, zeros_df]).sort_values([user_col, item_col]).reset_index(drop=True)
    logger.info('After ensure_sparsity, the sparsity is: %.4f pct' %\
                (100. * (1. - newdata[rating_col].sum() / len(newdata))))
    return newdata
