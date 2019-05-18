# -*- coding: utf-8 -*-
from pdb import set_trace as debug
from smule.utils.generic import create_logger

logger = create_logger(__name__)


def random_split_by_user(spark, data, val_size, user_col="user_id", seed=None):
    """Randomly split the sammple by user

    Args:
        spark:
        data:
        val_size:
        user_col:
        seed:
    """
    train_size = 1 - val_size
    fractions = {row[user_col]: train_size for row in data.select(user_col).distinct().collect()}

    train_df = data.sampleBy(user_col, fractions, seed)
    test_rdd = train_df.rdd.subtract(train_df.rdd)
    val_df = spark.createDataFrame(test_rdd, train_df.schema)
    return train_df, val_df
