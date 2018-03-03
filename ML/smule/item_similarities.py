# -*- coding: utf-8 -*-
"""Run the model with new data or new parameters

Procedure:
  $ source venv/bin/activate (python 3.6)
  $ source scripts/setenv.sh (Set environment variables like PYTHONPATH
  $ python smule/item_similarities.py --total_size=1.0
"""
from pdb import set_trace as debug
import os
import math
from sklearn.model_selection import train_test_split
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from smule.config import data_config as DC
from smule.config import model_config as MC
from smule.utils.generic import create_logger, to_spark
from smule.utils import data_utils as DU
from smule.models.models import MFALSOperator

logger = create_logger(__name__, level='info')


def get_item_factors(model_fit, column='item_id', verbose=0):
    """Get the item latent factors from model_fit

    Args
        model_fit: ALS model fit
        data: Data to be calculated on
        column: Name of the item column
    """
    # distinct() because an item might show in multiple rows in data
    factor_df = model_fit.itemFactors.select(F.col('id').alias(column),
                                             F.col('features').alias('factors'))\
        .distinct().orderBy(F.asc(column))
    if verbose >= 1:
        logger.info('Showing 20 random rows of item factor data:')
        factor_df.sample(False, 0.1).show(20, False)
    return factor_df


def normalized_dot_product(l, r):
    """Calculate dot product and then normalize it

    Formula is: (v1 dot v2) / (||v1||*||v2||). In sklearn, it is metrics.pairwise.cosine_similarity¶
    Args
        l: A list on the left hand side
        r: A list on the right hand side
    """
    # Numerator is v1 dot v2
    numer = sum([i[0] * i[1] for i in zip(l, r)])
    # calculate denominator
    denom = math.sqrt(sum([i*i for i in l])) * math.sqrt(sum([i*i for i in r]))
    return numer / denom


def calc_item_similarity(data, item_col='item_id', factor_col='factors', verbose=0):
    """Calculate item-item cosine similarity

    Args:
        data: Data to be calculated on
        item_col: Item column name
        factor_col:
    """
    # dot product udf
    dot_udf = F.udf(lambda x, y: float(normalized_dot_product(x, y)), T.DoubleType())

    # Generate 2 columns of item IDs (w/o duplicates) and calculate dot product on each row
    df = data.alias('left').join(data.alias('right'),
                                 F.col('left.{}'.format(item_col)) < F.col('right.{}'.format(item_col)))\
        .select(
            F.col('left.{}'.format(item_col)).alias('i'),
            F.col('right.{}'.format(item_col)).alias('j'),
            dot_udf('left.{}'.format(factor_col), 'right.{}'.format(factor_col)).alias('similarity'))\
        .sort('i', 'j')

    if verbose >= 1:
        logger.info("Showing item similarity:")
        df.show(10, False)
    return df


def main(src_data_file, total_size):
    """Main function
    
    Args:
        src_data_file Data file name
        total_size: % of data used for calculation. Small total_size speeds up computation
    """
    params = MC.best_model_params

    raw = DU.load_process_raw(src_data_file)
    full_mf_data = DU.gen_mf_data(raw, verbose=1)
    mf_data, _ = train_test_split(full_mf_data, test_size=1. - total_size)  # Throw away some to speed up training

    spark = SparkSession.builder.appName('refresh model').getOrCreate()
    spark.sparkContext.setCheckpointDir('./results/checkpoint')  # Set checkpoint to avoid stack overflow
    spark.sparkContext.setLogLevel('ERROR')
    logger.info('Successfully init a spark session')
    mf_data = to_spark(spark, mf_data, infer_schema=True)
    logger.info('Successfully converted to spark df')

    model_op = MFALSOperator(params)
    model_fit = model_op.fit_model(mf_data, verbose=1)
    logger.info('Successfully fit model')

    item_factors = get_item_factors(model_fit, column=DC.item_col, verbose=1)
    item_similarities = calc_item_similarity(item_factors, item_col=DC.item_col, factor_col='factors',
                                             verbose=1)
    # Save result as csv
    logger.info('Saving item similarities table to ./results/item_similarities')
    item_similarities.write.csv('./results/item_similarities', mode='overwrite', header=True, sep='|')
    return


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_data_file', default='{}/data/smule/incidences_piano.tsv'.format(os.environ['HOME']))
    parser.add_argument('--total_size', type=float, default=0.1,
        help='Percentage of total data belongs to entire process. Small size speeds up things')

    args = parser.parse_args()
    main(**vars(args))
    logger.info('ALL DONE\n')
