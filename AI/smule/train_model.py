# -*- coding: utf-8 -*-
"""

Procedure:
  $ source venv/bin/activate (python 3.6)
  $ source scripts/setenv.sh (Set environment variables like PYTHONPATH
  $ python train_model.py --src_data_file=/share/data/smule/incidences_piano.tsv --nfolds=5 --total_size=0.5 --val_size=0.2

Useful links:
  https://vinta.ws/code/build-a-recommender-system-with-pyspark-implicit-als.html

"""
from pdb import set_trace as debug
import os
import pandas as pd
from pyspark.sql import SparkSession
from smule.config import model_config as MC
from smule.utils.generic import create_logger, to_spark
from smule.utils import data_utils as DU
from smule.models.models import train_validate_mf_als

logger = create_logger(__name__, level="info")


def main(src_data_file, nfolds, total_size, val_size):
    """Main function
    
    Args:
        nfolds: Number of folds used in cross validation
    """
    raw = DU.load_process_raw(src_data_file)
    mf_data = DU.gen_mf_data(raw, verbose=1)
    logger.info("Cross validation started. It will take several minutes to hours")
    perf_scores = pd.DataFrame()
    for n in range(nfolds):  # NB xheng: this for loop can covert to multiprocessing
        logger.info("\n####################\nStarting fold idx: %d\n####################" % n)

        train_data, val_data = DU.gen_train_val_data(mf_data, total_size=total_size, val_size=val_size)
        spark = SparkSession.builder.appName("cross validation").getOrCreate()
        spark.sparkContext.setLogLevel("WARN")  # ERROR, WARN
        spark.sparkContext.setCheckpointDir("./results/checkpoint")  # to avoid stack overflow
        logger.info("Successfully init a spark session")
        train_data = to_spark(spark, train_data, infer_schema=True)
        val_data = to_spark(spark, val_data, infer_schema=True)
        logger.info("Successfully converted to spark df")

        # Go through each param set on the SAME data set
        for params in MC.model_param_grid:
            perf_row = train_validate_mf_als(train_data, val_data, params)
            perf_row["fold_idx"] = n
            perf_scores = pd.concat([perf_scores, perf_row])
            logger.info("Showing current perf_scores:\n%s" % perf_scores.to_string(line_width=144))
        logger.info("Successfully ran fold idx: %d" % n)
    logger.info("Successfully completed entire cross validation")

    avg_perf = perf_scores.groupby(["params"]).mean().reset_index().drop("fold_idx", axis=1)
    logger.info("Showing avg_perf:\n%s" % avg_perf.to_string(line_width=144))
    return avg_perf


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--src_data_file", default="{}/data/smule/incidences_piano.tsv".format(os.environ["HOME"]))
    parser.add_argument("--nfolds", type=int, default=5, help="how many times to repeat the train-test process")
    parser.add_argument(
        "--total_size",
        type=float,
        default=0.1,
        help="Percentage of total data belongs to entire training process. Small size speeds up training",
    )
    parser.add_argument("--val_size", type=float, default=0.1, help="Percentage of data belongs to validation")

    args = parser.parse_args()
    main(**vars(args))
    logger.info("ALL DONE\n")
