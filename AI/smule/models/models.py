# -*- coding: utf-8 -*-
"""
Useful links:
  https://github.com/apache/spark/blob/master/examples/src/main/python/ml/als_example.py
"""
from pdb import set_trace as debug
import pandas as pd
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from smule.config import data_config as DC
from smule.utils.generic import create_logger

logger = create_logger(__name__, level="info")


class MFALSOperator(object):
    def __init__(self, params, user_col=DC.user_col, item_col=DC.item_col, rating_col=DC.rating_col):
        self.params = params
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col

    def fit_model(self, data, verbose=0):
        # rating_col must be float or double
        als = ALS(
            maxIter=self.params["max_iter"],
            regParam=self.params["reg_param"],
            alpha=self.params["alpha"],
            rank=self.params["rank"],
            userCol=self.user_col,
            itemCol=self.item_col,
            ratingCol=self.rating_col,
            implicitPrefs=True,
            nonnegative=False,
            checkpointInterval=10,
            coldStartStrategy="drop",
        )
        assert als.getImplicitPrefs(), "Must be implicit feedback"
        if verbose >= 1:
            logger.info("Showing 10 random rows before fitting")
            data.select([self.user_col, self.item_col, self.rating_col]).sample(False, 0.1).show(10)
        model_fit = als.fit(data)
        if verbose >= 1:
            logger.info("Successfully fit model")
        return model_fit

    def predict(self, model_fit, data):
        return model_fit.transform(data)

    def evaluate_model(self, model_fit, data, metric="rmse", verbose=0):
        evaluator = RegressionEvaluator(metricName=metric, labelCol=self.rating_col, predictionCol="prediction")
        predictions = self.predict(model_fit, data)
        perf_score = evaluator.evaluate(predictions)
        return perf_score


def train_validate_mf_als(train_data, val_data, params):
    """Train the model and validate on validation data

    Args:
        train_data: S
        val_data:
        params: Hyper-parameter set
    """
    model_op = MFALSOperator(params)
    model_fit = model_op.fit_model(train_data, verbose=1)
    perf_score = model_op.evaluate_model(model_fit, val_data, metric="rmse", verbose=1)

    perf_row = pd.DataFrame(
        [{"rmse": perf_score, "params": ",".join(["{}:{}".format(k, v) for k, v in params.items()])}]
    )
    return perf_row
