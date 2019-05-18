# -*- coding: utf-8 -*-
"""Generic utility functions
"""
from __future__ import print_function
from pdb import set_trace as debug
import logging
import pyspark.sql.types as T


def retry(n_try, sleep=1, *exception_types):
    """Retry a function several times
    :param n_try: Number of trials
    :param sleep: Delay in seconds. Default=1
    :param exceptionType: Types of exceptions
    """
    def try_fn(func, *args, **kwargs):
        for n in range(n_try):
            try:
                return func(*args, **kwargs)
            except exception_types or Exception as e:
                print('Trial {n} failed with exception: {e} .\nTrying again after a {sleep} second sleep'.format(
                    n=n, e=str(e), sleep=sleep))
                time.sleep(sleep)
    return try_fn


def create_logger(name,
                  level='info',
                  fmt='%(asctime)s %(levelname)s %(name)s: %(message)s',
                  datefmt='%y-%m-%d %H:%M:%S',
                  add_console_handler=True,
                  add_file_handler=False,
                  logfile='/tmp/tmp.log'):
    """Create a formatted logger at module level

    :param fmt: Format of the log message
    :param datefmt: Datetime format of the log message
    :example:
    logger = create_logger(__name__, level='info')
    logger.info('Hello world')
    """
    level = {
        'debug': logging.DEBUG, 'info': logging.INFO,
        'warn': logging.WARN, 'error': logging.ERROR
    }[level]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logFmt = logging.Formatter(fmt=fmt, datefmt=datefmt)

    if add_console_handler:  # Print on console
        ch = logging.StreamHandler()
        ch.setFormatter(logFmt)
        logger.addHandler(ch)

    if add_file_handler:  # Print in a log file
        th = logging.RotatingFileHandler(logfile, backupCount=5)
        th.doRollover()
        th.setFormatter(logFmt)
        logger.addHandler(th)

    return logger


def to_spark(spark, data, infer_schema=False, schema=None):
    """Convert Pandas to Spark data frame

    Args:
        spark: Spark session
        data: Pandas data frame to be converted
        infer_schema: Wheter to infer data type, Default is False
        schema:
    """
    def find_type(x):
        if x.dtype in ['object', 'str']:
            return T.StringType()
        elif x.dtype == 'int':
            return T.IntegerType()
        elif x.dtype == 'float':
            return T.FloatType()
        elif x.dtype == 'bool':
            return T.BooleanType()
        raise TypeError('%s type is unknown' %(x.dtype))

    if infer_schema:
        schema = T.StructType([T.StructField(str(col), find_type(data[col])) for col in data.columns])
    sdf = spark.createDataFrame(data, schema=schema)
    return sdf
