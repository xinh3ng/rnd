"""

# Link to PyTorch中的 数独 RNN
https://www.toutiao.com/a6640028992575373828/?timestamp=1586213213&app=news_article&group_id=6640028992575373828&req_id=202004070646530100140470310D689AE5
"""
import json
import os
from pyspark.sql import SparkSession
from pypchutils.generic import create_logger

logger = create_logger(__name__)


def main(data_file, key_column: str = "en_curid"):
    spark = (
        SparkSession.builder.appName("spark-redis")
        .master("local[2]")
        .config("spark.executor.memory", "2g")
        .config("spark.driver.memory", "2g")
        .config("spark.jars", ",".join(["./assets/spark-redis_2.11-2.4.3-SNAPSHOT-jar-with-dependencies.jar"]))
        .config("spark.redis.host", "localhost")
        .config("spark.redis.port", "6379")
        # .config("spark.redis.auth", "passwd")
        .config("spark.sql.execution.arrow.enabled", "true")
        .enableHiveSupport()
        .getOrCreate()
    )
    logger.info("spark session's setting: {}".format(str(spark.sparkContext.getConf().getAll())))

    data = spark.read.csv("dbfs:/FileStore/users/xheng/pantheon.tsv", sep="\t", quote="", header=True, inferSchema=True)
    logger.info(
        "\n%s" % data.select([key_column, "countryCode", "occupation"]).limit(5).toPandas().to_string(line_width=120)
    )

    # Same a small data
    data.write.format("org.apache.spark.sql.redis").option("table", "people").option("key.column", key_column).save()
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file", default=f"",
    )

    # Parse the cmd line args
    args = vars(parser.parse_args())
    logger.info("Cmd line args:\n{}".format(json.dumps(args, sort_keys=True, indent=4)))

    main(**args)
    logger.info("ALL DONE!\n")
