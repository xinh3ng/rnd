"""

# Links
https://www.infoq.com/articles/data-processing-redis-spark-streaming/


"""
import json
import os
from pyspark.sql import SparkSession
from pyspark.sql import types as T, functions as F


def main(data_file: str):
    spark = (
        SparkSession.builder.appName("spark-redis")
        .master("local[*]")
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
    print("spark session's setting: {}".format(str(spark.sparkContext.getConf().getAll())))

    clicks = (
        spark.readStream.format("redis")
        .option("stream.keys", "clicks")
        .schema(T.StructType([T.StructField("asset", T.StringType()), T.StructField("cost", T.LongType())]))
        .load()
    )
    print("clicks: %s" % clicks)
    # clicks.show()

    by_asset = clicks.groupBy("asset").count()
    query = by_asset.writeStream.start()
    query.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file", default=f"",
    )

    # Parse the cmd line args
    args = vars(parser.parse_args())
    print("Cmd line args:\n{}".format(json.dumps(args, sort_keys=True, indent=4)))

    main(**args)
    print("\nALL DONE!\n")
