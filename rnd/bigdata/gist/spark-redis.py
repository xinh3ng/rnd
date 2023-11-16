"""

# Links
https://redislabs.com/blog/getting-started-redis-apache-spark-python/

https://www.infoq.com/articles/data-processing-redis-spark-streaming/


"""
import json
import os
from pyspark.sql import SparkSession


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
    print("spark session's setting: {}".format(str(spark.sparkContext.getConf().getAll())))

    data = spark.read.csv(
        "{}/data/pantheon.tsv".format(os.environ.get("HOME")), sep="\t", quote="", header=True, inferSchema=True
    )
    print(
        "Showing a small data sample\n%s"
        % data.select([key_column, "countryCode", "occupation"])
        .sample(0.1, False)
        .limit(5)
        .toPandas()
        .to_string(line_width=120)
    )

    ################
    # Write
    ################
    print("Saving the data in Redis")
    data.write.format("org.apache.spark.sql.redis").mode("overwrite").option("table", "people").option(
        "key.column", key_column
    ).save()

    ################
    # Read
    ################
    data = (
        spark.read.format("org.apache.spark.sql.redis")
        .option("table", "people")
        .option("key.column", key_column)
        .load()
    ).select([key_column, "countryCode", "occupation"])
    data.show(5)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file",
        default=f"",
    )

    # Parse the cmd line args
    args = vars(parser.parse_args())
    print("Cmd line args:\n{}".format(json.dumps(args, sort_keys=True, indent=4)))

    main(**args)
    print("\nALL DONE!\n")
