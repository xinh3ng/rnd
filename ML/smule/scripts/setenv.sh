#!/usr/bin/env bash
# Define environment variables

clustername=`echo $HOSTNAME | awk '{print substr($0, 0, 7)}'`

export SPARK_HOME=/usr/local/Cellar/apache-spark/2.2.1/libexec
export PYSPARK_PYTHON=`which python`
export PYTHONPATH=`pwd`
export PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.4-src.zip:$PYTHONPATH

printf "SPARK_HOME:     $SPARK_HOME\n"
printf "PYSPARK_PYTHON: $PYSPARK_PYTHON\n"
printf "PYTHONPATH:     $PYTHONPATH\n"
printf "setenv.sh - successfully completed\n\n"
