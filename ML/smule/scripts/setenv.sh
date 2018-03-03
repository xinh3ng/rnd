#!/usr/bin/env bash
# Define environment variables

export PYSPARK_PYTHON=`which python`
export PYTHONPATH=$(dirname $(pwd))

unameout="$(uname -s)"
case "${unameout}" in
    Linux*)     machine=Linux;;
    Darwin*)    machine=Mac;;
    CYGWIN*)    machine=Cygwin;;
    MINGW*)     machine=MinGw;;
    *)          machine="UNKNOWN:${unameOut}"
esac

if [ $machine = 'Mac' ]; then 
    export SPARK_HOME=/usr/local/Cellar/apache-spark/2.2.1/libexec
    export PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.4-src.zip:$PYTHONPATH

elif [ $machine = 'Linux' ]; then
    export SPARK_HOME=/usr/local/spark
    export PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.6-src.zip:$PYTHONPATH
fi

printf "machine:        $machine\n"
printf "SPARK_HOME:     $SPARK_HOME\n"
printf "PYSPARK_PYTHON: $PYSPARK_PYTHON\n"
printf "PYTHONPATH:     $PYTHONPATH\n"
printf "setenv.sh - successfully completed\n\n"
