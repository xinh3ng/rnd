#!/usr/bin/env bash

# Set the following env variables whenever needed
version=  # package version
# export USER=  # xinheng
export AWS_ACCESS_KEY=  #
export AWS_SECRET_KEY=  #
export DBRICKS_TOKEN=  #

export BRANCH_KEY=
export BRANCH_SECRET=

# Derived variables
whl_file=rnd-${version}-py3-none-any.whl

docker_image_tag=retrieval:${version}

# Build docker image file
docker build -t ${docker_image_tag} .

# Create a wheel package file and install the package
pip uninstall rnd -y; rm -f dist/; python setup.py sdist bdist_wheel; pip install -I ./dist/${whl_file}

# Install on a databricks cluster
dbfs cp dist/${whl_file} dbfs:/FileStore/xinheng/packages/${whl_file} --overwrite
# dbfs cp dbfs:/FileStore/xinheng/packages/${whl_file} dbfs:/FileStore/prod/packages/${whl_file} --overwrite
