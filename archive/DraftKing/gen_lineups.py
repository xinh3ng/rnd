#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate random lineups

Created on Sat Jun 10 07:02:40 2017
@author: xin.heng
"""
from pdb import set_trace as debug
from pydsutils.generic import createLogger

logger = createLogger(__name__, level="info")


def genLineupMainFn(sport):
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sport", type=str, default="MLB")
    args = parser.parse_args()

    lineups = genLineupMainFn(args.sport)
    logger.info("\nALL DONE!\n")
