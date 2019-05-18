#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

https://bitesofcode.wordpress.com/2017/09/12/augmented-reality-with-python-and-opencv-part-1/
"""
from pdb import set_trace as debug
import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pydsutils.generic import create_logger

logger = create_logger(__name__)


def cv2_to_plt(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def find_matches(des_model, des_scene, min_matches=15, verbose=1, kp_model=None, kp_scene=None, model=None, scene=None):
    matches = bf.match(des_model, des_scene)
    matches = sorted(matches, key=lambda x: x.distance)  # Sort in the order of their distance
    if len(matches) < min_matches:
        logger.error("Not enough matches have been found - %d/%d" % (len(matches), MIN_MATCHES))
        sys.exit()

    if verbose > 0:
        # draw first few matches
        scene_matches = cv2.drawMatches(model, kp_model, scene, kp_scene, matches[:MIN_MATCHES], 0, flags=2)
        plt.imshow(scene_matches)
        # cv2.imshow('scene_matches', scene_matches)
        # cv2.waitKey(1000)
    return matches


MIN_MATCHES = 15

scene = cv2.imread("%s/data/card_scene.jpg" % os.environ["HOME"], 0)
model = cv2.imread("%s/data/card_model.jpg" % os.environ["HOME"], 0)

orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # brute foace matcher

kp_model, des_model = orb.detectAndCompute(model, None)  # Compute keypoints and its descriptors
kp_scene, des_scene = orb.detectAndCompute(scene, None)

matches = find_matches(
    des_model, des_scene, min_matches=MIN_MATCHES, kp_model=kp_model, kp_scene=kp_scene, model=model, scene=scene
)

# assuming matches stores the matches found
# differenciate between source points and destination points
src_pts = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)  # compute Homography

# Draw a rectangle that marks the found model in the frame
h, w = model.shape
pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
dst = cv2.perspectiveTransform(pts, M)  # project corners into frame
# connect them with lines
scene_lines = cv2.polylines(scene, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
plt.imshow(scene_lines)

#
logger.info("ALL DONE!\n")
