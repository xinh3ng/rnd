"""
"""
from pdb import set_trace as debug
import time
import cv2

from pydsutils.generic import create_logger


def imshow(title, image):
    # show the frame and record if the user presses a key
    cv2.imshow("Security Feed", image)
    # cv2.imshow("Thresh", thresh)
    # cv2.imshow("Frame Delta", frame_delta)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):  # if the `q` key is pressed, break from the lop
        return {"quit": True}
    return {"quit": False}


def get_video_object(video_file=None):
    # if the video argument is None, then we are reading from the web cam
    if video_file is None:
        video = cv2.VideoCapture(0)
        time.sleep(0.25)
    else:
        video = cv2.VideoCapture(video_file)
    return video
