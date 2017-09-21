"""
Links:
  http://www.steinm.com/blog/motion-detection-webcam-python-opencv-differential-images/
"""
from pdb import set_trace as debug
import os
import sys
import cv2
from pydsutils.generic import create_logger

sys.path.insert(1, os.path.abspath("."))
import image_utils
from tracking_algos.tracking_algos import ImageDiffAlgo, OpencvOemAlgo

logger = create_logger(__name__, level="info")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_file", default="")
    parser.add_argument("--min_area", type=int, default=500, help="minimum area size")
    args = vars(parser.parse_args())
    args["video_file"] = "/Users/{user}/data/motion_tracking/sample_1.mp4".format(user=os.environ["USER"])

    image_width = 1000
    do_blur = True
    algo_name = "OpencvOemAlgo"  # OpencvOemAlgo, ImageDiffAlgo

    video = image_utils.get_video_object(args.get("video_file", None))

    algo_map = {
        "ImageDiffAlgo": ImageDiffAlgo(video, image_width, do_blur),
        "OpencvOemAlgo": OpencvOemAlgo(video, image_width, tracking_algo_name="MIL")
    }
    algo_map[algo_name].run()

    # cleanup the video and close any open windows
    video.release()
    cv2.destroyAllWindows()

    logger.info("ALL DONE\n")
