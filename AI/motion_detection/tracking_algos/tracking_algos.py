"""
Links:
  http://www.steinm.com/blog/motion-detection-webcam-python-opencv-differential-images/
"""
from pdb import set_trace as debug
import datetime
import cv2
from pydsutils.generic import create_logger
import image_utils
from tracking_algos.utils import add_caption

logger = create_logger(__name__, level="info")


def process_color_image(image, image_width):
    # Resize the image
    height, width = image.shape[:2]
    new_height = int(image_width / width * height)
    image = cv2.resize(image, (image_width, new_height))

    # NB xheng: maybe doing something else in future
    return image


def process_gray_image(image, do_blur=True):
    """Process grayscale image

    :param image:
    :param do_blur:
    :return:
    """
    if do_blur:
        image = cv2.GaussianBlur(image, (21, 21), 0)  # 21 is supposed to be related to 500
    return image


def process_bw_image(image):
    return image


def diff_image_pair(images):
    return cv2.absdiff(images[1], images[0])


def diff_image_trio(images):
    """Take differences on three images

    :param images: A tuple of 3 image objects
    :return:
    """
    d2 = cv2.absdiff(images[2], images[1])
    d1 = cv2.absdiff(images[1], images[0])
    return cv2.bitwise_and(d1, d2)


def label_moving_objects(image, contours, min_area):
    for ctr in contours:
        if cv2.contourArea(ctr) < min_area:  # ignore if the contour is too small
            continue
        x, y, w, h = cv2.boundingRect(ctr)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image


class ImageDiffAlgo(object):
    def __init__(self, video, image_width, do_blur, min_area=500):
        self.video = video
        self.image_width = image_width
        self.do_blur = do_blur
        self.min_area = min_area

    def run(self):
        # Grab the first 2 frames
        _, color_frame = self.video.read()
        ci = process_color_image(color_frame, self.image_width)
        first_recent_gray = process_gray_image(cv2.cvtColor(ci, cv2.COLOR_BGR2GRAY), do_blur=self.do_blur)
        _, color_frame = self.video.read()
        ci = process_color_image(color_frame, self.image_width)
        gray_frame = process_gray_image(cv2.cvtColor(ci, cv2.COLOR_BGR2GRAY), do_blur=self.do_blur)

        # Grab frames continuously
        while True:
            grabbed, color_frame = self.video.read()
            text = "Unoccupied"
            if not grabbed:  # if the frame could not be grabbed
                break

            second_recent_gray = first_recent_gray
            first_recent_gray = gray_frame
            color_frame = process_color_image(color_frame, self.image_width)
            gray_frame = process_gray_image(cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY), do_blur=self.do_blur)
            # Take a image difference and threshold it
            frame_delta = diff_image_pair((first_recent_gray, gray_frame))
            # frame_delta = diff_image_trio((second_recent_gray, first_recent_gray, gray_frame))

            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            _, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            color_frame = label_moving_objects(color_frame, contours, self.min_area)
            color_frame = add_caption(color_frame, text="")

            do_quit = image_utils.imshow("Security camera", color_frame)["quit"]
            if do_quit:
                break
        return


class OpencvOemAlgo(object):

    tracker_init_fns = {"MIL": cv2.TrackerMIL_create}

    def __init__(self, video, image_width, tracking_algo_name="MIL"):
        self.video = video
        self.image_width = image_width
        self.tracking_algo_name = tracking_algo_name
        self.tracker = self.tracker_init_fns[self.tracking_algo_name]()

    def run(self):
        grabbed, color_frame = self.video.read()
        color_frame = process_color_image(color_frame, self.image_width)

        # Define an initial bounding box
        bbox = (287, 23, 86, 320)
        # Uncomment the line below to select a different bounding box
        # bbox = cv2.selectROI(color_frame, False)

        # Initialize tracker with first frame and bounding box
        grabbed = self.tracker.init(color_frame, bbox)

        while True:
            grabbed, color_frame = self.video.read()
            if not grabbed:
                break

            color_frame = process_color_image(color_frame, self.image_width)
            grabbed, bbox = self.tracker.update(color_frame)

            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(color_frame, p1, p2, (0, 0, 255))

            color_frame = add_caption(color_frame, text="")
            do_quit = image_utils.imshow("Security camera", color_frame)["quit"]
            if do_quit:
                break
