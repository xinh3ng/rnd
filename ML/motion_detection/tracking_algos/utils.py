import cv2
import datetime

def add_caption(image, text=""):
    cv2.putText(image, "Status: {}".format(text), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(image, datetime.datetime.now().strftime("%Y-%m-%d %I:%M:%S%p"),
                (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    return image


