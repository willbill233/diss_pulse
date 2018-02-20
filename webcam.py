import cv2
import numpy as np


class WebCam(object):
    def __init__(self):
        # Grabs first camera
        self.cam = cv2.VideoCapture(0)
        self.valid = False
        try:
            # Attempts to read a frame to validate that the camera has been identified
            resp = self.cam.read()
            self.shape = resp[1].shape  # Throws if image could not be retrieved
            self.valid = True
        except:
            self.shape = None

    def get_frame(self):
        if self.valid:
            # If cam valid, grabs image to ultimately be processed
            retval, image = self.cam.read()
        else:
            # If cam is invalid, produce and return blank mock frame
            image = np.ones((1280, 720, 3), dtype=np.uint8)
            colour = (0, 0, 255)
            cv2.putText(image, "(Error: web_camera not accessible)",
                        (50, 200), cv2.FONT_HERSHEY_PLAIN, 2, colour)
        return image

    def release(self):
        self.cam.release()
