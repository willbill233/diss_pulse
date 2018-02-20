from webcam import WebCam


class PulseDetector(object):
    def __init__(self, ui):
        # Initialise WebCam Object - Utilises OpenCV to grab the first identified web cam
        web_cam = WebCam()
        if web_cam.valid:
            self.web_cam = web_cam

        print(self.web_cam.get_frame())
        self.pulse_detector_ui = ui

    def main_loop(self):
        # Process frames
        return
