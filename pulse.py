from image_processor import ImageProcessor
from webcam import WebCam


class PulseDetector(object):
    def __init__(self, ui):
        # Initialise WebCam Object - Utilises OpenCV to grab the first identified web cam
        web_cam = WebCam()
        if web_cam.valid:
            self.web_cam = web_cam
        self.width = 0
        self.height = 0
        self.pulse_detector_ui = ui

        # Handles all image processing, signal analysis, face detection etc
        self.processor = ImageProcessor()

    def toggle_search(self):
        # Lock on found face and begin pulse detection / Find faces
        is_locked = self.processor.find_faces_toggle()
        print("face detection lock =", not is_locked)

    def main_loop(self):
        # Get current image frame from the web_cam
        frame = self.web_cam.get_frame()
        self.height, self.width, depth = frame.shape

        # set current image frame to the processor's input
        self.processor.frame_in = frame
        # process the image frame to perform all needed analysis
        self.processor.find_and_detect()
        # collect the output frame for display
        output_frame = self.processor.frame_out

        # show the processed/annotated output frame
        self.pulse_detector_ui.show_frame(output_frame)

