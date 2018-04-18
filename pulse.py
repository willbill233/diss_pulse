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

        # Init parameters for the cardiac data plot
        self.cardiac_data = False
        self.plot_title = "Cardiac Data"

    def toggle_search(self):
        # Lock on found face and begin pulse detection / Find faces
        is_locked = self.processor.find_faces_toggle()
        print("face detection lock =", not is_locked)

    def toggle_cardiac_data(self):
        # Shows cardiac data / Hides cardiac data
        if self.cardiac_data:
            print("cardiac data disabled")
            self.cardiac_data = False
        else:
            print("cardiac data enabled")
            if self.processor.find_faces:
                self.toggle_search()
            self.cardiac_data = True
            self.make_cardiac_plot()

    def make_cardiac_plot(self):
        # Makes cardiac plots
        self.pulse_detector_ui.plot_time_samples(self.processor.times, self.processor.samples)
        self.pulse_detector_ui.plot_bpm_fft(self.processor.bpms, self.processor.relevant_fft)
        self.pulse_detector_ui.plot_heart_signal(self.processor.even_times[4:-4], self.processor.filtered[4:-4])

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

        # create and/or update the raw data display if needed
        if self.cardiac_data:
            self.make_cardiac_plot()
