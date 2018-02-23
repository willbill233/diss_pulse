import numpy as np
import time
import cv2
import os


class ImageProcessor(object):

    def __init__(self):
        cascade_file = os.path.abspath("haarcascade_frontalface_alt.xml")
        if not os.path.exists(cascade_file):
            print("Cascade file not present!")

        self.face_cascade = cv2.CascadeClassifier(cascade_file)
        self.frame_in = np.zeros((1, 1))
        self.frame_out = np.zeros((1, 1))
        self.grey_frame = np.zeros((1, 1))
        self.label_colour = (0, 255, 0)
        self.fps = 0
        self.no_of_avgs_limit = 250
        self.forehead_colour_avgs = []
        self.times = []
        self.samples = []
        self.angles = []
        self.raw_freqs = []
        self.fft = []
        self.filtered = []
        self.even_times = []
        self.roi = [[0]]
        self.start_time = time.time()
        self.bpm = 0
        self.face_rect = [1, 1, 1, 1]
        self.last_centre_coords = np.array([0, 0])
        self.find_faces = True

    def find_faces_toggle(self):
        self.find_faces = not self.find_faces
        return self.find_faces

    def rect_movement(self, detected):
        # Find centre of detected face - calculate the distance from last detected centre
        # Normalise so that we can accurately feedback when we need to reassign/redraw our rectangles
        x, y, w, h = detected
        centre_coords = np.array([x + 0.5 * w, y + 0.5 * h])
        centre_differential = np.linalg.norm(centre_coords - self.last_centre_coords)
        self.last_centre_coords = centre_coords
        return centre_differential

    def forehead_rect(self, fh_x=0.5, fh_y=0.18, fh_w=0.25, fh_h=0.15):
        x, y, w, h = self.face_rect
        return [int(x + w * fh_x - (w * fh_w / 2.0)),
                int(y + h * fh_y - (h * fh_h / 2.0)),
                int(w * fh_w),
                int(h * fh_h)]

    def forehead_colour_avg(self, coord):
        # AVG of each column (BGR) then avg the total of those averages
        x, y, w, h = coord
        forehead_only_frame = self.frame_in[y:y + h, x:x + w, :]
        blue_avg = np.mean(forehead_only_frame[:, :, 0])
        green_avg = np.mean(forehead_only_frame[:, :, 1])
        red_avg = np.mean(forehead_only_frame[:, :, 2])

        return (blue_avg + green_avg + red_avg) / 3.0

    def draw_frame_details(self):
        if self.find_faces:
            forehead = self.forehead_rect()
            fx, fy, fw, fh = self.face_rect
            fhx, fhy, fhw, fhh = forehead
            cv2.rectangle(self.frame_out, (fx, fy), (fx + fw, fy + fh), (0, 0, 255), 1)
            cv2.rectangle(self.frame_out, (fhx, fhy), (fhx + fhw, fhy + fhh), (0, 255, 0), 1)
            cv2.rectangle(self.frame_out, (fx, fy), (fx + fw, fy + fh), (0, 0, 255), 1)
            cv2.putText(self.frame_out, "Face",
                        (fx, fy - 5), cv2.FONT_HERSHEY_COMPLEX, 1, self.label_colour)
            cv2.rectangle(self.frame_out, (fhx, fhy), (fhx + fhw, fhy + fhh), (0, 255, 0), 1)
            cv2.putText(self.frame_out, "Forehead",
                        (fhx, fhy - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5, self.label_colour)

    def get_transformed_values(self, no_of_examples):
        self.fps = float(no_of_examples) / (self.times[-1] - self.times[0])
        # Evenly spaced times between first and lastly recorded times in array
        even_times = np.linspace(self.times[0], self.times[-1], no_of_examples)
        # Produce data points for plots x = time y = RgbAverages
        plots = np.interp(even_times, self.times, self.samples)
        # Get Hamming window and multiply against each data point to improve smoothness of values
        plots = np.hamming(no_of_examples) * plots
        # Minus the mean of all plots from each value to produce values that will be used in Fourier Transformation
        plots = plots - np.mean(plots)
        # Time -> Frequency domain via Fourier Transformation
        transformed = np.fft.rfft(plots)

        # Transformed vals in radians & absolute values of transformed
        self.angles = np.angle(transformed)
        self.fft = np.abs(transformed)
        self.raw_freqs = float(self.fps) / no_of_examples * np.arange(no_of_examples / 2 + 1)

        relevant_ids = np.where((self.raw_freqs > 50/60.) & (self.raw_freqs < 180/60.))
        x = 0 * transformed
        x[relevant_ids] = transformed[relevant_ids]
        self.filtered = np.fft.irfft(x)
        self.filtered = self.filtered / np.hamming(len(self.filtered))
        self.even_times = np.linspace(self.times[0], self.times[-1], len(self.filtered))

    def sync_light_intensity_changes(self, largest_relevant_angle, area_of_interest):
        # Calculating the sine value to identify discrete changes in light intensity so we can sync with the users
        # Pulse to give some sort of UI feedback
        t = (np.sin(largest_relevant_angle) + 1.0) / 2.0
        primary = t
        secondary = 1 - t

        # Keep B and R the same, use G as the indicator by adding with the grey frame pixel values
        fhx, fhy, fhw, fhh = area_of_interest
        blue = primary * self.frame_in[fhy:fhy + fhh, fhx:fhx + fhw, 0]
        green = primary * self.frame_in[fhy:fhy + fhh, fhx:fhx + fhw, 1] + secondary * self.grey_frame[fhy:fhy + fhh,
                                                                                       fhx:fhx + fhw]
        red = primary * self.frame_in[fhy:fhy + fhh, fhx:fhx + fhw, 2]
        self.frame_out[fhy:fhy + fhh, fhx:fhx + fhw] = cv2.merge([blue, green, red])

    def find_and_detect(self):
        self.times.append(time.time() - self.start_time)
        self.frame_out = self.frame_in
        self.grey_frame = cv2.equalizeHist(cv2.cvtColor(self.frame_in, cv2.COLOR_BGR2GRAY))  # For facial detection

        # If a face has not been locked yet then we should populate the frame with instruction text
        if self.find_faces:
            self.draw_frame_details()
            self.forehead_colour_avgs, self.times = [], []
            detected_rects = list(self.face_cascade.detectMultiScale(self.grey_frame,
                                                                     scaleFactor=1.3))

            if len(detected_rects) > 0:
                # Sort by area or rectangles
                detected_rects.sort(key=lambda rect: rect[-1] * rect[-2])
                if self.rect_movement(detected_rects[-1]) > 10:
                    self.face_rect = detected_rects[-1]  # Largest rect
                # Draw rectangles and labels for identified areas
            self.draw_frame_details()
            return
        # Put different labels if they have locked and we're monitoring heart rate
        self.draw_frame_details()

        forehead = self.forehead_rect()
        avg = self.forehead_colour_avg(forehead)
        self.forehead_colour_avgs.append(avg)

        # Pop first element of respective arrays via slicing if at limit
        no_of_avgs = len(self.forehead_colour_avgs)
        if no_of_avgs > self.no_of_avgs_limit:
            self.forehead_colour_avgs = self.forehead_colour_avgs[1:]
            self.times = self.times[1:]
            no_of_avgs = self.no_of_avgs_limit

        processed = np.array(self.forehead_colour_avgs)
        self.samples = processed
        if no_of_avgs > 10:
            self.get_transformed_values(no_of_avgs)

            # Grab element id's when they're within normal bpm params
            freqs = 60.0 * self.raw_freqs
            ids = np.where((freqs > 50) & (freqs < 180))

            relevant_values = self.fft[ids]  # Grab absolute values the ids correspond with
            relevant_angles = self.angles[ids]  # Grab angles in radians the ids correspond with
            relevant_freqs = freqs[ids]  # Grab the actual frequents the ids correspond with

            self.raw_freqs = relevant_freqs
            self.fft = relevant_values
            if len(relevant_values) <= 0:
                return

            # Grab id of largest relevant value and use this to grab the bpm estimation
            max_id = np.argmax(relevant_values)
            forehead = self.forehead_rect()
            self.bpm = self.raw_freqs[max_id]
            self.sync_light_intensity_changes(relevant_angles[max_id], forehead)

            # Draw bpm estimation on screen with label
            fx, fy, fw, fh = self.face_rect
            fhx, fhy, fhw, fhh = forehead
            self.roi = [np.copy(self.frame_out[fx:fy + fh, fx:fx + fw, 1])]
            text = "Estimate: %0.1f bpm" % self.bpm
            gap = (self.no_of_avgs_limit - no_of_avgs) / self.fps
            if gap:
                text = "Estimate: %0.1f bpm, stable in %0.0f s" % (self.bpm, gap)
            cv2.putText(self.frame_out, text,
                        (int(fhx - fhw / 2), int(fhy - 5)), cv2.FONT_HERSHEY_PLAIN, 1, self.label_colour)
