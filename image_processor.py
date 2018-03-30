import numpy as np
import pandas as pd
import time
import cv2
import os
from ica import ICA
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ImageProcessor(object):

    def __init__(self):
        cascade_file = os.path.abspath("haarcascade_frontalface_alt.xml")
        if not os.path.exists(cascade_file):
            print("Cascade file not present!")

        self.face_cascade = cv2.CascadeClassifier(cascade_file)
        self.frame_in = np.zeros((1, 1))
        self.frame_out = np.zeros((1, 1))
        self.grey_frame = np.zeros((1, 1))
        self.ica = ICA()
        self.label_colour = (0, 255, 0)
        self.fps = 0
        self.gap = 1
        self.no_of_avgs_limit = 250
        self.forehead_colour_avgs = []
        self.r_avgs = []
        self.g_avgs = []
        self.b_avgs = []
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
        # self.bpm_regressor = 0
        self.face_rect = [1, 1, 1, 1]
        self.last_centre_coords = np.array([0, 0])
        self.find_faces = True
    #     # ml values specific vals
    #     data = pd.read_csv('work_train.csv', header=0)
    #     attributes = data.loc[:, data.columns != 'bpm_actual']
    #     attributes = attributes.as_matrix()
    #     targets = np.array(data.bpm_actual.values)
    #
    #     attributes_train = attributes[:-33]
    #     attributes_test = attributes[-33:]
    #     targets_train = targets[:-33]
    #     targets_test = targets[-33:]
    #
    #     self.regressor = linear_model.LinearRegression()
    #     self.regressor.fit(attributes_train, targets_train)
    #     target_predictions = self.regressor.predict(attributes_test)
    #     # The coefficients
    #     print('Coefficients: \n', self.regressor.coef_)
    #     # The mean squared error
    #     print("Mean absolute error: %.2f"
    #           % mean_absolute_error(targets_test, target_predictions))
    #     # The mean squared error
    #     print("Mean squared error: %.2f"
    #           % mean_squared_error(targets_test, target_predictions))
    #     # Explained variance score: 1 is perfect prediction
    #     print('Variance score: %.2f' % r2_score(targets_test, target_predictions))
    #
    # def get_example(self):
    #     if self.gap or self.find_faces:
    #         return False
    #     data = {
    #         'bpm_estimate': self.bpm
    #     }
    #     for frame_no in np.arange(self.no_of_avgs_limit - 30, self.no_of_avgs_limit):
    #         data['r_avg' + str(frame_no)] = self.r_avgs[frame_no]
    #         data['g_avg' + str(frame_no)] = self.g_avgs[frame_no]
    #         data['b_avg' + str(frame_no)] = self.b_avgs[frame_no]
    #         data['roi_avg' + str(frame_no)] = self.forehead_colour_avgs[frame_no]
    #     return data

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

    def roi_channel_avgs(self, coord):
        # AVG of each column (BGR) then avg the total of those averages
        x, y, w, h = coord
        forehead_only_frame = self.frame_in[y:y + h, x:x + w, :]
        blue_avg = np.mean(forehead_only_frame[:, :, 0])
        green_avg = np.mean(forehead_only_frame[:, :, 1])
        red_avg = np.mean(forehead_only_frame[:, :, 2])
        self.r_avgs.append(red_avg)
        self.g_avgs.append(green_avg)
        self.b_avgs.append(blue_avg)
        avg = red_avg + green_avg + blue_avg / 3.
        self.forehead_colour_avgs.append(avg)

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

    def get_transformed_values(self, signal):
        no_of_examples = len(self.times)
        self.fps = float(no_of_examples) / (self.times[-1] - self.times[0])
        # Evenly spaced times between first and lastly recorded times in array
        even_times = np.linspace(self.times[0], self.times[-1], no_of_examples)
        # Produce data points for plots x = time y = RgbAverages
        plots = np.interp(even_times, self.times, signal)
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
        self.roi_channel_avgs(forehead)

        # Pop first element of respective arrays via slicing if at limit
        no_of_avgs = len(self.forehead_colour_avgs)
        if no_of_avgs > self.no_of_avgs_limit:
            self.forehead_colour_avgs = self.forehead_colour_avgs[1:]
            self.times = self.times[1:]
            self.r_avgs = self.r_avgs[1:]
            self.g_avgs = self.g_avgs[1:]
            self.b_avgs = self.b_avgs[1:]
            no_of_avgs = self.no_of_avgs_limit

        processed = np.array(self.forehead_colour_avgs)
        self.samples = processed
        if no_of_avgs > 10:
            signals = self.ica.analyse_rgb_channels([self.r_avgs, self.g_avgs, self.b_avgs], no_of_avgs)
            selected_signal = self.ica.select_pertinent_signal(signals)
            self.get_transformed_values(selected_signal)
            # Grab element id's when they're within normal bpm params
            freqs = 60.0 * self.raw_freqs
            ids = np.where((freqs > 50) & (freqs < 150))

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

            # data = self.get_example()
            # if data:
            #     df = pd.DataFrame(data=data, index=[0])
            #     df.reindex_axis(sorted(df.columns), axis=1)
            #     example_array = df.as_matrix()
            #     prediction = self.regressor.predict(example_array)
            #     if self.bpm - 5 < prediction < self.bpm + 5:
            #         self.bpm_regressor = prediction[0]

            # Draw bpm estimation on screen with label
            fx, fy, fw, fh = self.face_rect
            fhx, fhy, fhw, fhh = forehead
            self.roi = [np.copy(self.frame_out[fx:fy + fh, fx:fx + fw, 1])]
            text = "Estimate: %0.1f bpm" % self.bpm
            self.gap = (self.no_of_avgs_limit - no_of_avgs) / self.fps
            if self.gap:
                text = "Estimate: %0.1f bpm, stable in %0.0f s" % (self.bpm, self.gap)
            cv2.putText(self.frame_out, text,
                        (int(fhx - fhw / 2), int(fhy - 5)), cv2.FONT_HERSHEY_PLAIN, 1, self.label_colour)
            # cv2.putText(self.frame_out, 'Machine learning estimate: %0.1f bpm' % self.bpm_regressor,
            #             (5, 470), cv2.FONT_HERSHEY_PLAIN, 1, self.label_colour)
