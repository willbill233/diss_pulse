import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from PIL import Image, ImageTk
from pulse import PulseDetector
import matplotlib
import matplotlib.animation as anim
from matplotlib import style
matplotlib.use("TkAgg")
style.use("dark_background")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class PlotWindow(tk.Toplevel):
    def __init__(self, original):
        tk.Toplevel.__init__(self)
        self.geometry('900x900')
        self.original = original

        frame = tk.Frame(self)
        frame.pack(fill='both')

        # Figure to show how consistent the samples being processed are
        # I.E how still the subject is keeping their forehead.
        # Good indicator of accuracy.
        self.rgb_samples_figure = Figure(figsize=(7, 3), dpi=100)
        self.time_samples = self.rgb_samples_figure.add_subplot(111)
        self.time_samples.yaxis.set_visible(False)
        self.time_samples_canvas = FigureCanvasTkAgg(self.rgb_samples_figure, master=frame)
        self.time_samples_canvas.show()
        self.time_samples_canvas.get_tk_widget().pack(fill='both')

        # Mayer Waves, Attempting to be similar to the output of an ECG device.
        self.bpm_figure = Figure(figsize=(7, 3), dpi=100)
        self.bpm_fft = self.bpm_figure.add_subplot(111)
        self.bpm_fft.yaxis.set_visible(False)
        self.bpm_fft_canvas = FigureCanvasTkAgg(self.bpm_figure, master=frame)
        self.bpm_fft_canvas.show()
        self.bpm_fft_canvas.get_tk_widget().pack(fill='both')

        self.signal_figure = Figure(figsize=(7, 3), dpi=100)
        self.heart_signal = self.signal_figure.add_subplot(111)
        self.heart_signal.yaxis.set_visible(False)
        self.heart_signal_canvas = FigureCanvasTkAgg(self.signal_figure, master=frame)
        self.heart_signal_canvas.show()
        self.heart_signal_canvas.get_tk_widget().pack(fill='both')

        self.bpm_ani = anim.FuncAnimation(self.bpm_figure, self.animate_bpm)
        self.bpm = None
        self.fft = None
        self.samples_ani = anim.FuncAnimation(self.rgb_samples_figure, self.animate_samples)
        self.time = None
        self.samples = None
        self.signal_ani = anim.FuncAnimation(self.signal_figure, self.animate_signals)
        self.even_times = None
        self.filtered_fft = None

    def plot_time_samples(self, x, y):
        self.time = np.array(x)
        self.samples = np.array(y)

    def plot_bpm_fft(self, x, y):
        self.bpm = np.array(x)
        self.fft = np.array(y)

    def plot_heart_signal(self, x, y):
        self.even_times = np.array(x)
        self.filtered_fft = np.array(y)

    def animate_bpm(self, i):
        self.bpm_fft.clear()
        self.bpm_fft.plot(self.bpm, self.fft, '-b')
        if len(self.fft) > 5 and len(self.bpm) > 5:
            ymax = np.amax(self.fft)
            xpos = np.argmax(self.fft)
            xmax = self.bpm[xpos]
            text = '{:0.1f} BPM'.format(xmax)
            self.bpm_fft.annotate(text, xy=(xmax, ymax), xytext=(xmax, ymax), color='white')

    def animate_samples(self, i):
        self.time_samples.clear()
        self.time_samples.plot(self.time, self.samples, '-r')

    def animate_signals(self, i):
        self.heart_signal.clear()
        self.heart_signal.plot(self.even_times, self.filtered_fft, '-g')


class MainWindow:
    def __init__(self, master, title='Pulse Detector'):
        self.app = PulseDetector(self)
        self.master = master
        self.master.title(title)
        self.master.geometry('960x600')

        self.__cardiac_window = None

        # Key strokes received on main frame (entire window)
        self.main_frame = tk.Frame(master)
        self.main_frame.bind('s', self.start_stop)
        self.main_frame.bind('c', self.cardiac_data)
        self.main_frame.bind('<Escape>', self.quit)
        
        # Cam frames
        self.left_frame = tk.Frame(self.main_frame)

        # Instructions and buttons
        self.right_frame = tk.Frame(self.main_frame)
        self.greeting = tk.Label(self.right_frame, text='Welcome to pulse detector!')
        self.greeting.pack()

        self.instruction_label = tk.Label(self.right_frame, text='Instructions: ', justify='left')
        self.instruction_label.pack(fill='both')
        self.instruction_label.place()

        self.instructions_label = tk.Label(self.right_frame,
                                           text='1) Remove any eyewear.\n'
                                                '2) Wait for your face to be detected.\n'
                                                '3) Ensure hair is not obstructing your forehead.\n'
                                                '4) Hit \'Start\' or \'S\' to begin.\n'
                                                '5) Hit \'Cardiac Data\' or \'C\' to see ECG-type plots.\n'
                                                '6) Hit \'Quit\' or \'ESC\' to quit the application.',
                                           justify='left')
        self.instructions_label.pack(fill='both')
        self.instructions_label.place()

        self.start_stop_text = 'Start'
        self.start_button = tk.Button(self.right_frame, text=self.start_stop_text, command=self.start_stop, width=50)
        self.start_button.pack()
        self.cardiac_button = tk.Button(self.right_frame, text='Cardiac Data', command=self.cardiac_data, width=50)
        self.cardiac_button.pack()
        self.quit_button = tk.Button(self.right_frame, text='Quit', command=self.quit, width=50)
        self.quit_button.pack()
        self.error_message = tk.Label(self.right_frame, text='', foreground='red')
        self.error_message.pack()
        self.actual_bpm_label = tk.Label(self.right_frame, text="Actual BPM: ")
        self.actual_bpm_label.pack(side='left')
        self.actual_bpm = tk.Entry(self.right_frame, justify='right')
        self.actual_bpm.pack(side='left')
        self.save_button = tk.Button(self.right_frame, text='Save', command=self.save_record, width=15)
        self.save_button.pack(side='left', padx=5)
        self.left_frame.pack(side='left')
        self.right_frame.pack(side='left')
        self.main_frame.focus()
        self.main_frame.pack()

        self.video_feed = None
        self.start_toggle = False
        self.cardiac_data_toggle = False

    def show_frame(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        if self.video_feed is None:
            self.video_feed = tk.Label(self.left_frame, image=image, borderwidth=2, relief="solid")
            self.video_feed.image = image
            self.video_feed.pack()
        else:
            self.video_feed.configure(image=image)
            self.video_feed.image = image

    def start_stop(self, event=None):
        self.start_toggle = not self.start_toggle
        self.start_stop_text = 'Stop' if self.start_toggle else 'Start'
        self.start_button.config(text=self.start_stop_text)
        self.app.toggle_search()

    def cardiac_data(self, event=None):
        if self.__cardiac_window:
            self.__cardiac_window.destroy()
            self.__cardiac_window = None
        else:
            self.__cardiac_window = PlotWindow(self)
        self.app.toggle_cardiac_data()

    def plot_time_samples(self, time, samples):
        self.__cardiac_window.plot_time_samples(time, samples)

    def plot_bpm_fft(self, bpm, fft):
        self.__cardiac_window.plot_bpm_fft(bpm, fft)

    def plot_heart_signal(self, even_times, filtered_fft):
        self.__cardiac_window.plot_heart_signal(even_times, filtered_fft)

    def save_record(self):
        try:
            self.error_message.config(text='', foreground='red')

            bpm = float(self.actual_bpm.get())
            data = self.app.processor.get_example()
            if not data:
                self.error_message.config(text='Start/Wait until the estimation is stable before saving')
                return

            data['frequency_actual'] = bpm / 60.
            data['frequency_estimate_reg'] = self.app.processor.bpm_regressor / 60.
            df = pd.DataFrame(data=data, index=[0])
            df.reindex_axis(sorted(df.columns), axis=1)
            with open('train_fft.csv', 'a') as train:
                df.to_csv(train, header=False)
                self.error_message.config(text='Save successful!', foreground='green')
        except ValueError:
            self.error_message.config(text='Please enter a number.')
            return

    def quit(self, event=None):
        self.master.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    GUI = MainWindow(root)
    while True:
        GUI.app.main_loop()
        root.update_idletasks()
        root.update()
