import tkinter as tk
from pulse import PulseDetector


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
        
        # Cam frames will sit here
        self.left_frame = tk.Frame(self.main_frame)

        # Instructions and buttons
        self.right_frame = tk.Frame(self.main_frame)
        self.greeting = tk.Label(self.right_frame, text='Welcome to pulse detector!')
        self.greeting.pack()

        self.instruction_label = tk.Label(self.right_frame, text='Instructions: ', justify='left')
        self.instruction_label.pack(fill='both')
        self.instruction_label.place()

        self.instructions_label = tk.Label(self.right_frame,
                                           text='Instuctions: ...',
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
        self.left_frame.pack(side='left')
        self.right_frame.pack(side='left')
        self.main_frame.focus()
        self.main_frame.pack()

        self.start_toggle = False
        self.cardiac_data_toggle = False

    def start_stop(self, event=None):
        return

    def cardiac_data(self, event=None):
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
