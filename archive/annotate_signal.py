# %%
"""
Demonstrates some customized mouse interaction by drawing a crosshair that follows 
the mouse.
"""

import numpy as np

import pyqtgraph as pg
import glob
import pickle
from PyQt6.QtCore import Qt
from datetime import datetime
import os
from natsort import natsorted

# TODO: click and drag for start and end points
# TODO: visualize the start and end points


class TimeSeriesPlotter:
    def __init__(self, title="Time Series Plot"):
        self.app = pg.mkQApp("Time Series Plotter")
        self.win = pg.GraphicsLayoutWidget(show=True)
        self.win.setWindowTitle(title)
        self.label = pg.LabelItem(justify='right')
        self.win.addItem(self.label)
        self.average_plot = self.win.addPlot(row=1, col=0)
        self.plot = self.win.addPlot(row=1, col=0)
        self.plot.setAutoVisible(y=True)
        self.plot.setMouseEnabled(x=False, y=False)
        self.average_plot.setMouseEnabled(x=False, y=False)
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.hLine = pg.InfiniteLine(angle=0, movable=False)
        self.plot.addItem(self.vLine, ignoreBounds=True)
        self.plot.addItem(self.hLine, ignoreBounds=True)
        self.plot.setMenuEnabled(False)
        self.average_plot.setMenuEnabled(False)
        self.vb = self.plot.vb
        self.file_names = glob.glob('/home/briancottle/Code/SoundPass/datasets/phantom_datasets/**/*.pkl', recursive=True)
        self.current_index = 1131
        self.data = None
        self.average_data = None
        self.starts = []
        self.ends = []
        
        # Sort the file names naturally
        self.file_names = natsorted(self.file_names)

        # Load all pickle files into a single numpy array
        all_data = []
        for file_name in self.file_names:
            with open(file_name, 'rb') as f:
                data = pickle.load(f)
                averaged_data = np.mean(data[0], axis=0)
            all_data.append(averaged_data)
        self.all_data_array = all_data

        print(f'starting from index {self.current_index} of {len(self.file_names)} files')

        # Create a new save directory using the current date and "ground_truth"
        current_date = datetime.now().strftime("%Y-%m-%d")
        self.save_dir = f"/home/briancottle/Code/SoundPass/{current_date}_ground_truth"
        os.makedirs(self.save_dir, exist_ok=True)

    def load_data(self):
        self.data = self.all_data_array[self.current_index]
        # Compute the average of the past 3 samples and the next 3 samples
        start_idx = max(0, self.current_index - 3)
        end_idx = min(len(self.all_data_array), self.current_index + 4)  # +4 because range is exclusive
        self.average_data = np.mean(self.all_data_array[start_idx:end_idx], axis=0)
        

    def plot_data(self):
        self.load_data()
        self.average_plot.plot(self.average_data, pen=pg.mkPen(color=(255, 255, 255, 128)))  # White with alpha 0.5
        self.plot.plot(self.data, pen=pg.mkPen(color=(255, 0, 0, 128)))  # Red with alpha 0.5
        
    def key_pressed(self, evt):
        if evt.key() == Qt.Key.Key_Space:
            gt = np.zeros(len(self.data))
            if self.starts and self.ends:
                for start, end in zip(self.starts, self.ends):
                    gt[start:end] = 1
            else:
                print("No start or end points recorded. Ground truth will remain all zeros.")

            save_path = os.path.join(self.save_dir, f"ground_truth_{self.current_index}.pkl")
            with open(save_path, 'wb') as f:
                pickle.dump({'data': self.data, 'ground_truth': gt}, f)
            print(f"Saved ground truth and data to {save_path}")
            self.starts = []
            self.ends = []

            self.current_index += 1
            self.plot.clear()
            self.average_plot.clear() 
            self.plot_data()
              

    def mouse_moved(self, evt):
        pos = evt
        if self.plot.sceneBoundingRect().contains(pos):
            mousePoint = self.vb.mapSceneToView(pos)
            self.vLine.setPos(mousePoint.x())
            self.hLine.setPos(mousePoint.y())

    def mouse_clicked(self, evt):
        x_location = int(self.vb.mapSceneToView(evt.scenePos()).x())
        y_location = int(self.vb.mapSceneToView(evt.scenePos()).y())

        print(f'Mouse clicked at x:{x_location}, y:{y_location}')

        if evt.button() == Qt.MouseButton.RightButton:
            self.ends.append(x_location)
            print(f'Signal end recorded at: {x_location}')

        if evt.button() == Qt.MouseButton.LeftButton:
            self.starts.append(x_location)
            print(f'Signal start recorded at: {x_location}')

        if evt.button() == Qt.MouseButton.MiddleButton:
            self.starts = []
            self.ends = []
            print("Start and end points have been reset.")



    def connect_signals(self):
        self.plot.scene().sigMouseMoved.connect(self.mouse_moved)
        self.plot.scene().sigMouseClicked.connect(self.mouse_clicked)
        self.app.installEventFilter(self.win)
        self.win.keyPressEvent = self.key_pressed

    def run(self):
        self.connect_signals()
        self.load_data
        self.plot_data()
        pg.exec()


if __name__ == '__main__':
    plotter = TimeSeriesPlotter()
    plotter.run()


