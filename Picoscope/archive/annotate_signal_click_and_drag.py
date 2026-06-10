# ...existing imports...

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
import numpy as np
import glob
import pickle
from PyQt6.QtCore import Qt
from datetime import datetime
import os
from natsort import natsorted

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
        self.current_index = 114
        self.data = None
        self.average_data = None
        self.starts = []
        self.ends = []
        self.regions = []  # Store LinearRegionItems here
        
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

    def add_region(self, start, end):
        """
        Add a new LinearRegionItem to the plot and store its start and end points.
        """
        region = pg.LinearRegionItem([start, end], movable=True)
        region.setZValue(10)  # Ensure it appears above the plot
        region.sigRegionChanged.connect(self.update_region_bounds)
        self.plot.addItem(region)
        self.regions.append(region)
        self.starts.append(int(start))
        self.ends.append(int(end))
        print(f"Added region: start={start}, end={end}")

    def update_region_bounds(self):
        """
        Update the start and end points of regions when they are moved.
        """
        self.starts = []
        self.ends = []
        for region in self.regions:
            bounds = region.getRegion()
            self.starts.append(int(bounds[0]))
            self.ends.append(int(bounds[1]))
        print(f"Updated regions: starts={self.starts}, ends={self.ends}")

    def mouse_clicked(self, evt):
        """
        Handle mouse clicks to add or reset regions.
        """
        if evt.button() == Qt.MouseButton.LeftButton:
            x_location = self.vb.mapSceneToView(evt.scenePos()).x()
            self.add_region(x_location, x_location + 10)  # Add a small region by default
        elif evt.button() == Qt.MouseButton.MiddleButton:
            # Reset all regions
            for region in self.regions:
                self.plot.removeItem(region)
            self.regions = []
            self.starts = []
            self.ends = []
            print("All regions have been reset.")

    def connect_signals(self):
        self.plot.scene().sigMouseClicked.connect(self.mouse_clicked)
        self.app.installEventFilter(self.win)
        self.win.keyPressEvent = self.key_pressed

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
            self.regions = []

            self.current_index += 1
            self.plot.clear()
            self.average_plot.clear()
            self.plot_data()

    def run(self):
        self.connect_signals()
        self.plot_data()
        pg.exec()


if __name__ == '__main__':
    plotter = TimeSeriesPlotter()
    plotter.run()