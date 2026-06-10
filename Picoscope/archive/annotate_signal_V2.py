import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
import numpy as np
import glob
import pickle
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QVBoxLayout, QWidget, QLabel, QSpinBox
from datetime import datetime
import os
from natsort import natsorted

class CustomViewBox(pg.ViewBox):
    def __init__(self, parent_plotter):
        super().__init__()
        self.parent_plotter = parent_plotter
        self.current_region = None  # Track the region being created

    def mousePressEvent(self, event):
        """
        Handle mouse press events.
        """
        if event.button() == Qt.MouseButton.MiddleButton:
            # Reset all regions
            for region in self.parent_plotter.regions:
                self.parent_plotter.plot.removeItem(region)
            self.parent_plotter.regions = []
            self.parent_plotter.starts = []
            self.parent_plotter.ends = []
            print("All regions have been reset.")
        elif event.button() == Qt.MouseButton.LeftButton:
            x_location = self.mapSceneToView(event.scenePos()).x()
            self.current_region = pg.LinearRegionItem([x_location, x_location], movable=True)
            self.current_region.setZValue(10)  # Ensure it appears above the plot
            self.parent_plotter.plot.addItem(self.current_region)
            self.parent_plotter.regions.append(self.current_region)
            print(f"Started region at: {x_location}")
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """
        Handle mouse movement to adjust the region being created.
        """
        if self.current_region is not None:
            x_location = self.mapSceneToView(event.scenePos()).x()
            bounds = self.current_region.getRegion()
            self.current_region.setRegion([bounds[0], x_location])
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """
        Handle mouse release to finalize the region.
        """
        if event.button() == Qt.MouseButton.LeftButton and self.current_region is not None:
            bounds = self.current_region.getRegion()
            print(f"Region bounds in x-coordinates: {bounds}")

            # Map the x-coordinates to indices of the data array
            start_index = max(0, np.searchsorted(self.parent_plotter.distance, bounds[0]))
            end_index = min(len(self.parent_plotter.data), np.searchsorted(self.parent_plotter.distance, bounds[1]))

            # Append the indices to starts and ends
            self.parent_plotter.starts.append(start_index)
            self.parent_plotter.ends.append(end_index)
            print(f"Region finalized: start={start_index}, end={end_index}")

            self.current_region = None
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class TimeSeriesPlotter(QWidget):
    def __init__(self, title="Time Series Plot"):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(1200, 800)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Layout for the dropdown and plot
        layout = QVBoxLayout(self)
        self.setLayout(layout)

        # Directory selector dropdown
        self.dir_selector = QComboBox()
        layout.addWidget(self.dir_selector)
        self.dir_selector.currentIndexChanged.connect(self.directory_changed)

        # File name display
        self.file_label = QLabel("Current file: ")
        self.file_label.setStyleSheet("font-size: 18pt; color: white;")
        layout.addWidget(self.file_label)

        # File index selector
        self.file_index_selector = QSpinBox()
        self.file_index_selector.setMinimum(0)
        self.file_index_selector.valueChanged.connect(self.file_index_changed)
        layout.addWidget(self.file_index_selector)

        # PyQtGraph setup
        self.app = pg.mkQApp("Time Series Plotter")
        self.win = pg.GraphicsLayoutWidget(show=True)
        layout.addWidget(self.win)
        self.win.setWindowTitle(title)
        self.label = pg.LabelItem(justify='right')
        self.win.addItem(self.label)
        self.average_plot = self.win.addPlot(row=1, col=0)
        self.plot = self.win.addPlot(row=1, col=0, viewBox=CustomViewBox(self))  # Use custom ViewBox
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

        # Initialize variables
        self.current_index = 0
        self.data = None
        self.average_data = [0]
        self.starts = []
        self.ends = []
        self.regions = []  # Store LinearRegionItems here
        self.distance = [0]

        # Load subdirectories
        self.top_level_directory = "/Volumes/Samsung_T5/SoundPass_datasets/new_datasets/2025-09-15"
        self.directories = [
            os.path.join(self.top_level_directory, d)
            for d in os.listdir(self.top_level_directory)
            if os.path.isdir(os.path.join(self.top_level_directory, d))
        ]
        self.directories = natsorted(self.directories)
        self.dir_selector.addItems(self.directories)

        # Automatically select the first directory if available
        if self.directories:
            self.dir_selector.setCurrentIndex(0)

        # Create a new save directory using the current date and "ground_truth"
        current_date = datetime.now().strftime("%Y-%m-%d")
        self.save_dir = f"/Users/briancottle/Library/Mobile Documents/com~apple~CloudDocs/Consulting/SoundPass/data/{current_date}_ground_truth"
        os.makedirs(self.save_dir, exist_ok=True)

    def directory_changed(self, idx):
        """
        Load files from the selected directory.
        """
        directory = self.directories[idx]
        self.file_names = natsorted(glob.glob(os.path.join(directory, "*.pkl")))
        self.all_data_array = []
        self.all_file_names = []
        self.distances = []  # Add this to store distances for each file

        if not self.file_names:
            print(f"No valid files found in directory: {directory}")
            self.current_index = 0
            self.all_data_array = []
            self.all_file_names = []
            self.distances = []
            self.plot.clear()
            self.average_plot.clear()
            self.file_label.setText("No files in directory.")
            self.file_index_selector.setMaximum(0)
            return

        for file_name in self.file_names:
            with open(file_name, 'rb') as f:
                data = pickle.load(f)
                averaged_data = np.mean(data[0], axis=0)
                self.all_data_array.append(averaged_data)
                self.distances.append(data[1])  # Assuming data[1] contains the distance array
                self.all_file_names.append(file_name)

        self.current_index = 0
        self.file_index_selector.setMaximum(len(self.file_names) - 1)  # Update the spinbox range
        self.plot_data()

    def file_index_changed(self, index):
        """
        Handle changes to the file index via the spinbox.
        """
        self.current_index = index
        self.plot_data()

    def load_data(self):
        self.data = self.all_data_array[self.current_index]
        self.distance = self.distances[self.current_index]  # Set distance for the current file

        # Compute the average of the past 3 samples and the next 3 samples
        start_idx = max(0, self.current_index - 3)
        end_idx = min(len(self.all_data_array), self.current_index + 4)  # +4 because range is exclusive
        self.average_data = np.mean(self.all_data_array[start_idx:end_idx], axis=0)
        self.current_file_name = self.all_file_names[self.current_index]
        print(f"Loading data from {self.current_file_name}")

        # Update the file label
        self.file_label.setText(f"Current file: {self.current_index + 1}/{len(self.all_file_names)} - {os.path.basename(self.current_file_name)}")

    def plot_data(self):
        self.load_data()
        self.average_plot.clear()
        self.plot.clear()
        self.average_plot.plot(self.distance, self.average_data, pen=pg.mkPen(color=(255, 255, 255, 128)))  # White with alpha 0.5
        self.plot.plot(self.distance, self.data, pen=pg.mkPen(color=(255, 0, 0, 128)))  # Red with alpha 0.5

    def key_pressed(self, evt):
        if evt.key() == Qt.Key.Key_Space:
            gt = np.zeros(len(self.data))
            if self.starts and self.ends:
                print(f"Starts: {self.starts}, Ends: {self.ends}")
                for start, end in zip(self.starts, self.ends):
                    if 0 <= start < len(gt) and 0 <= end <= len(gt):
                        gt[start:end] = 1
                    else:
                        print(f"Invalid region: start={start}, end={end}")
            else:
                print("No start or end points recorded. Ground truth will remain all zeros.")

            # Get the name of the selected subdirectory
            selected_subdir = os.path.basename(self.directories[self.dir_selector.currentIndex()])
            sub_save_dir = os.path.join(self.save_dir, selected_subdir)
            os.makedirs(sub_save_dir, exist_ok=True)

            current_file_id = self.current_file_name.split('/')[-1].split('.')[0]
            save_path = os.path.join(sub_save_dir, f"{current_file_id}_ground_truth_{self.current_index}.pkl")
            with open(save_path, 'wb') as f:
                pickle.dump({'data': self.data, 'ground_truth': gt, 'file_name': self.current_file_name}, f)
            print(f"Saved ground truth and data to {save_path}")
            self.starts = []
            self.ends = []
            self.regions = []

            self.current_index += 1
            if self.current_index >= len(self.file_names):
                # Move to the next directory
                current_dir_idx = self.dir_selector.currentIndex()
                if current_dir_idx + 1 < len(self.directories):
                    self.dir_selector.setCurrentIndex(current_dir_idx + 1)
                else:
                    print("All directories processed.")
                    return
            self.plot_data()

    def keyPressEvent(self, event):
        """
        Handle key press events.
        """
        if event.key() == Qt.Key.Key_Space:
            self.key_pressed(event)
        else:
            super().keyPressEvent(event)

    def mouse_clicked(self, evt):
        """
        Handle mouse clicks to add or reset regions.
        """
        if evt.button() == Qt.MouseButton.MiddleButton:
            # Reset all regions
            for region in self.regions:
                self.plot.removeItem(region)
            self.regions = []
            self.starts = []
            self.ends = []
            print("All regions have been reset.")

    def run(self):
        self.plot_data()
        pg.exec()


if __name__ == '__main__':
    import sys
    app = pg.mkQApp("Time Series Plotter")  # Create the QApplication
    plotter = TimeSeriesPlotter()
    plotter.show()  # Show the main widget
    sys.exit(app.exec())  # Start the event loop