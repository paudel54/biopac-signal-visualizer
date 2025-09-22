
import sys
import pandas as pd
import numpy as np
import os

# sets an environment variable that tells QT to use rendering compatabile for MAC
if sys.platform == "darwin":
    os.environ["QT_MAC_WANTS_LAYER"] = "1"

#QT widgets:
"""
QApplication: the main application object; manages the event loop.
QMainWindow: a top-level window with menus, toolbars, central widget (on our main app window).
QWidget: the base class for all UI elements (generic widget).
QVBoxLayout / QHBoxLayout: layout managers (vertical/horizontal stacking of widgets).
QComboBox: drop-down selector.
QSpinBox: numeric spinner input.
QLabel: simple text/image display widget.
QFileDialog: system file chooser dialog.
QPushButton: clickable button.
"""
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QComboBox, QSpinBox, QLabel, QHBoxLayout, QFileDialog, QPushButton
)
from PyQt5.QtCore import Qt, QPropertyAnimation
import pyqtgraph as pg 
#Saving plots to disks. 
from pyqtgraph.exporters import ImageExporter,SVGExporter

class TimeSeriesViewer(QMainWindow):
    def __init__(self):
        #Calls the constructor of parent class (QMainWindow)
        super().__init__()
        self.setWindowTitle("Signals Visualizer")
        #GUI position when it pop's up initially.
        self.setGeometry(100, 100, 1200, 700)

        # Opens the file chooser dialog for user to select the CSV from merged biopac file. 
        csv_file, _ = QFileDialog.getOpenFileName(self, "Select merged CSV file", "", "CSV Files (*.csv)")
        if not csv_file:
            sys.exit("No file selected. Exiting.")
        self.load_data(csv_file)

        # Default settings
        self.current_signal = "ECG"
        self.sampling_rate = 30
        self.pan_mode = False

        self.signal_colors = {
            "ECG": "blue",
            "PPG": "orange",
            "EDA": "purple",
            "SKT": "darkorange"
        }
        self.signal_y_ranges = {
            "ECG": (-10, 10),
            "PPG": (-10, 10),
            "EDA": (0, 20),
            "SKT": (0, 40)
        }
        self.signal_units = {
            "ECG": "mV",
            "PPG": "a.u.",
            "EDA": "µS",
            "SKT": "°C"
        }

        self.init_ui()
        self.update_plot()

    def load_data(self, csv_file):
        self.data = pd.read_csv(csv_file, low_memory=False)
        self.data['relative_time_s'] = (
            self.data['timestamp_ms'] - self.data['timestamp_ms'].iloc[0]
        ) / 1000.0
        self.label_rows = self.data[
            self.data['label'].notna() & (self.data['label'].astype(str).str.strip() != "")
        ]

    def init_ui(self):
        #Central controller of all window
        widget = QWidget()
        #Arranges the child widgets vertically 
        layout = QVBoxLayout()
        #Arranges the child widgets vertically 
        control_layout = QHBoxLayout()

        #Adding save image button 
        self.save_btn = QPushButton("Save Image")
        self.save_btn.setStyleSheet("background-color: #1976d2; color: white; font-weight: bold;")
        self.save_btn.clicked.connect(self.save_image)
        control_layout.addWidget(self.save_btn)

        #Adding menu tab
        self.signal_selector = QComboBox()
        self.signal_selector.addItems(["ECG", "PPG", "SKT", "EDA"])
        #Sends information of signal change to on_signal_change() function to update plot. 
        self.signal_selector.currentTextChanged.connect(self.on_signal_change)

        #box for changing the sampling frequency
        self.sampling_rate_box = QSpinBox()
        self.sampling_rate_box.setRange(10, 1000)
        self.sampling_rate_box.setValue(self.sampling_rate)
        self.sampling_rate_box.valueChanged.connect(self.on_sampling_rate_change)

        # Pan/Zoom Toggle Button
        self.pan_button = QPushButton()
        self.pan_button.clicked.connect(self.toggle_pan_mode)

        #Toast Popup 
        self.toast = QLabel("", self)
        self.toast.setAlignment(Qt.AlignCenter)
        self.toast.setStyleSheet(
            "background-color: lightgray; color: black; padding: 8px; "
            "border-radius: 6px; font-weight: bold;"
        )
        self.toast.setVisible(False)

        control_layout.addWidget(QLabel("Select Signal:"))
        control_layout.addWidget(self.signal_selector)
        control_layout.addWidget(QLabel("Sampling Rate (Hz):"))
        control_layout.addWidget(self.sampling_rate_box)
        control_layout.addWidget(self.pan_button)
        layout.addLayout(control_layout)

        # Plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#f4f4f4')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setMouseEnabled(x=True, y=True)
        self.plot_widget.setMenuEnabled(True)
        self.plot_widget.setClipToView(True)
        self.plot_widget.setDownsampling(auto=True, mode='peak')
        self.plot_widget.getAxis("left").setTextPen("black")
        self.plot_widget.getAxis("bottom").setTextPen("black")
        self.plot_widget.getAxis("left").setStyle(tickFont=pg.QtGui.QFont("Arial", 10))
        self.plot_widget.getAxis("bottom").setStyle(tickFont=pg.QtGui.QFont("Arial", 10))
        self.plot_widget.scene().sigMouseClicked.connect(self.reset_zoom_on_double_click)
        self.plot_widget.getViewBox().setMouseMode(pg.ViewBox.RectMode)

        #Crosshair (Vertical + Horizontal)
        self.vLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('gray', style=pg.QtCore.Qt.DashLine))
        self.hLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('gray', style=pg.QtCore.Qt.DashLine))
        self.plot_widget.addItem(self.vLine, ignoreBounds=True)
        self.plot_widget.addItem(self.hLine, ignoreBounds=True)

        #Text Tooltip 
        self.tooltip = pg.TextItem("", anchor=(0, 1), color="black")
        self.plot_widget.addItem(self.tooltip)

        # Mouse move event for crosshair + tooltip
        self.proxy = pg.SignalProxy(self.plot_widget.scene().sigMouseMoved, rateLimit=60, slot=self.on_mouse_move)

        layout.addWidget(self.plot_widget)
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        # self.update_pan_button_color()  
        self.pan_button.setText("Activate pan mode")
        self.pan_button.setStyleSheet("background-color: green; color: white; font-weight: bold;")


#Function to save image:
    # --- NEW: save current plot as an image ---
    def save_image(self):
        """
        Export the current plot as an image using pyqtgraph's ImageExporter.
        You can save as PNG, JPG, or SVG.
        """
        default_name = f"{self.current_signal}_plot"
        fname, ffilter = QFileDialog.getSaveFileName(
            self,
            "Save Plot Image",
            f"{default_name}.png",
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;SVG Vector (*.svg)"
        )
        if not fname:
            return

        # Ensuring extension matches selected filter if user omitted it
        if '.' not in os.path.basename(fname):
            if "PNG" in ffilter:
                fname += ".png"
            elif "JPEG" in ffilter:
                fname += ".jpg"
            elif "SVG" in ffilter:
                fname += ".svg"
            else:
                fname += ".png"

        # # Create an exporter for the PlotItem (includes axes, title, grid, etc.)
        # exporter = ImageExporter(self.plot_widget.getPlotItem())
        # # Optional: bump resolution for sharper export (in pixels)
        # # Set desired width; height scales to keep aspect ratio
        # params = exporter.parameters()
        # params['width'] = 2560  # can be changed to (e.g., 2560 for 1440p)
            # --- pick exporter by extension ---
        if fname.lower().endswith(".svg"):
            exporter = SVGExporter(self.plot_widget.getPlotItem())
        else:
            exporter = ImageExporter(self.plot_widget.getPlotItem())
            exporter.parameters()['width'] = 2560  
        # ----------------------------------

        try:
            exporter.export(fname)
            self.show_toast(f"Saved: {os.path.basename(fname)}", color="#c8e6c9")
        except Exception as e:
            self.show_toast(f"Save failed: {e}", color="#ffcdd2")
    # -----------------------------------------
    
    def show_toast(self, text, color="lightgray", duration=3000):
        """Show toast popup stacked at the top-right corner of the plot widget."""
        self.toast.setText(text)
        self.toast.setStyleSheet(
            f"background-color: {color}; color: black; padding: 8px; "
            f"border-radius: 6px; font-weight: bold;"
        )
        self.toast.adjustSize()

        #Place relative to plot_widget (not global screen)
        padding_top = 3
        padding_right = 3
        self.toast.setParent(self.plot_widget)  # Important: makes it a child of the plot
        self.toast.move(
            self.plot_widget.width() - self.toast.width() - padding_right,
            padding_top
        )

        self.toast.raise_()  # Ensures it stays above the graph
        self.toast.setWindowOpacity(1.0)
        self.toast.setVisible(True)

        #Fade-out animation
        self.animation = QPropertyAnimation(self.toast, b"windowOpacity")
        self.animation.setDuration(duration)
        self.animation.setStartValue(1.0)
        self.animation.setEndValue(0.0)
        self.animation.finished.connect(lambda: self.toast.setVisible(False))
        self.animation.start()



    def update_pan_button_color(self):
        """Update button & show toast depending on mode"""
        if self.pan_mode:
            self.pan_button.setText("Activate zoom mode")
            self.pan_button.setStyleSheet("background-color: yellow; color: black; font-weight: bold;")
            self.show_toast("Pan mode active (drag to move)", "lightgreen")
        else:
            self.pan_button.setText("Activate pan mode")
            self.pan_button.setStyleSheet("background-color: green; color: white; font-weight: bold;")
            self.show_toast("Zoom mode active (drag to zoom, double-click to reset)", "yellow")

    def toggle_pan_mode(self):
        self.pan_mode = not self.pan_mode
        if self.pan_mode:
            self.plot_widget.getViewBox().setMouseMode(pg.ViewBox.PanMode)
        else:
            self.plot_widget.getViewBox().setMouseMode(pg.ViewBox.RectMode)
        self.update_pan_button_color()

    def reset_zoom_on_double_click(self, event):
        if event.double():
            self.plot_widget.enableAutoRange()

    def downsample_data(self):
        original_rate = 1000
        if self.sampling_rate >= original_rate:
            return self.data
        step = max(1, int(original_rate / self.sampling_rate))
        downsampled = self.data.iloc[::step, :]
        combined = (
            pd.concat([downsampled, self.label_rows])
            .drop_duplicates(subset='timestamp_ms')
            .sort_values('timestamp_ms')
        )
        return combined

    def update_plot(self):
        self.plot_widget.clear()
        df = self.downsample_data()
        self.current_df = df

        pen_color = self.signal_colors.get(self.current_signal, "blue")
        pen = pg.mkPen(color=pen_color, width=1.5)
        self.plot_widget.plot(df['relative_time_s'], df[self.current_signal], pen=pen)

        y_min, y_max = self.signal_y_ranges.get(
            self.current_signal, (df[self.current_signal].min(), df[self.current_signal].max())
        )
        self.plot_widget.setYRange(y_min, y_max, padding=0)

        y_base = y_max * 1.05
        stagger_height = 0.08 * abs(y_base)
        stagger_direction = [1, -1, 2, -2]
        count = 0
        for _, row in df.iterrows():
            if isinstance(row['label'], str) and row['label'].strip() != "":
                x_val = row['relative_time_s']
                vline = pg.InfiniteLine(pos=x_val, angle=90, pen=pg.mkPen('r', style=pg.QtCore.Qt.DashLine))
                self.plot_widget.addItem(vline)
                offset = stagger_direction[count % len(stagger_direction)] * stagger_height
                text_y = y_base + offset
                text = pg.TextItem(row['label'], color='r', anchor=(0, 0))
                text.setPos(x_val, text_y)
                text.setRotation(90)
                self.plot_widget.addItem(text)
                count += 1

        unit = self.signal_units.get(self.current_signal, "")
        self.plot_widget.setTitle(
            f"{self.current_signal} (Visualization Rate: {self.sampling_rate} Hz)",
            color="black", size="12pt"
        )
        self.plot_widget.setLabel('left', f"{self.current_signal} ({unit})", color="black")
        self.plot_widget.setLabel('bottom', 'Time (s)', color="black")

    def on_mouse_move(self, event):
        pos = event[0]
        if self.plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = self.plot_widget.getViewBox().mapSceneToView(pos)
            x = mouse_point.x()
            if hasattr(self, "current_df") and not self.current_df.empty:
                df = self.current_df
                idx = (np.abs(df['relative_time_s'] - x)).idxmin()
                x_val = df['relative_time_s'].iloc[idx]
                y_val = df[self.current_signal].iloc[idx]
            else:
                x_val, y_val = mouse_point.x(), mouse_point.y()

            self.vLine.setPos(x_val)
            self.hLine.setPos(y_val)

            unit = self.signal_units.get(self.current_signal, "")
            self.tooltip.setHtml(
                f"<span style='color:black;'>Time: {x_val:.2f}s<br>Value: {y_val:.3f} {unit}</span>"
            )
            self.tooltip.setPos(x_val, y_val)

    def on_signal_change(self, signal):
        self.current_signal = signal
        self.update_plot()

    def on_sampling_rate_change(self, rate):
        self.sampling_rate = rate
        self.update_plot()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = TimeSeriesViewer()
    viewer.show()
    sys.exit(app.exec_())
