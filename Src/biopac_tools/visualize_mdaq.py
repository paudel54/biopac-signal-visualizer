# mdaq_timeseries_viewer.py
import sys, os, logging
from pathlib import Path
import numpy as np
import pandas as pd

# macOS Qt quirk
if sys.platform == "darwin":
    os.environ["QT_MAC_WANTS_LAYER"] = "1"

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFileDialog, QLabel, QComboBox, QSpinBox, QPushButton, QMessageBox, QCheckBox
)
from PyQt5.QtCore import Qt, QPropertyAnimation
import pyqtgraph as pg
#Saving plots to disks. 
from pyqtgraph.exporters import ImageExporter,SVGExporter

# --- CHANGE: Custom axis with smart tick formatting (Example 2) ---
class PlainAxis(pg.AxisItem):
    """A custom axis that displays values in plain decimal notation with smart precision."""
    def tickStrings(self, values, scale, spacing):
        out = []
        for v in values:
            # Choose decimals based on spacing between ticks
            if spacing >= 1:
                s = f"{v:.0f}"
            elif spacing >= 0.1:
                s = f"{v:.1f}"
            elif spacing >= 0.01:
                s = f"{v:.2f}"
            elif spacing >= 0.001:
                s = f"{v:.3f}"
            elif spacing >= 0.0001:
                s = f"{v:.4f}"
            else:
                s = f"{v:.5f}"
            out.append(s)
        return out
# -----------------------------------------------------------------

# Columns your mDAQ CSV uses
NUMERIC_COLS_HINT = [
    "ecg","eda","ir","red",
    "acc_x","acc_y","acc_z",
    "gyr_x","gyr_y","gyr_z",
    "batt%","relative_humidity","ambient_temp","body_temp"
]
SENTINELS = {-99, -99.0, ""}

def setup_logger(log_path: Path):
    log = logging.getLogger("mdaq_viewer")
    log.setLevel(logging.INFO)
    for h in list(log.handlers):
        log.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch = logging.StreamHandler(sys.stdout); ch.setFormatter(fmt)
    log.addHandler(ch)
    try:
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8"); fh.setFormatter(fmt)
        log.addHandler(fh)
        log.info(f"Logging to: {log_path}")
    except Exception as e:
        log.warning(f"Could not create log file: {e}")
    return log

class TimeSeriesViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("mDAQ Time Series Viewer")
        self.setGeometry(80, 80, 1280, 760)

        # Pick CSV first so we can place the log next to it
        csv_path, _ = QFileDialog.getOpenFileName(self, "Select merged mDAQ CSV", "", "CSV Files (*.csv)")
        if not csv_path:
            print("No file selected. Exiting.")
            sys.exit(1)
        self.csv_path = Path(csv_path)
        self.log = setup_logger(self.csv_path.with_suffix(".viewer.log"))
        self.log.info(f"Selected CSV: {self.csv_path}")

        # Defaults
        self.current_signal = None
        self.target_rate = 30  # Hz (for visualization)
        self.pan_mode = False
        self.last_step_logged = None

        # Units (adjust as needed)
        self.units = {
            "ecg": "mV", "eda": "µS",
            "ir": "a.u.", "red": "a.u.",
            "acc_x": "g", "acc_y": "g", "acc_z": "g",
            "gyr_x": "°/s", "gyr_y": "°/s", "gyr_z": "°/s",
            "batt%": "%", "relative_humidity": "%", "ambient_temp": "°C", "body_temp": "°C"
        }

        # Load data
        try:
            self.load_data(self.csv_path)
        except Exception as e:
            QMessageBox.critical(self, "Load error", f"Failed to load CSV:\n{type(e).__name__}: {e}")
            raise

        if not self.current_signal:
            self.current_signal = self.pick_default_signal()

        # UI
        self.init_ui()
        self.update_plot()

    # ------------- data -------------
    def load_data(self, path: Path):
        self.log.info("Loading CSV...")
        df = pd.read_csv(path, low_memory=False)
        self.log.info(f"CSV shape: {df.shape}; columns: {list(df.columns)}")

        if "timestamp_ms" not in df.columns:
            raise ValueError("CSV must contain 'timestamp_ms'.")
        if "label" not in df.columns:
            df["label"] = ""

        # Cast numerics & (intentionally) DO NOT replace -99 with NaN
        for c in df.columns:
            if c in ("timestamp_ms", "label"):  # timestamp_ms stays numeric; label stays string
                continue
            df[c] = pd.to_numeric(df[c], errors="coerce")
            if c in NUMERIC_COLS_HINT:
                before = df[c].isna().sum()

                # Intentionally left commented to preserve -99 in data
                # df.loc[df[c].isin(SENTINELS), c] = np.nan

                after = df[c].isna().sum()
                if after > before:
                    self.log.info(f"Sentinel→NaN in '{c}': {after-before} cells")

        # Ensure timestamp is int64-like
        df["timestamp_ms"] = pd.to_numeric(df["timestamp_ms"], errors="coerce").astype("Int64")
        if df["timestamp_ms"].isna().any():
            bad = int(df["timestamp_ms"].isna().sum())
            self.log.warning(f"Dropping {bad} rows with bad timestamp_ms")
            df = df.dropna(subset=["timestamp_ms"])
            df["timestamp_ms"] = df["timestamp_ms"].astype(np.int64)

        # Relative time
        t0 = int(df["timestamp_ms"].iloc[0])
        df["relative_time_s"] = (df["timestamp_ms"] - t0) / 1000.0

        # Clean labels: keep only meaningful labels; collapse consecutive duplicates (transitions)
        lab = df["label"]
        mask = (~lab.isna()) & (lab.astype(str).str.strip() != "") & (lab.astype(str).str.lower() != "nan")
        label_rows = df.loc[mask, ["timestamp_ms", "relative_time_s", "label"]].copy()
        self.log.info(f"Label rows (raw cleaned): {len(label_rows)}")
        if not label_rows.empty:
            # Keep only transitions
            label_rows["__prev__"] = label_rows["label"].shift(fill_value="__START__")
            label_rows = label_rows[label_rows["label"] != label_rows["__prev__"]].drop(columns="__prev__")
        self.label_rows = label_rows.reset_index(drop=True)
        self.log.info(f"Label rows (transitions): {len(self.label_rows)}")

        # Time span & source rate
        t_first = int(df["timestamp_ms"].iloc[0])
        t_last  = int(df["timestamp_ms"].iloc[-1])
        diffs = np.diff(df["timestamp_ms"].astype(np.int64).values)
        diffs = diffs[diffs > 0]
        med_ms = float(np.median(diffs)) if diffs.size else np.nan
        self.source_hz = (1000.0 / med_ms) if med_ms and med_ms > 0 else 100.0
        self.log.info(f"Time range: {t_first} → {t_last} ms (Δ={(t_last-t_first)/1000.0:.2f}s); median dt≈{med_ms:.2f} ms → src≈{self.source_hz:.2f} Hz")

        self.data = df

    def pick_default_signal(self):
        for c in ["ecg","eda","ir","red","acc_x","gyr_x","ambient_temp","batt%"]:
            if c in self.data.columns and pd.api.types.is_numeric_dtype(self.data[c]):
                self.log.info(f"Default signal: {c}")
                return c
        for c in self.data.columns:
            if c not in ("timestamp_ms","relative_time_s","label") and pd.api.types.is_numeric_dtype(self.data[c]):
                self.log.info(f"Default numeric signal: {c}")
                return c
        return "ecg"

    # ------------- UI -------------
    def init_ui(self):
        widget = QWidget()
        outer = QVBoxLayout()

        # Controls row
        controls = QHBoxLayout()

         #Adding save image button 
        self.save_btn = QPushButton("Save Image")
        self.save_btn.setStyleSheet("background-color: #1976d2; color: white; font-weight: bold;")
        self.save_btn.clicked.connect(self.save_image)
        controls.addWidget(self.save_btn)

        controls.addWidget(QLabel("Signal:"))
        self.signal_box = QComboBox()
        numeric_cols = [c for c in self.data.columns
                        if c not in ("timestamp_ms","relative_time_s","label")
                        and pd.api.types.is_numeric_dtype(self.data[c])]
        self.signal_box.addItems(sorted(numeric_cols))
        if self.current_signal in numeric_cols:
            self.signal_box.setCurrentText(self.current_signal)
        self.signal_box.currentTextChanged.connect(self.on_signal_change)
        controls.addWidget(self.signal_box)

        controls.addSpacing(12)
        controls.addWidget(QLabel("Visualization rate (Hz):"))
        self.rate_box = QSpinBox()
        self.rate_box.setRange(5, 2000)
        self.rate_box.setValue(self.target_rate)
        self.rate_box.valueChanged.connect(self.on_rate_change)
        controls.addWidget(self.rate_box)

        controls.addSpacing(12)
        self.no_ds_cb = QCheckBox("No downsampling")
        self.no_ds_cb.stateChanged.connect(lambda _: self.update_plot())
        controls.addWidget(self.no_ds_cb)

        self.show_labels_cb = QCheckBox("Show labels")
        self.show_labels_cb.setChecked(True)
        self.show_labels_cb.stateChanged.connect(lambda _: self.update_plot())
        controls.addWidget(self.show_labels_cb)

        controls.addStretch(1)

        self.pan_btn = QPushButton("Activate pan mode")
        self.pan_btn.setStyleSheet("background-color: green; color: white; font-weight: bold;")
        self.pan_btn.clicked.connect(self.toggle_pan_mode)
        controls.addWidget(self.pan_btn)

        outer.addLayout(controls)

        # Use the custom PlainAxis on the left
        left_axis = PlainAxis(orientation='left')
        self.plot = pg.PlotWidget(axisItems={'left': left_axis})
        self.plot.setBackground("#fafafa")
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setClipToView(True)
        self.plot.setDownsampling(auto=True, mode="peak")
        self.plot.getAxis('bottom').enableAutoSIPrefix(False)  # Keep bottom axis in plain seconds
        self.plot.getAxis("left").setTextPen("black")
        self.plot.getAxis("bottom").setTextPen("black")
        self.plot.getViewBox().setMouseMode(pg.ViewBox.RectMode)
        self.plot.scene().sigMouseClicked.connect(self.on_mouse_double_reset)

        # Crosshair + tooltip
        self.vLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('gray', style=pg.QtCore.Qt.DashLine))
        self.hLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('gray', style=pg.QtCore.Qt.DashLine))
        self.plot.addItem(self.vLine, ignoreBounds=True)
        self.plot.addItem(self.hLine, ignoreBounds=True)
        self.tooltip = pg.TextItem("", anchor=(0, 1), color="black")
        self.plot.addItem(self.tooltip)
        self.proxy = pg.SignalProxy(self.plot.scene().sigMouseMoved, rateLimit=30, slot=self.on_mouse_move)

        outer.addWidget(self.plot)

        # Toast + status
        self.toast = QLabel("", self.plot)
        self.toast.setVisible(False)
        self.status = self.statusBar()
        self.status.showMessage("Ready")

        widget.setLayout(outer)
        self.setCentralWidget(widget)

    # ------------- helpers -------------
    def toast_msg(self, text, bg="lightyellow", ms=2200):
        self.toast.setText(text)
        self.toast.setStyleSheet(
            f"background-color:{bg}; color:black; padding:6px 10px; border-radius:6px; font-weight:bold;"
        )
        self.toast.adjustSize()
        pad = 6
        self.toast.move(self.plot.width() - self.toast.width() - pad, pad)
        self.toast.raise_()
        self.toast.setWindowOpacity(1.0)
        self.toast.setVisible(True)
        self.anim = QPropertyAnimation(self.toast, b"windowOpacity")
        self.anim.setDuration(ms)
        self.anim.setStartValue(1.0)
        self.anim.setEndValue(0.0)
        self.anim.finished.connect(lambda: self.toast.setVisible(False))
        self.anim.start()

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
            exporter = SVGExporter(self.plot.getPlotItem())
        else:
            exporter = ImageExporter(self.plot.getPlotItem())
            exporter.parameters()['width'] = 2560  
        # ----------------------------------

        try:
            exporter.export(fname)
            self.toast_msg(f"Saved: {os.path.basename(fname)}", bg="#c8e6c9", ms=2200)
        except Exception as e:
            self.toast_msg(f"Save failed: {e}", bg="#ffcdd2", ms=2200)
    # -----------------------------------------

    def toggle_pan_mode(self):
        self.pan_mode = not self.pan_mode
        if self.pan_mode:
            self.plot.getViewBox().setMouseMode(pg.ViewBox.PanMode)
            self.pan_btn.setText("Activate zoom mode")
            self.pan_btn.setStyleSheet("background-color: goldenrod; color: black; font-weight: bold;")
            self.toast_msg("Pan mode active (drag to move)", "lightgreen")
            self.log.info("Pan mode ON")
        else:
            self.plot.getViewBox().setMouseMode(pg.ViewBox.RectMode)
            self.pan_btn.setText("Activate pan mode")
            self.pan_btn.setStyleSheet("background-color: green; color: white; font-weight: bold;")
            self.toast_msg("Zoom mode active (drag to zoom; double-click to reset)")
            self.log.info("Zoom (rect) mode ON")

    def on_mouse_double_reset(self, ev):
        if ev.double():
            self.plot.enableAutoRange()
            self.log.info("View reset")

    def on_signal_change(self, sig):
        self.current_signal = sig
        self.log.info(f"Signal → {sig}")
        self.update_plot()

    def on_rate_change(self, val):
        self.target_rate = val
        self.log.info(f"View rate → {val} Hz")
        self.update_plot()

    def downsample_for_view(self, df):
        """Row-stride to ~target_rate, always keep label rows."""
        if len(df) < 3:
            return df
        if self.no_ds_cb.isChecked():
            return (pd.concat([df, self.label_rows])
                      .drop_duplicates(subset="timestamp_ms")
                      .sort_values("timestamp_ms"))

        src = max(1.0, self.source_hz)
        if self.target_rate >= src:
            step = 1
            out = df
        else:
            step = max(1, int(round(src / float(self.target_rate))))
            out = df.iloc[::step, :]

        out = (pd.concat([out, self.label_rows])
                 .drop_duplicates(subset="timestamp_ms")
                 .sort_values("timestamp_ms"))

        if step != self.last_step_logged:
            self.log.info(f"Downsampling step = {step} (src≈{src:.1f} Hz → target {self.target_rate} Hz)")
            self.last_step_logged = step
        return out

    def robust_yrange(self, s):
        s = pd.to_numeric(s, errors="coerce").dropna()
        if s.empty:
            return (-1, 1)
        q1, q99 = np.nanpercentile(s, [1, 99])
        if not np.isfinite(q1) or not np.isfinite(q99) or np.isclose(q1, q99):
            pad = 1.0 if not np.isfinite(q99) or q99 == 0 else abs(q99) * 0.2
            return (float(q1) - pad, float(q99) + pad)
        return (float(q1), float(q99))

    # ------------- plot -------------
    def update_plot(self):
        try:
            self.plot.clear()
            df = self.downsample_for_view(self.data).copy()
            self.current_df = df

            sig = self.current_signal
            if sig not in df.columns:
                self.log.warning(f"Signal '{sig}' missing.")
                return

            pen = pg.mkPen(color="dodgerblue", width=1.5)
            self.plot.plot(df["relative_time_s"], df[sig], pen=pen)

            ymin, ymax = self.robust_yrange(df[sig])
            if ymin == ymax:
                ymin -= 1.0; ymax += 1.0
            self.plot.setYRange(ymin, ymax, padding=0)

            # Draw labels (optional, smart-capped)
            if self.show_labels_cb.isChecked() and not self.label_rows.empty:
                try:
                    xmin, xmax = self.plot.getViewBox().viewRange()[0]
                    vis = self.label_rows[
                        (self.label_rows["relative_time_s"] >= xmin) &
                        (self.label_rows["relative_time_s"] <= xmax)
                    ]
                except Exception:
                    vis = self.label_rows

                MAX_LABELS = 300
                lbl_df = vis
                if len(lbl_df) > MAX_LABELS:
                    step = int(np.ceil(len(lbl_df) / MAX_LABELS))
                    lbl_df = lbl_df.iloc[::step, :]
                    self.log.info(f"Label draw capped: {len(vis)} visible → {len(lbl_df)} drawn")

                y_span = (ymax - ymin) if ymax > ymin else 1.0
                y_base = ymax + y_span * 0.06
                stagger = y_span * 0.05
                signs = [1, -1, 2, -2]
                for k, (_, row) in enumerate(lbl_df.iterrows()):
                    x = float(row["relative_time_s"]); lbl = str(row["label"])
                    self.plot.addItem(pg.InfiniteLine(pos=x, angle=90,
                                                      pen=pg.mkPen('r', style=pg.QtCore.Qt.DashLine)))
                    t = pg.TextItem(lbl, color='r', anchor=(0, 0))
                    t.setRotation(90)
                    t.setPos(x, y_base + signs[k % len(signs)] * stagger)
                    self.plot.addItem(t)
                label_count = len(lbl_df)
            else:
                label_count = 0

            unit = self.units.get(sig, "")
            self.plot.setTitle(f"{sig}  |  view≈{int(self.target_rate)} Hz (source≈{int(self.source_hz)} Hz)",
                               color="black", size="12pt")
            self.plot.setLabel("left", f"{sig} ({unit})", color="black")
            self.plot.setLabel("bottom", "Time (s)", color="black")

            self.status.showMessage(
                f"Rows: {len(self.data)} | Labels: {len(self.label_rows)} | Source≈{self.source_hz:.1f} Hz "
                f"| View~{self.target_rate} Hz | Signal='{sig}' | y[{ymin:.3g},{ymax:.3g}] | drawn labels={label_count}"
            )
            self.log.info(f"Plotted {sig}: shown_rows={len(df)}, labels_drawn={label_count}, y=[{ymin:.3g},{ymax:.3g}]")

        except Exception as e:
            self.log.exception("Plot update failed")
            QMessageBox.critical(self, "Plot error", f"Error updating plot:\n{type(e).__name__}: {e}")

    # ------------- mouse tooltip -------------
    def on_mouse_move(self, ev):
        pos = ev[0]
        if not self.plot.sceneBoundingRect().contains(pos):
            return
        vb = self.plot.getViewBox()
        mouse_pt = vb.mapSceneToView(pos)
        x = mouse_pt.x()

        if hasattr(self, "current_df") and not self.current_df.empty:
            idx = (np.abs(self.current_df["relative_time_s"] - x)).idxmin()
            x_val = float(self.current_df["relative_time_s"].loc[idx])
            y_val = float(self.current_df[self.current_signal].loc[idx])
        else:
            x_val, y_val = mouse_pt.x(), mouse_pt.y()

        self.vLine.setPos(x_val)
        self.hLine.setPos(y_val)
        unit = self.units.get(self.current_signal, "")
        self.tooltip.setHtml(f"<span style='color:black;'>Time: {x_val:.2f}s<br>Value: {y_val:.3f} {unit}</span>")
        self.tooltip.setPos(x_val, y_val)

def main():
    app = QApplication(sys.argv)
    viewer = TimeSeriesViewer()
    viewer.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
