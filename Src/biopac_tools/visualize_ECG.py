# ecg_viewer_50hz.py
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg

pg.setConfigOptions(antialias=True)

class ECGViewer(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ECG Plotter • CSV Loader (50 Hz default)")
        self.resize(1150, 680)

        # --- Central UI
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        vbox = QtWidgets.QVBoxLayout(central)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(6)

        # --- Toolbar
        toolbar = QtWidgets.QHBoxLayout()
        self.openBtn = QtWidgets.QPushButton("Open CSV…")
        self.resetBtn = QtWidgets.QPushButton("Reset View")

        self.resampleChk = QtWidgets.QCheckBox("Resample for display")
        self.resampleChk.setChecked(True)

        self.rateSpin = QtWidgets.QSpinBox()
        self.rateSpin.setRange(1, 2000)
        self.rateSpin.setValue(50)  # default 50 Hz
        self.rateSpin.setSuffix(" Hz")
        self.rateSpin.setToolTip("Display sampling rate (interpolated).")

        self.labelsChk = QtWidgets.QCheckBox("Show labels")
        self.labelsChk.setChecked(True)

        toolbar.addWidget(self.openBtn)
        toolbar.addWidget(self.resetBtn)
        toolbar.addSpacing(20)
        toolbar.addWidget(self.resampleChk)
        toolbar.addWidget(self.rateSpin)
        toolbar.addStretch(1)
        toolbar.addWidget(self.labelsChk)
        vbox.addLayout(toolbar)

        # --- Plot
        self.plot = pg.PlotWidget(background="w")
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel("bottom", "Time", units="s")
        self.plot.setLabel("left", "ECG", units="(a.u.)")
        vbox.addWidget(self.plot, stretch=1)

        # --- Status
        self.status = QtWidgets.QLabel("Select a CSV file to begin…")
        vbox.addWidget(self.status)

        # --- Data + graphics handles
        self.data_df = None
        self.curve = self.plot.plot([], [], pen=pg.mkPen("#1f77b4", width=2))
        self.scatter = None
        self.label_items = []

        # Crosshair
        self.vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen((120,120,120), width=1))
        self.hline = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen((120,120,120), width=1))
        self.plot.addItem(self.vline, ignoreBounds=True)
        self.plot.addItem(self.hline, ignoreBounds=True)
        self.proxy = pg.SignalProxy(self.plot.scene().sigMouseMoved, rateLimit=60, slot=self._mouse_moved)

        # Signals
        self.openBtn.clicked.connect(self.open_csv)
        self.resetBtn.clicked.connect(self.reset_view)
        self.labelsChk.stateChanged.connect(self.refresh_plot)
        self.resampleChk.stateChanged.connect(self.refresh_plot)
        self.rateSpin.valueChanged.connect(self.refresh_plot)

        # Pop up the file dialog on start (Windows-friendly)
        QtCore.QTimer.singleShot(0, self.open_csv)

    # ---------- File I/O ----------
    def open_csv(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open ECG CSV (ECG,Time,Label)",
            str(Path.home()),
            "CSV files (*.csv);;All files (*)",
        )
        if not path:
            return
        try:
            df = pd.read_csv(path)
            self.load_dataframe(df)
            self.status.setText(f"Loaded: {Path(path).name}  • Rows: {len(df)}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load CSV:\n{e}")

    def load_dataframe(self, df: pd.DataFrame):
        # Expect columns: ECG, Time, Label (case-insensitive)
        cols = {c.lower(): c for c in df.columns}
        if "ecg" not in cols or "time" not in cols:
            raise ValueError("CSV must include 'ECG' and 'Time' columns.")
        ECG_col = cols["ecg"]
        Time_col = cols["time"]
        Label_col = cols.get("label")

        # Coerce types
        df = df.copy()
        df[ECG_col] = pd.to_numeric(df[ECG_col], errors="coerce")
        df[Time_col] = pd.to_numeric(df[Time_col], errors="coerce")

        if Label_col is None:
            df["Label"] = np.nan
            Label_col = "Label"
        else:
            df[Label_col] = df[Label_col].replace("", np.nan)

        df = df.dropna(subset=[ECG_col, Time_col]).reset_index(drop=True)
        df = df.sort_values(Time_col).reset_index(drop=True)

        # Normalize column names
        df = df.rename(columns={ECG_col: "ECG", Time_col: "Time", Label_col: "Label"})
        self.data_df = df

        self.refresh_plot()
        self.reset_view()

    # ---------- Plotting ----------
    def refresh_plot(self):
        self._clear_labels()
        if self.data_df is None or self.data_df.empty:
            self.curve.setData([], [])
            return

        t = self.data_df["Time"].to_numpy()
        y = self.data_df["ECG"].to_numpy()

        # Optional resampling for display
        if self.resampleChk.isChecked() and len(t) > 1:
            disp_hz = max(1, int(self.rateSpin.value()))
            dt = 1.0 / float(disp_hz)
            t0, t1 = t[0], t[-1]
            # Ensure strictly increasing target grid; include endpoint
            t_new = np.arange(t0, t1 + 0.5 * dt, dt, dtype=float)
            # Interpolate; if t isn't strictly increasing, enforce it
            # (already sorted, but guard against duplicates)
            uniq_mask = np.concatenate(([True], np.diff(t) > 0))
            t_u = t[uniq_mask]
            y_u = y[uniq_mask]
            y_new = np.interp(t_new, t_u, y_u)
            self.curve.setData(t_new, y_new, pen=pg.mkPen("#1f77b4", width=2))
        else:
            self.curve.setData(t, y, pen=pg.mkPen("#1f77b4", width=2))

        # Labels
        if self.labelsChk.isChecked():
            self._draw_labels(self.data_df)

    def _clear_labels(self):
        for it in self.label_items:
            try:
                self.plot.removeItem(it)
            except Exception:
                pass
        self.label_items = []

    def _draw_labels(self, df_full: pd.DataFrame):
        labs = df_full.dropna(subset=["Label"])
        if labs.empty:
            return

        times = labs["Time"].to_numpy()
        # Interpolate ECG at label times for a neat dot marker
        ecg_at = np.interp(times, df_full["Time"], df_full["ECG"])

        scatter = pg.ScatterPlotItem(times, ecg_at, size=8, brush=pg.mkBrush(200, 50, 50, 150))
        self.plot.addItem(scatter)
        self.label_items.append(scatter)

        # Vertical lines + text
        vb = self.plot.getViewBox()
        y_min, y_max = vb.viewRange()[1]
        y_text = y_max

        for t, txt in zip(times, labs["Label"].astype(str)):
            vline = pg.InfiniteLine(pos=t, angle=90, pen=pg.mkPen((200, 50, 50, 160), width=1.5))
            self.plot.addItem(vline)
            self.label_items.append(vline)

            text_item = pg.TextItem(txt, color=(50, 50, 50), anchor=(0.5, 1.0))
            text_item.setPos(t, y_text)
            self.plot.addItem(text_item)
            self.label_items.append(text_item)

    def reset_view(self):
        if self.data_df is None or self.data_df.empty:
            return
        t = self.data_df["Time"].to_numpy()
        y = self.data_df["ECG"].to_numpy()
        if len(t) == 0:
            return
        xpad = 0.02 * (t.max() - t.min() if len(t) > 1 else 1.0)
        ypad = 0.05 * (np.nanmax(y) - np.nanmin(y) if len(y) > 1 else 1.0)
        self.plot.setXRange(t.min() - xpad, t.max() + xpad)
        self.plot.setYRange(np.nanmin(y) - ypad, np.nanmax(y) + ypad)

    # ---------- Interaction ----------
    def _mouse_moved(self, evt):
        if self.data_df is None:
            return
        pos = evt[0]
        if self.plot.sceneBoundingRect().contains(pos):
            mousePoint = self.plot.plotItem.vb.mapSceneToView(pos)
            x = mousePoint.x()
            y = mousePoint.y()
            self.vline.setPos(x)
            self.hline.setPos(y)
            self.status.setText(f"t = {x:.6f} s    ECG = {y:.6f}")

def main():
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    app = QtWidgets.QApplication(sys.argv)
    win = ECGViewer()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
