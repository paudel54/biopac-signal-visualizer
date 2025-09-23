# compare_biopac_mdaq_viewer.py
import sys, os, logging
from pathlib import Path
import numpy as np
import pandas as pd

# macOS Qt quirk
if sys.platform == "darwin":
    os.environ["QT_MAC_WANTS_LAYER"] = "1"

from PyQt5.QtCore import Qt, QPropertyAnimation, QTimer
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFileDialog, QLabel, QComboBox, QSpinBox, QPushButton, QMessageBox, QCheckBox
)

import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter, SVGExporter


# ---------- Shared helpers ----------
def setup_logger(log_path: Path):
    log = logging.getLogger(str(log_path))
    log.setLevel(logging.INFO)
    for h in list(log.handlers):
        log.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch = logging.StreamHandler(sys.stdout); ch.setFormatter(fmt)
    log.addHandler(ch)
    try:
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8"); fh.setFormatter(fmt)
        log.addHandler(fh)
    except Exception as e:
        log.warning(f"Could not create log file: {e}")
    return log


class PlainAxis(pg.AxisItem):
    """Axis that shows plain decimals with smart precision."""
    def tickStrings(self, values, scale, spacing):
        out = []
        for v in values:
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


# ---------- One reusable pane (plot + controls) ----------
class SignalPane(QWidget):
    def __init__(self, name: str, csv_path: Path, slow_cols=None, parent=None):
        """
        name: "BIOPAC" or "mDAQ" (title used on UI + default filenames)
        csv_path: path to CSV to load
        slow_cols: columns to forward-fill (for mDAQ 1 Hz channels); can be None
        """
        super().__init__(parent)
        self.name = name
        self.csv_path = Path(csv_path)
        self.log = setup_logger(self.csv_path.with_suffix(f".{self.name.lower()}.viewer.log"))
        self.slow_cols = set(slow_cols or [])

        # defaults
        self.current_signal = None
        self.target_rate = 30
        self.last_step_logged = None

        # units (tweak as needed)
        self.units = {
            "ECG": "mV", "PPG": "a.u.", "EDA": "µS", "SKT": "°C",  # BIOPAC typical
            "ecg": "mV", "eda": "µS", "ir": "a.u.", "red": "a.u.",
            "acc_x": "g", "acc_y": "g", "acc_z": "g",
            "gyr_x": "°/s", "gyr_y": "°/s", "gyr_z": "°/s",
            "batt%": "%", "relative_humidity": "%", "ambient_temp": "°C", "body_temp": "°C"
        }

        # data
        self.load_data(self.csv_path)

        # UI
        self.build_ui()
        self.update_plot()

    # ----- data -----
    def load_data(self, path: Path):
        df = pd.read_csv(path, low_memory=False)

        if "timestamp_ms" not in df.columns:
            raise ValueError(f"{self.name}: CSV must contain 'timestamp_ms'.")

        # ensure numeric timestamps -> np.int64 (not pandas Int64Dtype)
        df["timestamp_ms"] = pd.to_numeric(df["timestamp_ms"], errors="coerce")
        bad = int(df["timestamp_ms"].isna().sum())
        if bad:
            self.log.warning(f"{self.name}: dropping {bad} rows with bad timestamp_ms")
            df = df.dropna(subset=["timestamp_ms"])
        df["timestamp_ms"] = df["timestamp_ms"].astype(np.int64)

        # ensure label column exists and is stringy
        if "label" not in df.columns:
            df["label"] = ""
        df["label"] = df["label"].astype(str)

        # cast other columns numeric when possible
        for c in df.columns:
            if c in ("timestamp_ms", "label"): 
                continue
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # relative time (seconds from first sample)
        t0 = int(df["timestamp_ms"].iloc[0])
        df["relative_time_s"] = (df["timestamp_ms"] - t0) / 1000.0

        # label transitions only
        lab = df["label"]
        mask = (lab.str.strip() != "") & (lab.str.lower() != "nan")
        lbl = df.loc[mask, ["timestamp_ms", "relative_time_s", "label"]].copy()
        if not lbl.empty:
            lbl["__prev__"] = lbl["label"].shift(fill_value="__START__")
            lbl = lbl[lbl["label"] != lbl["__prev__"]].drop(columns="__prev__")
        lbl["timestamp_ms"] = lbl["timestamp_ms"].astype(np.int64)
        self.label_rows = lbl.reset_index(drop=True)

        # estimate source Hz
        diffs = np.diff(df["timestamp_ms"].values.astype(np.int64))
        diffs = diffs[diffs > 0]
        med_ms = float(np.median(diffs)) if diffs.size else np.nan
        self.source_hz = (1000.0 / med_ms) if med_ms and med_ms > 0 else 100.0
        self.log.info(f"{self.name}: source≈{self.source_hz:.1f} Hz, rows={len(df)}")

        self.data = df

        # pick default signal
        if self.current_signal is None:
            for c in df.columns:
                if c not in ("timestamp_ms", "relative_time_s", "label") and pd.api.types.is_numeric_dtype(df[c]):
                    self.current_signal = c
                    break

    # ----- UI -----
    def build_ui(self):
        outer = QVBoxLayout(self)

        # controls
        controls = QHBoxLayout()
        title = QLabel(f"<b>{self.name}</b>")
        controls.addWidget(title)

        controls.addSpacing(10)
        controls.addWidget(QLabel("Select signal"))
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
        controls.addWidget(QLabel("Sampling rate"))
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

        self.save_btn = QPushButton("Save Image")
        self.save_btn.clicked.connect(self.save_image)  # <-- FIX: bound to this pane
        controls.addWidget(self.save_btn)

        outer.addLayout(controls)

        # plot
        left_axis = PlainAxis(orientation='left')
        self.plot = pg.PlotWidget(axisItems={'left': left_axis})
        self.plot.setBackground("#f7f7f7")
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setClipToView(True)
        self.plot.setDownsampling(auto=True, mode="peak")
        self.plot.getAxis("left").setTextPen("black")
        self.plot.getAxis("bottom").setTextPen("black")
        self.plot.getViewBox().setMouseMode(pg.ViewBox.RectMode)
        outer.addWidget(self.plot)

        # crosshair + tooltip
        self.vLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('gray', style=pg.QtCore.Qt.DashLine))
        self.hLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('gray', style=pg.QtCore.Qt.DashLine))
        self.plot.addItem(self.vLine, ignoreBounds=True)
        self.plot.addItem(self.hLine, ignoreBounds=True)
        self.tooltip = pg.TextItem("", anchor=(0, 1), color="black")
        self.plot.addItem(self.tooltip)
        self.proxy = pg.SignalProxy(self.plot.scene().sigMouseMoved, rateLimit=30, slot=self.on_mouse_move)
        self.plot.scene().sigMouseClicked.connect(self.on_mouse_double_reset)

    # ----- saving just this pane -----
    def save_image(self):
        default_name = f"{self.name}_{self.current_signal}_plot"
        fname, ffilter = QFileDialog.getSaveFileName(
            self, "Save Plot Image", f"{default_name}.png",
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;SVG Vector (*.svg)"
        )
        if not fname:
            return
        if '.' not in os.path.basename(fname):
            if "PNG" in ffilter:   fname += ".png"
            elif "JPEG" in ffilter: fname += ".jpg"
            elif "SVG" in ffilter:  fname += ".svg"
            else:                    fname += ".png"

        if fname.lower().endswith(".svg"):
            exporter = SVGExporter(self.plot.getPlotItem())
        else:
            exporter = ImageExporter(self.plot.getPlotItem())
            exporter.parameters()['width'] = 2560

        exporter.export(fname)

    # ----- resampling / plotting -----
    def downsample_for_view(self, df):
        """Resample onto a uniform grid (ms) at target_rate; keep dtype int64; forward-fill slow columns if configured."""
        if len(df) < 3:
            return df.sort_values("timestamp_ms")

        if self.no_ds_cb.isChecked():
            return df.sort_values("timestamp_ms")

        t0 = int(df["timestamp_ms"].iloc[0])
        t1 = int(df["timestamp_ms"].iloc[-1])
        step_ms = max(1, int(round(1000.0 / float(self.target_rate))))
        grid = pd.DataFrame({"timestamp_ms": np.arange(t0, t1 + 1, step_ms, dtype=np.int64)})

        src = df.sort_values("timestamp_ms").copy()
        src["timestamp_ms"] = src["timestamp_ms"].astype(np.int64)
        tol = step_ms
        out = pd.merge_asof(grid, src, on="timestamp_ms", direction="nearest", tolerance=tol)

        # forward/backward fill configured slow cols (e.g., mDAQ temps & RH)
        for c in self.slow_cols:
            if c in out.columns:
                out.loc[out[c].isin([-99, -99.0]), c] = np.nan
                out[c] = out[c].ffill().bfill()

        # snap labels too
        if hasattr(self, "label_rows") and not self.label_rows.empty:
            labels_src = self.label_rows.sort_values("timestamp_ms").copy()
            labels_src["timestamp_ms"] = labels_src["timestamp_ms"].astype(np.int64)
            labels_grid = pd.merge_asof(
                out[["timestamp_ms"]], labels_src,
                on="timestamp_ms", direction="nearest", tolerance=tol
            )
            if "label" in out.columns and "label" in labels_grid.columns:
                out["label"] = labels_grid["label"].combine_first(out["label"])
            else:
                out["label"] = labels_grid["label"]

        out["relative_time_s"] = (out["timestamp_ms"] - out["timestamp_ms"].iloc[0]) / 1000.0

        if step_ms != self.last_step_logged:
            self.log.info(f"{self.name}: resample step = {step_ms} ms (view≈{self.target_rate} Hz)")
            self.last_step_logged = step_ms

        return out

    def robust_yrange(self, s):
        s = pd.to_numeric(s, errors="coerce")
        s = s[~s.isin([-99, -99.0])].dropna()
        if s.empty:
            return (-1, 1)
        q1, q99 = np.nanpercentile(s, [1, 99])
        if not np.isfinite(q1) or not np.isfinite(q99) or np.isclose(q1, q99):
            pad = 1.0 if not np.isfinite(q99) or q99 == 0 else abs(q99) * 0.2
            return (float(q1) - pad, float(q99) + pad)
        return (float(q1), float(q99))

    def update_plot(self):
        self.plot.clear()
        df = self.downsample_for_view(self.data).copy()
        self.current_df = df

        sig = self.current_signal
        if sig not in df.columns:
            self.log.warning(f"{self.name}: signal '{sig}' missing")
            return

        pen = pg.mkPen(color="dodgerblue", width=1.5)
        self.plot.plot(df["relative_time_s"], df[sig], pen=pen)

        ymin, ymax = self.robust_yrange(df[sig])
        if ymin == ymax:
            ymin -= 1.0; ymax += 1.0
        self.plot.setYRange(ymin, ymax, padding=0)

        # labels (optional)
        label_count = 0
        if self.show_labels_cb.isChecked() and hasattr(self, "label_rows") and not self.label_rows.empty:
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

        unit = self.units.get(sig, "")
        self.plot.setTitle(f"{self.name} — {sig} | view≈{int(self.target_rate)} Hz (src≈{int(self.source_hz)} Hz)",
                           color="black", size="12pt")
        self.plot.setLabel("left", f"{sig} ({unit})", color="black")
        self.plot.setLabel("bottom", "Time (s)", color="black")

        self.log.info(f"{self.name}: plotted '{sig}' rows_shown={len(df)} labels_drawn={label_count}")

    # ----- callbacks -----
    def on_signal_change(self, sig):
        self.current_signal = sig
        self.update_plot()

    def on_rate_change(self, val):
        self.target_rate = val
        self.update_plot()

    def on_mouse_double_reset(self, ev):
        if ev.double():
            self.plot.enableAutoRange()

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


# ---------- Main window with two panes + Save Both ----------
class CompareViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BIOPAC vs mDAQ — Time Series Comparison")
        self.setGeometry(60, 60, 1280, 860)

        # Select files
        biopac_path, _ = QFileDialog.getOpenFileName(self, "Select BIOPAC CSV", "", "CSV Files (*.csv)")
        if not biopac_path:
            sys.exit("No BIOPAC file selected.")
        mdaq_path, _ = QFileDialog.getOpenFileName(self, "Select mDAQ CSV", "", "CSV Files (*.csv)")
        if not mdaq_path:
            sys.exit("No mDAQ file selected.")

        # Layout
        central = QWidget(); self.setCentralWidget(central)
        outer = QVBoxLayout(central)

        # BIOPAC pane (no slow cols to fill)
        self.biopac = SignalPane("BIOPAC", Path(biopac_path), slow_cols=[])
        outer.addWidget(self.biopac)

        # mDAQ pane (forward-fill 1 Hz channels)
        slow = ["relative_humidity", "ambient_temp", "body_temp"]
        self.mdaq = SignalPane("mDAQ", Path(mdaq_path), slow_cols=slow)
        outer.addWidget(self.mdaq)

        # bottom row: Save Both
        bottom = QHBoxLayout()
        bottom.addStretch(1)
        self.save_both_btn = QPushButton("Save Both (same window)")
        self.save_both_btn.clicked.connect(self.save_both)
        bottom.addWidget(self.save_both_btn)
        outer.addLayout(bottom)

        # toast overlay (one for the whole window)
        self.toast = QLabel("", self)
        self.toast.setVisible(False)
        self.toast.setStyleSheet(
            "background-color: #c8e6c9; color: black; padding: 6px 10px; "
            "border-radius: 6px; font-weight: bold;"
        )
        self.toast_anim = None

    # robust toast that ALWAYS disappears
    def show_toast(self, text, bg="#c8e6c9", ms=2200):
        self.toast.setText(text)
        self.toast.setStyleSheet(
            f"background-color: {bg}; color: black; padding: 6px 10px; "
            f"border-radius: 6px; font-weight: bold;"
        )
        self.toast.adjustSize()
        pad = 8
        self.toast.move(self.width() - self.toast.width() - pad, pad)
        self.toast.setWindowOpacity(1.0)
        self.toast.show()

        # stop any prior animation
        if self.toast_anim is not None:
            self.toast_anim.stop()
            self.toast_anim.deleteLater()
            self.toast_anim = None

        self.toast_anim = QPropertyAnimation(self.toast, b"windowOpacity", self)
        self.toast_anim.setDuration(ms)
        self.toast_anim.setStartValue(1.0)
        self.toast_anim.setEndValue(0.0)
        self.toast_anim.finished.connect(self.toast.hide)
        self.toast_anim.start()

        # belt & suspenders: force hide even if animation fails
        QTimer.singleShot(ms + 300, self.toast.hide)

    def save_both(self):
        # Pick a base filename; we derive two files: *_BIOPAC.png and *_mDAQ.png
        base, _ = QFileDialog.getSaveFileName(
            self, "Save Both Plots (base name)", "comparison_biopac_mdaq.png",
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;SVG Vector (*.svg)"
        )
        if not base:
            return

        # Normalize base (strip extension for our two files)
        stem = Path(base)
        stem = stem.with_suffix("")  # drop extension
        suffix = Path(base).suffix.lower() or ".png"
        if suffix not in (".png", ".jpg", ".jpeg", ".svg"):
            suffix = ".png"

        f_biopac = stem.with_name(stem.name + "_BIOPAC").with_suffix(suffix)
        f_mdaq   = stem.with_name(stem.name + "_mDAQ").with_suffix(suffix)

        # Export BIOPAC
        if suffix == ".svg":
            exp1 = SVGExporter(self.biopac.plot.getPlotItem())
        else:
            exp1 = ImageExporter(self.biopac.plot.getPlotItem())
            exp1.parameters()['width'] = 2560
        exp1.export(str(f_biopac))

        # Export mDAQ
        if suffix == ".svg":
            exp2 = SVGExporter(self.mdaq.plot.getPlotItem())
        else:
            exp2 = ImageExporter(self.mdaq.plot.getPlotItem())
            exp2.parameters()['width'] = 2560
        exp2.export(str(f_mdaq))

        self.show_toast(f"Saved: {f_biopac.name} + {f_mdaq.name}")

def main():
    app = QApplication(sys.argv)
    viewer = CompareViewer()
    viewer.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
