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
    QFileDialog, QLabel, QComboBox, QSpinBox, QPushButton, QCheckBox
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
            if spacing >= 1: s = f"{v:.0f}"
            elif spacing >= 0.1: s = f"{v:.1f}"
            elif spacing >= 0.01: s = f"{v:.2f}"
            elif spacing >= 0.001: s = f"{v:.3f}"
            elif spacing >= 0.0001: s = f"{v:.4f}"
            else: s = f"{v:.5f}"
            out.append(s)
        return out


# ---------- One reusable pane (plot + controls) ----------
class SignalPane(QWidget):
    def __init__(self, name: str, csv_path: Path, slow_cols=None, parent=None, custom_header:str = ""):
        super().__init__(parent)
        self.custom_header = custom_header.strip()
        # Canonical display names for Y-axis (same across BIOPAC & mDAQ)
        self.display_names = {
            # ECG
            "ECG": "ECG",
            "ecg": "ECG",

            # Skin temperature
            "SKT": "Skin Temp",
            "body_temp": "Body Temp",

            # EDA
            "EDA": "EDA",
            "eda": "EDA",

            # PPG / light channels
            "PPG": "PPG",
            "ir": "PPG (Infra Red)",
            "red": "PPG (Red)",

            # Motion
            "acc_x": "Acceleration",
            "acc_y": "Acceleration",
            "acc_z": "Acceleration",
            "gyr_x": "Angular rate",
            "gyr_y": "Angular rate",
            "gyr_z": "Angular rate",

            # Environment / misc.
            "relative_humidity": "Relative Humidity",
            "ambient_temp": "Ambient Temp",
            "batt%": "Battery",
        }

        self.name = name
        self.csv_path = Path(csv_path)
        self.log = setup_logger(self.csv_path.with_suffix(f".{self.name.lower()}.viewer.log"))
        self.slow_cols = set(slow_cols or [])

        self.current_signal = None
        self.target_rate = 30
        self.last_step_logged = None
        self.t0_global = None  # shared origin (ms), set from main window

        self.units = {
            "ECG": "mV", "PPG": "a.u.", "EDA": "µS", "SKT": "°C",
            "ecg": "mV", "eda": "µS", "ir": "a.u.", "red": "a.u.",
            "acc_x": "g", "acc_y": "g", "acc_z": "g",
            "gyr_x": "°/s", "gyr_y": "°/s", "gyr_z": "°/s",
            "batt%": "%", "relative_humidity": "%", "ambient_temp": "°C", "body_temp": "°C"
        }

        self.load_data(self.csv_path)
        self.build_ui()
        # don't draw yet; the main window will set the global t0, then call update_plot()

    # ----- data -----
    def load_data(self, path: Path):
        df = pd.read_csv(path, low_memory=False)

        if "timestamp_ms" not in df.columns:
            raise ValueError(f"{self.name}: CSV must contain 'timestamp_ms'.")

        df["timestamp_ms"] = pd.to_numeric(df["timestamp_ms"], errors="coerce")
        bad = int(df["timestamp_ms"].isna().sum())
        if bad:
            self.log.warning(f"{self.name}: dropping {bad} rows with bad timestamp_ms")
            df = df.dropna(subset=["timestamp_ms"])
        df["timestamp_ms"] = df["timestamp_ms"].astype(np.int64)

        # robust labels: ensure string dtype, no NaN, and ignore obvious non-labels
        if "label" not in df.columns:
            df["label"] = ""
        df["label"] = df["label"].astype("string").fillna("")
        # cast other columns
        for c in df.columns:
            if c in ("timestamp_ms", "label"):
                continue
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # temporary local relative time (will be recomputed from global t0 later)
        t0_local = int(df["timestamp_ms"].iloc[0])
        df["relative_time_s"] = (df["timestamp_ms"] - t0_local) / 1000.0

        # label transitions (ignore empty / 'nan' / 'none' / booleans-as-strings)
        lab = df["label"].str.strip()
        bad_tokens = {"", "nan", "none", "true", "false"}
        mask = ~lab.str.lower().isin(bad_tokens)
        lbl = df.loc[mask, ["timestamp_ms", "label"]].copy()
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

        if self.current_signal is None:
            for c in df.columns:
                if c not in ("timestamp_ms", "relative_time_s", "label") and pd.api.types.is_numeric_dtype(df[c]):
                    self.current_signal = c
                    break

    # set a shared origin so both panes align
    def set_global_t0(self, t0_ms: int):
        self.t0_global = int(t0_ms)
        if not self.data.empty:
            self.data["relative_time_s"] = (self.data["timestamp_ms"] - self.t0_global) / 1000.0
        if not self.label_rows.empty:
            self.label_rows["relative_time_s"] = (self.label_rows["timestamp_ms"] - self.t0_global) / 1000.0

    # ----- UI -----
    def build_ui(self):
        outer = QVBoxLayout(self)

        # controls
        controls = QHBoxLayout()
        controls.addWidget(QLabel(f"<b>{self.name}</b>"))

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
        self.save_btn.clicked.connect(self.save_image)
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
        """
        Build output 'out' and ALWAYS compute relative_time_s from shared t0 before returning.
        """
        # base output
        if len(df) < 3 or self.no_ds_cb.isChecked():
            out = df.sort_values("timestamp_ms").copy()
        else:
            t0 = int(df["timestamp_ms"].iloc[0])
            t1 = int(df["timestamp_ms"].iloc[-1])
            step_ms = max(1, int(round(1000.0 / float(self.target_rate))))
            grid = pd.DataFrame({"timestamp_ms": np.arange(t0, t1 + 1, step_ms, dtype=np.int64)})

            src = df.sort_values("timestamp_ms").copy()
            src["timestamp_ms"] = src["timestamp_ms"].astype(np.int64)
            tol = step_ms
            out = pd.merge_asof(grid, src, on="timestamp_ms", direction="nearest", tolerance=tol)

            # forward/backward fill configured slow cols
            for c in self.slow_cols:
                if c in out.columns:
                    out.loc[out[c].isin([-99, -99.0]), c] = np.nan
                    out[c] = out[c].ffill().bfill()

            # nearest labels on grid
            if hasattr(self, "label_rows") and not self.label_rows.empty:
                labels_src = self.label_rows.sort_values("timestamp_ms").copy()
                labels_src["timestamp_ms"] = labels_src["timestamp_ms"].astype(np.int64)
                labels_grid = pd.merge_asof(
                    out[["timestamp_ms"]], labels_src,
                    on="timestamp_ms", direction="nearest", tolerance=tol
                )
                out["label"] = labels_grid["label"]

            step_ms = max(1, int(round(1000.0 / float(self.target_rate))))
            if step_ms != self.last_step_logged:
                self.log.info(f"{self.name}: resample step = {step_ms} ms (view≈{self.target_rate} Hz)")
                self.last_step_logged = step_ms

        # compute relative time from SHARED t0 (or local if not set yet)
        t0_ms = self.t0_global if self.t0_global is not None else int(out["timestamp_ms"].iloc[0])
        out["relative_time_s"] = (out["timestamp_ms"] - t0_ms) / 1000.0
        return out

    def robust_yrange(self, s):
        s = pd.to_numeric(s, errors="coerce")
        s = s[~s.isin([-99, -99.0])].dropna()
        if s.empty: return (-1, 1)
        q1, q99 = np.nanpercentile(s, [1, 99])
        if not np.isfinite(q1) or not np.isfinite(q99) or np.isclose(q1, q99):
            pad = 1.0 if not np.isfinite(q99) or q99 == 0 else abs(q99) * 0.2
            return (float(q1) - pad, float(q99) + pad)
        return (float(q1), float(q99))

    def update_plot(self):
        if self.data.empty or self.current_signal is None:
            return
        self.plot.clear()

        df = self.downsample_for_view(self.data).copy()
        self.current_df = df

        sig = self.current_signal
        if sig not in df.columns:
            self.log.warning(f"{self.name}: signal '{sig}' missing")
            return

        self.plot.plot(df["relative_time_s"], df[sig], pen=pg.mkPen(color="dodgerblue", width=1.5))

        ymin, ymax = self.robust_yrange(df[sig])
        if ymin == ymax: ymin -= 1.0; ymax += 1.0
        self.plot.setYRange(ymin, ymax, padding=0)

        # labels
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
            if len(vis) > MAX_LABELS:
                step = int(np.ceil(len(vis) / MAX_LABELS))
                vis = vis.iloc[::step, :]

            y_span = (ymax - ymin) if ymax > ymin else 1.0
            y_base = ymax + y_span * 0.06
            stagger = y_span * 0.05
            signs = [1, -1, 2, -2]
            for k, (_, row) in enumerate(vis.iterrows()):
                x = float(row["relative_time_s"]); lbl = str(row["label"])
                self.plot.addItem(pg.InfiniteLine(pos=x, angle=90,
                                                  pen=pg.mkPen('r', style=pg.QtCore.Qt.DashLine)))
                t = pg.TextItem(lbl, color='r', anchor=(0, 0))
                t.setRotation(90)
                t.setPos(x, y_base + signs[k % len(signs)] * stagger)
                self.plot.addItem(t)

        unit = self.units.get(sig, "")
        # self.plot.setTitle(f"{self.name} — {sig} | view≈{int(self.target_rate)} Hz (src≈{int(self.source_hz)} Hz)",
        #                    color="black", size="12pt")
        # self.plot.setLabel("left", f"{sig} ({unit})", color="black")
        unit = self.units.get(sig, "")
        display_name = self.display_names.get(sig, sig)  # fallback to raw name if not mapped
    #     self.plot.setTitle(
    #     f"{self.name} — {display_name} | view≈{int(self.target_rate)} Hz (src≈{int(self.source_hz)} Hz)",
    #      color="black",
    #         size="12pt"
    #   )
        base_title = (
            f"{self.name} — {display_name} | "
            f"view≈{int(self.target_rate)} Hz (src≈{int(self.source_hz)} Hz)"
        )
       
        title = f"{self.custom_header} — {base_title}" if self.custom_header else base_title
        self.plot.setTitle(title, color="black", size="12pt")
        self.plot.setLabel("left", f"{display_name} ({unit})", color="black")
        self.plot.setLabel("bottom", "Time (s)", color="black")

    # ----- callbacks & interactions -----
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

    def set_pan_mode(self, enabled: bool):
        """Toggle between horizontal pan and box-zoom modes."""
        vb = self.plot.getViewBox()
        if enabled:
            # drag pans; lock Y so we only move horizontally
            vb.setMouseMode(pg.ViewBox.PanMode)
            vb.setMouseEnabled(x=True, y=False)
            self.plot.setCursor(Qt.ClosedHandCursor)
        else:
            # rectangular zoom; restore normal zooming on Y
            vb.setMouseMode(pg.ViewBox.RectMode)
            vb.setMouseEnabled(x=True, y=True)
            self.plot.setCursor(Qt.ArrowCursor)


# ---------- Main window with two panes + Save Both ----------
class CompareViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        CUSTOM_HEADER = "P17 | Session 1 | ML023 "
        self.setWindowTitle("BIOPAC vs mDAQ — Time Series Comparison")
        self.setGeometry(60, 60, 1280, 860)

        # Select files
        biopac_path, _ = QFileDialog.getOpenFileName(self, "Select BIOPAC CSV", "", "CSV Files (*.csv)")
        if not biopac_path:
            sys.exit("No BIOPAC file selected.")
        mdaq_path, _ = QFileDialog.getOpenFileName(self, "Select mDAQ CSV", "", "CSV Files (*.csv)")
        if not mdaq_path:
            sys.exit("No mDAQ file selected.")

        central = QWidget(); outer = QVBoxLayout(central)
        self.setCentralWidget(central)

        # Panes
        self.biopac = SignalPane("BIOPAC", Path(biopac_path), slow_cols=[], custom_header=CUSTOM_HEADER)
        self.mdaq   = SignalPane("mDAQ",   Path(mdaq_path),   slow_cols=["relative_humidity","ambient_temp","body_temp"], custom_header=CUSTOM_HEADER)

        # Shared t0 so both align, then draw
        t0_global = int(min(self.biopac.data["timestamp_ms"].min(),
                            self.mdaq.data["timestamp_ms"].min()))
        self.biopac.set_global_t0(t0_global)
        self.mdaq.set_global_t0(t0_global)
        self.biopac.update_plot()
        self.mdaq.update_plot()

        # Link X range ONE WAY (either is fine; this avoids circular link quirks)
        self.mdaq.plot.setXLink(self.biopac.plot)

        # Add to UI
        outer.addWidget(self.biopac)
        outer.addWidget(self.mdaq)

        # bottom row: Save Both
        bottom = QHBoxLayout()
        bottom.addStretch(1)
        #Adding button for panning: 
        self.pan_btn = QPushButton("Pan (X-linked)")
        self.pan_btn.setCheckable(True)
        self.pan_btn.setToolTip("Drag to pan horizontally. Uncheck to return to box-zoom.")
        self.pan_btn.toggled.connect(self.on_toggle_pan)
        bottom.addWidget(self.pan_btn)

        self.save_both_btn = QPushButton("Save Both (same window)")
        self.save_both_btn.clicked.connect(self.save_both)
        bottom.addWidget(self.save_both_btn)
        outer.addLayout(bottom)

        # toast overlay
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
        QTimer.singleShot(ms + 300, self.toast.hide)

    def save_both(self):
        # Pick a base filename; we derive two files: *_BIOPAC.* and *_mDAQ.*
        base, _ = QFileDialog.getSaveFileName(
            self, "Save Both Plots (base name)", "comparison_biopac_mdaq.png",
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;SVG Vector (*.svg)"
        )
        if not base:
            return

        stem = Path(base).with_suffix("")
        suffix = Path(base).suffix.lower() or ".png"
        if suffix not in (".png", ".jpg", ".jpeg", ".svg"):
            suffix = ".png"

        f_biopac = stem.with_name(stem.name + "_BIOPAC").with_suffix(suffix)
        f_mdaq   = stem.with_name(stem.name + "_mDAQ").with_suffix(suffix)

        # Export BIOPAC
        if suffix == ".svg":
            exp1 = SVGExporter(self.biopac.plot.getPlotItem())
        else:
            exp1 = ImageExporter(self.biopac.plot.getPlotItem()); exp1.parameters()['width'] = 2560
        exp1.export(str(f_biopac))

        # Export mDAQ
        if suffix == ".svg":
            exp2 = SVGExporter(self.mdaq.plot.getPlotItem())
        else:
            exp2 = ImageExporter(self.mdaq.plot.getPlotItem()); exp2.parameters()['width'] = 2560
        exp2.export(str(f_mdaq))

        self.show_toast(f"Saved: {f_biopac.name} + {f_mdaq.name}")

    def on_toggle_pan(self, enabled: bool):
        # switch both panes together
        self.biopac.set_pan_mode(enabled)
        self.mdaq.set_pan_mode(enabled)
        # tiny toast so you know what mode you're in
        self.show_toast("Pan mode ON" if enabled else "Pan mode OFF (box-zoom)")



def main():
    app = QApplication(sys.argv)
    viewer = CompareViewer()
    viewer.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
