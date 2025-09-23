# dual_timeseries_viewer.py
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
from pyqtgraph.exporters import ImageExporter, SVGExporter


# ---------- helpers ----------
def setup_logger(path: Path, name="dual_viewer"):
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    for h in list(log.handlers):
        log.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); log.addHandler(sh)
    try:
        fh = logging.FileHandler(path, mode="w", encoding="utf-8"); fh.setFormatter(fmt); log.addHandler(fh)
    except Exception:
        pass
    return log


class PlainAxis(pg.AxisItem):
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


# ---------- reusable pane ----------
class SignalPane(QWidget):
    """
    Reusable plot pane (selector + rate + optional 'no ds' + labels + save image)
    """
    def __init__(self, title, slow_cols=None, parent=None):
        super().__init__(parent)
        self.title = title
        self.slow_cols = slow_cols or []          # columns to forward-fill (e.g., 1 Hz temp/RH)
        self.units = {}                           # per-signal units (optional)
        self.data = pd.DataFrame()
        self.label_rows = pd.DataFrame()
        self.source_hz = 100.0
        self.target_rate = 30
        self.current_signal = None
        self.last_step_logged = None
        self.t0_global = None

        # --- UI ---
        outer = QVBoxLayout(self)
        controls = QHBoxLayout()

        self.label_title = QLabel(f"<b>{self.title}</b>")
        controls.addWidget(self.label_title)

        controls.addSpacing(8)
        controls.addWidget(QLabel("Select signal"))
        self.signal_box = QComboBox()
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

        controls.addSpacing(12)
        self.show_labels_cb = QCheckBox("Show labels")
        self.show_labels_cb.setChecked(True)
        self.show_labels_cb.stateChanged.connect(lambda _: self.update_plot())
        controls.addWidget(self.show_labels_cb)

        controls.addStretch(1)
        self.save_btn = QPushButton("Save Image")
        self.save_btn.clicked.connect(self.save_image)
        controls.addWidget(self.save_btn)

        outer.addLayout(controls)

        left_axis = PlainAxis(orientation="left")
        self.plot = pg.PlotWidget(axisItems={'left': left_axis})
        self.plot.setBackground("#fafafa")
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setClipToView(True)
        self.plot.setDownsampling(auto=True, mode="peak")
        self.plot.getAxis('bottom').enableAutoSIPrefix(False)
        self.plot.getAxis("left").setTextPen("black")
        self.plot.getAxis("bottom").setTextPen("black")
        self.plot.getViewBox().setMouseMode(pg.ViewBox.RectMode)
        self.plot.scene().sigMouseClicked.connect(self._on_double_reset)

        # crosshair + tooltip
        self.vLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('gray', style=pg.QtCore.Qt.DashLine))
        self.hLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('gray', style=pg.QtCore.Qt.DashLine))
        self.plot.addItem(self.vLine, ignoreBounds=True)
        self.plot.addItem(self.hLine, ignoreBounds=True)
        self.tooltip = pg.TextItem("", anchor=(0, 1), color="black")
        self.plot.addItem(self.tooltip)
        self.proxy = pg.SignalProxy(self.plot.scene().sigMouseMoved, rateLimit=30, slot=self._on_mouse_move)

        outer.addWidget(self.plot)

        # toast
        self.toast = QLabel("", self.plot)
        self.toast.setVisible(False)

    # ----- load / set data -----
    def load_csv(self, path: Path, units_map=None):
        df = pd.read_csv(path, low_memory=False)

        if "timestamp_ms" not in df.columns:
            raise ValueError("CSV must contain 'timestamp_ms'.")
        if "label" not in df.columns:
            df["label"] = ""

        # numeric cast
        for c in df.columns:
            if c in ("timestamp_ms", "label"):
                continue
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # timestamp dtype
        df["timestamp_ms"] = pd.to_numeric(df["timestamp_ms"], errors="coerce")
        bad = int(df["timestamp_ms"].isna().sum())
        if bad:
            df = df.dropna(subset=["timestamp_ms"])
        df["timestamp_ms"] = df["timestamp_ms"].astype(np.int64)

        # label transitions
        lab = df["label"].astype(str)
        mask = (lab.str.strip() != "") & (lab.str.lower() != "nan")
        label_rows = df.loc[mask, ["timestamp_ms", "label"]].copy()
        if not label_rows.empty:
            label_rows["__prev__"] = label_rows["label"].shift(fill_value="__START__")
            label_rows = label_rows[label_rows["label"] != label_rows["__prev__"]].drop(columns="__prev__")
        label_rows["timestamp_ms"] = label_rows["timestamp_ms"].astype(np.int64)
        self.label_rows = label_rows.reset_index(drop=True)

        # source Hz from median dt
        diffs = np.diff(df["timestamp_ms"].values.astype(np.int64))
        diffs = diffs[diffs > 0]
        med_ms = float(np.median(diffs)) if diffs.size else np.nan
        self.source_hz = (1000.0 / med_ms) if med_ms and med_ms > 0 else 100.0

        # store
        self.data = df
        if units_map:
            self.units.update(units_map)

        # populate signals
        numeric_cols = [c for c in self.data.columns
                        if c not in ("timestamp_ms","relative_time_s","label")
                        and pd.api.types.is_numeric_dtype(self.data[c])]
        numeric_cols = sorted(numeric_cols)
        self.signal_box.clear()
        self.signal_box.addItems(numeric_cols)
        # pick a default
        prefer = ["ECG","PPG","EDA","SKT","ecg","eda","ir","red","acc_x","gyr_x","ambient_temp","batt%"]
        choose = next((c for c in prefer if c in numeric_cols), numeric_cols[0] if numeric_cols else None)
        self.current_signal = choose
        if choose:
            self.signal_box.setCurrentText(choose)

    def set_global_t0(self, t0_ms: int):
        """Set the shared origin for both panes and compute relative_time_s."""
        self.t0_global = int(t0_ms)
        if not self.data.empty:
            self.data["relative_time_s"] = (self.data["timestamp_ms"] - self.t0_global) / 1000.0
        if not self.label_rows.empty:
            # compute and keep for draw-window filtering
            self.label_rows["relative_time_s"] = (self.label_rows["timestamp_ms"] - self.t0_global) / 1000.0

    # ----- resampling -----
    def resample_for_view(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) < 3:
            return df

        if self.no_ds_cb.isChecked():
            out = df.sort_values("timestamp_ms").copy()
            if "relative_time_s" not in out.columns and self.t0_global is not None:
                out["relative_time_s"] = (out["timestamp_ms"] - self.t0_global) / 1000.0
            return out

        t0 = int(df["timestamp_ms"].iloc[0])
        t1 = int(df["timestamp_ms"].iloc[-1])
        step_ms = max(1, int(round(1000.0 / float(self.target_rate))))
        grid = pd.DataFrame({"timestamp_ms": np.arange(t0, t1 + 1, step_ms, dtype=np.int64)})

        src = df.sort_values("timestamp_ms").copy()
        src["timestamp_ms"] = src["timestamp_ms"].astype(np.int64)
        tol = step_ms
        out = pd.merge_asof(grid, src, on="timestamp_ms", direction="nearest", tolerance=tol)

        # forward/backward fill slow sensors (if present)
        for c in self.slow_cols:
            if c in out.columns:
                out.loc[out[c].isin([-99, -99.0]), c] = np.nan
                out[c] = out[c].ffill().bfill()

        # nearest labels on grid
        if not self.label_rows.empty:
            labels_src = self.label_rows.sort_values("timestamp_ms").copy()
            labels_src["timestamp_ms"] = labels_src["timestamp_ms"].astype(np.int64)
            labels_grid = pd.merge_asof(out[["timestamp_ms"]], labels_src,
                                        on="timestamp_ms", direction="nearest", tolerance=tol)
            out["label"] = labels_grid["label"]

        # shared relative time
        if self.t0_global is not None:
            out["relative_time_s"] = (out["timestamp_ms"] - self.t0_global) / 1000.0

        if step_ms != self.last_step_logged:
            self.last_step_logged = step_ms
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

    # ----- plot/update -----
    def update_plot(self):
        if self.data.empty or self.current_signal is None:
            return
        self.plot.clear()

        df = self.resample_for_view(self.data).copy()
        self._current_df = df

        sig = self.current_signal
        if sig not in df.columns:
            return

        self.plot.plot(df["relative_time_s"], df[sig], pen=pg.mkPen(color="dodgerblue", width=1.5))

        ymin, ymax = self.robust_yrange(df[sig])
        if ymin == ymax:
            ymin -= 1.0; ymax += 1.0
        self.plot.setYRange(ymin, ymax, padding=0)

        # labels
        if self.show_labels_cb.isChecked() and not self.label_rows.empty:
            try:
                xmin, xmax = self.plot.getViewBox().viewRange()[0]
                vis = self.label_rows[
                    (self.label_rows["relative_time_s"] >= xmin) &
                    (self.label_rows["relative_time_s"] <= xmax)
                ]
            except Exception:
                vis = self.label_rows

            MAX_L = 300
            if len(vis) > MAX_L:
                step = int(np.ceil(len(vis) / MAX_L))
                vis = vis.iloc[::step, :]

            y_span = (ymax - ymin) if ymax > ymin else 1.0
            y_base = ymax + y_span * 0.06
            stagger = y_span * 0.05
            signs = [1, -1, 2, -2]
            for k, (_, row) in enumerate(vis.iterrows()):
                x = float((row["timestamp_ms"] - self.t0_global)/1000.0) if "relative_time_s" not in row else float(row["relative_time_s"])
                self.plot.addItem(pg.InfiniteLine(pos=x, angle=90,
                                                  pen=pg.mkPen('r', style=pg.QtCore.Qt.DashLine)))
                t = pg.TextItem(str(row["label"]), color='r', anchor=(0, 0))
                t.setRotation(90)
                t.setPos(x, y_base + signs[k % len(signs)] * stagger)
                self.plot.addItem(t)

        unit = self.units.get(sig, "")
        self.plot.setTitle(f"{self.title} — {sig} | view≈{int(self.target_rate)} Hz (src≈{int(self.source_hz)} Hz)",
                           color="black", size="12pt")
        self.plot.setLabel("left", f"{sig} ({unit})" if unit else sig, color="black")
        self.plot.setLabel("bottom", "Time (s)", color="black")

    # ----- interactions -----
    def on_signal_change(self, text): self.current_signal = text; self.update_plot()
    def on_rate_change(self, v): self.target_rate = int(v); self.update_plot()
    def _on_double_reset(self, ev): 
        if ev.double(): self.plot.enableAutoRange()
    def _on_mouse_move(self, ev):
        pos = ev[0]
        if not self.plot.sceneBoundingRect().contains(pos): return
        vb = self.plot.getViewBox()
        mouse_pt = vb.mapSceneToView(pos)
        x = mouse_pt.x()

        if hasattr(self, "_current_df") and not self._current_df.empty:
            idx = (np.abs(self._current_df["relative_time_s"] - x)).idxmin()
            x_val = float(self._current_df["relative_time_s"].loc[idx])
            y_val = float(self._current_df[self.current_signal].loc[idx])
        else:
            x_val, y_val = mouse_pt.x(), mouse_pt.y()
        self.vLine.setPos(x_val); self.hLine.setPos(y_val)
        unit = self.units.get(self.current_signal, "")
        self.tooltip.setHtml(f"<span style='color:black;'>Time: {x_val:.2f}s<br>Value: {y_val:.3f} {unit}</span>")
        self.tooltip.setPos(x_val, y_val)

    # ----- save -----
    def save_image(self, fname: str = None):
        if fname is None:
            default_name = f"{self.title}_{self.current_signal}_plot"
            fname, ffilter = QFileDialog.getSaveFileName(
                self, "Save Plot Image", f"{default_name}.png",
                "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;SVG Vector (*.svg)"
            )
            if not fname: return
            if '.' not in os.path.basename(fname):
                if "PNG" in ffilter: fname += ".png"
                elif "JPEG" in ffilter: fname += ".jpg"
                elif "SVG" in ffilter: fname += ".svg"
                else: fname += ".png"

        if fname.lower().endswith(".svg"):
            exporter = SVGExporter(self.plot.getPlotItem())
        else:
            exporter = ImageExporter(self.plot.getPlotItem())
            exporter.parameters()['width'] = 2560
        exporter.export(fname)
        self._toast(f"Saved: {os.path.basename(fname)}", "#c8e6c9")

    def _toast(self, text, bg="lightyellow", ms=2000):
        self.toast.setText(text)
        self.toast.setStyleSheet(f"background-color:{bg}; color:black; padding:6px 10px; border-radius:6px; font-weight:bold;")
        self.toast.adjustSize()
        pad = 6
        self.toast.move(self.plot.width() - self.toast.width() - pad, pad)
        self.toast.raise_()
        self.toast.setWindowOpacity(1.0)
        self.toast.setVisible(True)
        anim = QPropertyAnimation(self.toast, b"windowOpacity")
        anim.setDuration(ms); anim.setStartValue(1.0); anim.setEndValue(0.0)
        anim.finished.connect(lambda: self.toast.setVisible(False))
        anim.start()


# ---------- main window with two panes ----------
class DualViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BIOPAC + mDAQ Time Series (Linked)")
        self.setGeometry(60, 60, 1280, 900)

        # pick files
        bio_path, _ = QFileDialog.getOpenFileName(self, "Select BIOPAC CSV", "", "CSV Files (*.csv)")
        if not bio_path: sys.exit("No BIOPAC file chosen.")
        mdaq_path, _ = QFileDialog.getOpenFileName(self, "Select mDAQ CSV", "", "CSV Files (*.csv)")
        if not mdaq_path: sys.exit("No mDAQ file chosen.")

        self.log = setup_logger(Path(bio_path).with_suffix(".dual_viewer.log"))
        self.log.info(f"BIOPAC: {bio_path}")
        self.log.info(f"mDAQ:   {mdaq_path}")

        # panes
        self.bio = SignalPane("BIOPAC")
        self.mdq = SignalPane("mDAQ", slow_cols=["relative_humidity","ambient_temp","body_temp"])

        # optional units (edit to taste)
        self.bio.units.update({"ECG":"mV","PPG":"a.u.","EDA":"µS","SKT":"°C"})
        self.mdq.units.update({
            "ecg":"mV","eda":"µS","ir":"a.u.","red":"a.u.",
            "acc_x":"g","acc_y":"g","acc_z":"g",
            "gyr_x":"°/s","gyr_y":"°/s","gyr_z":"°/s",
            "batt%":"%","relative_humidity":"%","ambient_temp":"°C","body_temp":"°C"
        })

        # load data
        self.bio.load_csv(Path(bio_path), units_map=self.bio.units)
        self.mdq.load_csv(Path(mdaq_path), units_map=self.mdq.units)

        # set a shared t0 so both x-axes align, then draw
        t0_global = int(min(self.bio.data["timestamp_ms"].min(), self.mdq.data["timestamp_ms"].min()))
        self.bio.set_global_t0(t0_global)
        self.mdq.set_global_t0(t0_global)

        # link x-axes
        self.mdq.plot.setXLink(self.bio.plot)

        # layout
        central = QWidget(); outer = QVBoxLayout(central)
        outer.addWidget(self.bio)
        outer.addWidget(self.mdq)

        # save both button row
        row = QHBoxLayout()
        row.addStretch(1)
        btn_save_both = QPushButton("Save Both (same window)")
        btn_save_both.clicked.connect(self.save_both)
        row.addWidget(btn_save_both)
        outer.addLayout(row)

        self.setCentralWidget(central)

        # initial plots
        self.bio.update_plot()
        self.mdq.update_plot()

    def save_both(self):
        """
        Export *both* plots with the same current x-range (they're linked).
        Creates two files with a shared prefix.
        """
        prefix, _ = QFileDialog.getSaveFileName(self, "Save Both (prefix only)", "comparison.png",
                                                "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;SVG Vector (*.svg)")
        if not prefix: return
        # normalize to stem + extension
        if '.' not in os.path.basename(prefix):
            prefix += ".png"
        stem, ext = os.path.splitext(prefix)

        # BIOPAC
        bio_out = f"{stem}_BIOPAC{ext}"
        if ext.lower() == ".svg":
            exp = SVGExporter(self.bio.plot.getPlotItem())
        else:
            exp = ImageExporter(self.bio.plot.getPlotItem())
            exp.parameters()['width'] = 2560
        exp.export(bio_out)

        # mDAQ
        mdq_out = f"{stem}_mDAQ{ext}"
        if ext.lower() == ".svg":
            exp2 = SVGExporter(self.mdq.plot.getPlotItem())
        else:
            exp2 = ImageExporter(self.mdq.plot.getPlotItem())
            exp2.parameters()['width'] = 2560
        exp2.export(mdq_out)

        # quick toast via the lower pane
        self.mdq._toast(f"Saved: {os.path.basename(bio_out)} + {os.path.basename(mdq_out)}", "#c8e6c9")


def main():
    app = QApplication(sys.argv)
    win = DualViewer()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
