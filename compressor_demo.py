"""Live TimesFM anomaly-detection demo on a simulated air compressor.

The simulator produces line-pressure readings (bar) from a compressor that
cycles between load (pumping) and unload (idle while demand drains the tank).
Every few ticks TimesFM forecasts the series and we flag an anomaly when the
latest readings fall below the model's predicted 10th percentile.

Keys (focus the plot window):
  d  toggle a pressure drop (air leak)
  r  reset to normal operation
  q  quit
"""

import threading
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import timesfm
import torch
from matplotlib.animation import FuncAnimation

# --- Simulation -------------------------------------------------------------
CUT_IN, CUT_OUT = 6.5, 7.5  # bar: compressor starts / stops pumping
PUMP_RATE = 0.05            # bar per sample while loading
DEMAND_RATE = 0.02          # bar per sample drawn by normal air demand
LEAK_COEF = 0.0067          # leak loss per sample, proportional to pressure
NOISE = 0.03
SAFE_MIN, SAFE_MAX = 6.0, 8.0  # acceptable operating range (for reference)

# --- Detection --------------------------------------------------------------
CONTEXT = 256        # samples of history given to TimesFM (~3.5 cycles)
CHECK_WINDOW = 16    # recent samples compared against a lagged forecast
AHEAD = 64           # samples forecast into the future for display
MIN_BELOW = 12       # flag if this many of CHECK_WINDOW fall below q10...
TOLERANCE = 0.05     # ...by more than this many bar
COOLDOWN = CONTEXT   # after an alarm, wait until the fault leaves the context
TICK_MS = 50         # one reading every 50 ms
INFER_EVERY = 4      # run TimesFM every N new readings


class Compressor:
    def __init__(self, rng):
        self.rng = rng
        self.reset()

    def reset(self):
        self.pressure = 7.0
        self.loading = False
        self.leak = False

    def step(self):
        if self.pressure <= CUT_IN:
            self.loading = True
        elif self.pressure >= CUT_OUT:
            self.loading = False
        delta = -DEMAND_RATE + (PUMP_RATE if self.loading else 0.0)
        if self.leak:
            delta -= LEAK_COEF * self.pressure
        self.pressure = max(0.0, self.pressure + delta)
        return self.pressure + self.rng.normal(0, NOISE)


def load_model():
    print("Loading TimesFM 2.5 (first run downloads ~800 MB)...")
    torch.set_float32_matmul_precision("high")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch", torch_compile=False
    )
    model.compile(
        timesfm.ForecastConfig(
            max_context=CONTEXT,
            max_horizon=128,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            fix_quantile_crossing=True,
            per_core_batch_size=2,
        )
    )
    return model


class Detector:
    """Runs TimesFM in a background thread so the chart stays responsive."""

    def __init__(self, model):
        self.model = model
        self.lock = threading.Lock()
        self.busy = False
        self.result = None  # latest forecast + verdict
        self.latency = 0.0

    def submit(self, series, t_end):
        if self.busy:
            return
        self.busy = True
        threading.Thread(target=self._run, args=(series, t_end), daemon=True).start()

    def _run(self, series, t_end):
        try:
            start = time.perf_counter()
            # Input 0: history *before* the check window -> what should the last
            #          CHECK_WINDOW readings have looked like?
            # Input 1: full history -> forecast for display going forward.
            lagged = series[-CONTEXT - CHECK_WINDOW:-CHECK_WINDOW]
            current = series[-CONTEXT:]
            _, quantiles = self.model.forecast(
                horizon=AHEAD, inputs=[lagged, current]
            )
            # quantiles[..., 0] is the mean, [..., 1..9] are q10..q90.
            check_q10 = quantiles[0, :CHECK_WINDOW, 1]
            check_q50 = quantiles[0, :CHECK_WINDOW, 5]
            actual = series[-CHECK_WINDOW:]
            below = actual < check_q10 - TOLERANCE
            result = {
                "t_end": t_end,
                "check_q10": check_q10,
                "check_q50": check_q50,
                "n_below": int(below.sum()),
                "deficit": float(np.mean(check_q50 - actual)),
                "anomaly": int(below.sum()) >= MIN_BELOW,
                "ahead_q10": quantiles[1, :, 1],
                "ahead_q50": quantiles[1, :, 5],
                "ahead_q90": quantiles[1, :, 9],
            }
            with self.lock:
                self.result = result
                self.latency = time.perf_counter() - start
        finally:
            self.busy = False


def main():
    rng = np.random.default_rng(0)
    comp = Compressor(rng)
    model = load_model()
    detector = Detector(model)

    # Pre-fill history so TimesFM has full context from the start.
    history = deque((comp.step() for _ in range(CONTEXT + CHECK_WINDOW)), maxlen=2000)
    t = len(history)
    tick = 0

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.canvas.manager.set_window_title("TimesFM compressor monitor")
    ax.axhspan(SAFE_MIN, SAFE_MAX, color="tab:green", alpha=0.06, label="Acceptable range")
    (line_actual,) = ax.plot([], [], color="black", lw=1.2, label="Line pressure")
    (line_fc,) = ax.plot([], [], color="tab:blue", lw=1.5, ls="--", label="TimesFM forecast (median)")
    band_fc = [ax.fill_between([], [], [], color="tab:blue", alpha=0.15)]
    (line_check,) = ax.plot([], [], color="tab:orange", lw=2, label="Expected (lagged q10)")
    status = ax.text(0.01, 0.97, "", transform=ax.transAxes, va="top", fontsize=13,
                     family="monospace", bbox=dict(boxstyle="round", fc="white", ec="gray"))
    ax.text(0.99, 0.02, "[d] toggle leak   [r] reset   [q] quit", transform=ax.transAxes,
            ha="right", fontsize=10, color="gray")
    ax.set_ylim(3.5, 9)
    ax.set_xlabel(f"Sample ({TICK_MS} ms)")
    ax.set_ylabel("Pressure (bar)")
    ax.legend(loc="upper right", fontsize=9)
    alarm_start = [None]
    cooldown_until = [0]
    live_span = [None]

    def on_key(event):
        if event.key == "d":
            comp.leak = not comp.leak
            print(f"Leak {'ON' if comp.leak else 'OFF'}")
        elif event.key == "r":
            comp.reset()
            history.extend(comp.step() for _ in range(CONTEXT))
            cooldown_until[0] = 0
            print("Reset")
        elif event.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)

    def update(_):
        nonlocal t, tick
        history.append(comp.step())
        t += 1
        tick += 1
        series = np.array(history, dtype=np.float32)
        if tick % INFER_EVERY == 0:
            detector.submit(series, t)

        view = 400
        xs = np.arange(t - min(view, len(series)), t)
        line_actual.set_data(xs, series[-len(xs):])
        ax.set_xlim(t - view, t + AHEAD)

        with detector.lock:
            res, latency = detector.result, detector.latency

        if res is not None:
            x_ahead = np.arange(res["t_end"], res["t_end"] + AHEAD)
            line_fc.set_data(x_ahead, res["ahead_q50"])
            band_fc[0].remove()
            band_fc[0] = ax.fill_between(x_ahead, res["ahead_q10"], res["ahead_q90"],
                                         color="tab:blue", alpha=0.15)
            line_check.set_data(np.arange(res["t_end"] - CHECK_WINDOW, res["t_end"]),
                                res["check_q10"])

            # Right after a fault the context still contains it, so forecasts
            # are unreliable; only the hard low-pressure limit applies then.
            recovering = res["t_end"] < cooldown_until[0]
            detected = series[-1] < SAFE_MIN if recovering else res["anomaly"]
            # TimesFM raises the alarm; it stays latched until the compressor
            # manages to pressurise back up to cut-out (a leaking line can't).
            recovered = series[-1] >= CUT_OUT - 0.1
            if detected or (alarm_start[0] is not None and not recovered):
                if alarm_start[0] is None:
                    alarm_start[0] = res["t_end"] - CHECK_WINDOW
                    print(f"ANOMALY at sample {alarm_start[0]}: "
                          f"{res['deficit']:.2f} bar below expected")
                state = "ANOMALY: PRESSURE DROP"
                color = "#ffd6d6"
            else:
                if alarm_start[0] is not None:
                    ax.axvspan(alarm_start[0], res["t_end"], color="red", alpha=0.12)
                    alarm_start[0] = None
                    cooldown_until[0] = res["t_end"] + COOLDOWN
                    recovering = True
                if recovering:
                    state = f"RECOVERING (re-learning, {cooldown_until[0] - t} samples)"
                    color = "#fff3cd"
                else:
                    state = "NORMAL"
                    color = "#d9f2d9"
            status.set_text(
                f"{state}\n"
                f"pressure {series[-1]:5.2f} bar   "
                f"{res['n_below']:2d}/{CHECK_WINDOW} below q10   "
                f"deficit {res['deficit']:+.2f} bar\n"
                f"leak {'ON ' if comp.leak else 'off'}   inference {latency * 1000:4.0f} ms"
            )
            status.get_bbox_patch().set_facecolor(color)
        else:
            status.set_text("Waiting for first forecast...")

        # Shade the active alarm region live.
        if live_span[0] is not None:
            live_span[0].remove()
            live_span[0] = None
        if alarm_start[0] is not None:
            live_span[0] = ax.axvspan(alarm_start[0], t, color="red", alpha=0.12)
        return ()

    _anim = FuncAnimation(fig, update, interval=TICK_MS, cache_frame_data=False)
    print(__doc__)
    plt.show()


if __name__ == "__main__":
    main()
