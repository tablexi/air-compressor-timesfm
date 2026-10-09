# TimesFM Compressor Monitor

An interactive demo of anomaly detection with Google's **TimesFM** forecasting
model. A simulated air compressor streams line-pressure readings into a live
chart, TimesFM keeps forecasting what the pressure *should* look like, and a
single key press opens an air leak that the model detects and flags, usually
before the pressure leaves its acceptable range.

## Getting it running on a Mac

Tested on Apple Silicon with macOS and Python 3.13. Python 3.10 or newer
should work.

1. **Install Python** (skip if you already have 3.10+). The simplest way is
   [uv](https://docs.astral.sh/uv/):

   ```sh
   brew install uv        # or: curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Create a virtual environment and install the dependencies** from this
   folder:

   ```sh
   uv venv --python 3.13 .venv
   uv pip install --python .venv/bin/python -r requirements.txt
   ```

   Without uv, the standard library tools work too:

   ```sh
   python3 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```

3. **Run the demo:**

   ```sh
   .venv/bin/python compressor_demo.py
   ```

   The first run downloads the TimesFM 2.5 model weights (~800 MB) from
   Hugging Face into `~/.cache/huggingface`. Later runs start in a few seconds.

4. **Click the chart window** so it has keyboard focus, then use:

   | Key | Action                                    |
   |-----|-------------------------------------------|
   | `d` | Start or stop an air leak (pressure drop) |
   | `r` | Reset the compressor to normal operation  |
   | `q` | Quit                                      |

### What you'll see

- **Black line**: the live pressure reading, one every 50 ms.
- **Green band**: the acceptable operating range (6.0–8.0 bar).
- **Blue dashed line and shaded band**: TimesFM's forecast of the next 64
  readings (median, with 10th–90th percentile range).
- **Orange line**: the lowest pressure TimesFM expected for the most recent
  readings. When the real pressure falls clearly below it, that's an anomaly.
- **Status box**: green `NORMAL`, red `ANOMALY: PRESSURE DROP`, or yellow
  `RECOVERING` (explained below). Alarm periods stay shaded red on the chart.

## The emulated air compressor

The simulator (`Compressor` in `compressor_demo.py`) models a typical
load/unload compressor feeding an air line:

- **Normal cycle.** The compressor switches on (*cut-in*) when pressure drops
  to **6.5 bar** and pumps until it reaches **7.5 bar** (*cut-out*). While it is
  off, normal air demand slowly drains the line. This gives a steady sawtooth
  of about 3.5 seconds per cycle, plus a little sensor noise (±0.03 bar).
- **Leak (`d`).** Extra air escapes in proportion to the line pressure, as it
  would through a real hole. The compressor can no longer keep up: pressure
  stops recovering when the compressor switches on and gradually settles at
  around 4.5 bar, well below the acceptable range.
- **Recovery.** Press `d` again to fix the leak. The compressor pressurises the
  line back up and the normal cycle resumes.

All rates, limits and noise levels are constants at the top of
`compressor_demo.py` if you want to change the machine's behaviour.

## How the anomaly detection works

Every four readings the demo asks TimesFM two things:

1. **"What should the last 16 readings have been?"** The model is given the
   history *before* those readings and forecasts them. If at least 12 of the 16
   actual readings fall more than 0.05 bar below the model's 10th-percentile
   forecast, the alarm goes off.
2. **"What happens next?"** Using the full history, it forecasts the next 64
   readings for the chart.

Because TimesFM has learned the sawtooth rhythm, it expects pressure to climb
as soon as the compressor switches on. A leak breaks that pattern immediately,
so the alarm usually goes off **about 1 second after the leak starts, while
the pressure is still inside the acceptable range**. A simple "below 6.0 bar"
threshold would only react later.

Two extra rules keep the alarm reliable:

- **The alarm stays on** until the compressor pressurises the line back up to
  cut-out (7.4 bar or more), which a leaking line can't do. Otherwise TimesFM
  would eventually accept the falling pressure as the "new normal" and clear
  the alarm while the leak is still there.
- **A `RECOVERING` period** (about 13 seconds) follows each alarm. During that
  time the leak is still part of the history TimesFM is reading, so its
  forecasts aren't trustworthy, and only the hard 6.0 bar limit can raise an
  alarm.

Each forecast takes roughly 60–100 ms on an Apple Silicon CPU. Forecasts run
in a background thread, so the chart stays smooth.

## About TimesFM

[TimesFM](https://docs.cloud.google.com/bigquery/docs/timesfm-model) (Time
Series Foundation Model) is a forecasting model from Google Research. Like a
large language model, it is a decoder-only transformer, but it was pretrained
on a very large collection of real and synthetic time series instead of text.
It splits a series into short patches (like "words") and predicts the patches
that come next.

Because of this pretraining, TimesFM does **zero-shot forecasting**: you give
it a series it has never seen, such as this compressor's pressure, and it
forecasts it straight away, with no training or tuning. Besides a single best
guess, it also predicts quantiles (10th to 90th percentile), which is what this
demo uses to judge whether a reading is unusually low.

This demo uses the open 200M-parameter **TimesFM 2.5** model through the
[`timesfm`](https://github.com/google-research/timesfm) Python package. The
same model is also available as a managed model in Google BigQuery. See the
[TimesFM model documentation](https://docs.cloud.google.com/bigquery/docs/timesfm-model).

## Files

| File                 | Purpose                                     |
|----------------------|---------------------------------------------|
| `compressor_demo.py` | Simulator, TimesFM detector and live chart  |
| `requirements.txt`   | Python dependencies                         |
