import os
import json
import pickle
import asyncio
import logging
import warnings
from collections import deque

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from scipy.stats import skew as _skew, kurtosis as _kurtosis

# ==============================================
#  CONFIGURATION
# ==============================================
BASE_DIR = os.path.dirname(__file__)
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

app = FastAPI(title="ESP Vibration Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("esp-backend")

# ==============================================
#  LOAD MODEL + SCALER
# ==============================================
def _find_estimator(obj):
    """Recursively find an estimator object with predict()."""
    if hasattr(obj, "predict"):
        return obj
    if isinstance(obj, (list, tuple)):
        for item in obj:
            est = _find_estimator(item)
            if est is not None:
                return est
    if isinstance(obj, dict):
        for v in obj.values():
            est = _find_estimator(v)
            if est is not None:
                return est
    return None


def _load_scaler(path):
    """Load scaler (StandardScaler or dict with mean_ and scale_)."""
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if hasattr(obj, "transform") and hasattr(obj, "mean_") and hasattr(obj, "scale_"):
            return obj
        elif isinstance(obj, dict) and "mean_" in obj and "scale_" in obj:
            class DictScaler:
                def __init__(self, mean_, scale_):
                    self.mean_ = np.array(mean_)
                    self.scale_ = np.array(scale_)
                def transform(self, X):
                    return (X - self.mean_) / self.scale_
            return DictScaler(obj["mean_"], obj["scale_"])
        else:
            warnings.warn(f"Unrecognized scaler format in {path}")
            return None
    except Exception as e:
        logger.warning(f"Could not load scaler from {path}: {e}")
        return None


try:
    with open(MODEL_PATH, "rb") as f:
        raw_model = pickle.load(f)
    # If the pickle contains metadata (dict with models/scalers/label_map), extract label_map
    label_map = None
    if isinstance(raw_model, dict) and "label_map" in raw_model:
        label_map = raw_model.get("label_map")
    model = _find_estimator(raw_model)
    logger.info(f"Loaded model: {type(model)}; label_map={'present' if label_map else 'none'}")
except Exception as e:
    logger.error(f"Failed to load model from {MODEL_PATH}: {e}")
    model = None

# Prefer a scaler embedded in the saved artifact if present (artifact may include 'scaler' or 'scalers')
artifact_scaler = None
if isinstance(raw_model, dict):
    if "scaler" in raw_model and raw_model.get("scaler") is not None:
        artifact_scaler = raw_model.get("scaler")
    elif "scalers" in raw_model and isinstance(raw_model.get("scalers"), (list, tuple)) and len(raw_model.get("scalers")) > 0:
        # pick the first scaler as a sensible default; training code can choose to save a single scaler key instead
        artifact_scaler = raw_model.get("scalers")[0]

if artifact_scaler is not None:
    scaler = artifact_scaler
    logger.info("Loaded scaler from model artifact")
else:
    scaler = _load_scaler(SCALER_PATH)
    if scaler:
        logger.info(f"Loaded scaler from {SCALER_PATH}")
    else:
        logger.warning("No valid scaler found, continuing without normalization.")

# After both model and scaler are initialized, do a dry-run predict to surface
# any immediate issues (useful during startup / diagnostics).
try:
    if model is not None:
        sample_obs = np.zeros((1, 4), dtype=float)
        if scaler:
            try:
                # If scaler expects feature names, present a DataFrame to avoid warnings
                if hasattr(scaler, "feature_names_in_") and len(getattr(scaler, "feature_names_in_")) == sample_obs.shape[1]:
                    cols = list(getattr(scaler, "feature_names_in_"))
                    sample_obs = scaler.transform(pd.DataFrame(sample_obs, columns=cols))
                else:
                    sample_obs = scaler.transform(sample_obs)
            except Exception:
                # fallback to raw zeros
                pass
        try:
            dr = model.predict(sample_obs)
            logger.info(f"Dry-run model.predict succeeded, sample output: {dr}")
        except Exception as e:
            logger.warning(f"Dry-run predict failed: {e}")
except Exception:
    pass

# ==============================================
#  PREDICTION UTILITIES
# ==============================================
BUFFER_LOCK = asyncio.Lock()
SENSOR_BUFFERS = {}
SENSOR_STATE_HISTORY = {}  # sensor_id -> list of (timestamp, state) tuples
WINDOW_SIZE = int(os.environ.get("WINDOW_SIZE", "10"))  # reduced for local testing; set to 256 in production
WINDOW_STEP = int(os.environ.get("WINDOW_STEP", str(WINDOW_SIZE)))


def compute_obs_from_window(window_samples):
    """Compute [mean, rms, kurtosis, skew] for a window of samples."""
    arr = np.asarray(window_samples, dtype=float)
    mean = float(np.mean(arr))
    rms = float(np.sqrt(np.mean(arr ** 2)))
    kurt = float(_kurtosis(arr, fisher=False, bias=False))
    sk = float(_skew(arr, bias=False))
    obs = np.array([mean, rms, kurt, sk], dtype=float)
    # Prepare the observation for the scaler. Some StandardScaler instances
    # were fitted on a DataFrame and expect feature names; transforming a
    # plain numpy array produces a harmless warning and may misalign columns.
    X = obs.reshape(1, -1)
    if scaler:
        try:
            # If the scaler was fitted with feature names, present a DataFrame
            # with identical column names so transform behaves without warnings.
            if hasattr(scaler, "feature_names_in_"):
                cols = list(getattr(scaler, "feature_names_in_"))
                # Only use a DataFrame wrapper when the scaler's feature names
                # match our expected number of features. Otherwise sklearn will
                # warn and may misalign columns; fallback to numeric array.
                if len(cols) == X.shape[1]:
                    X_df = pd.DataFrame(X, columns=cols)
                    Xt = scaler.transform(X_df)
                else:
                    logger.info(f"Scaler feature_names length ({len(cols)}) != obs features ({X.shape[1]}), using numeric transform")
                    Xt = scaler.transform(X)
            else:
                Xt = scaler.transform(X)
            # ensure we return a 2D numpy array
            obs = np.asarray(Xt).reshape(1, -1)
        except Exception as e:
            # If scaling fails for any reason, log and continue with raw features
            logger.warning(f"Scaler transform failed, continuing without scaling: {e}")
            obs = X.reshape(1, -1)
    else:
        obs = X.reshape(1, -1)
    return obs


# ==============================================
#  ENDPOINTS
# ==============================================
@app.post("/sample")
async def sample_endpoint(request: Request):
    """
    Accepts single sensor samples or short sequences.
    Example payload:
    {"sensor_id": "esp1", "value": 0.123}
    """
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    sensor_id = payload.get("sensor_id", "default")

    # Handle single numeric sample
    if "value" not in payload:
        return JSONResponse(status_code=400, content={"error": "Missing 'value' in payload"})

    try:
        val = float(payload["value"])
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Value must be numeric"})

    # Buffer management
    async with BUFFER_LOCK:
        buf = SENSOR_BUFFERS.get(sensor_id)
        if buf is None:
            # keep deque bounded to WINDOW_SIZE to avoid transient lengths > window
            buf = deque(maxlen=WINDOW_SIZE)
            SENSOR_BUFFERS[sensor_id] = buf
        buf.append(val)

        if len(buf) < WINDOW_SIZE:
            buffer_len = min(len(buf), WINDOW_SIZE)
            return {"status": "buffering", "buffer_len": buffer_len, "buffer_size": WINDOW_SIZE}

        window = list(buf)[-WINDOW_SIZE:]
        for _ in range(WINDOW_STEP):
            if buf:
                buf.popleft()

    # Prediction
    try:
        obs = compute_obs_from_window(window)
        logger.info(f"Computed features for sensor={sensor_id}: {obs.tolist()}")

        # Ensure model exists
        if model is None:
            logger.error("No model loaded, cannot predict")
            return JSONResponse(status_code=500, content={"error": "No model available"})

        # Run prediction and handle any model errors
        try:
            y_pred_raw = model.predict(obs)
            y_pred = y_pred_raw[0] if hasattr(y_pred_raw, "__len__") else y_pred_raw
        except Exception as pe:
            logger.error(f"Model prediction error: {pe}")
            return JSONResponse(status_code=500, content={"error": "Prediction failed"})

        # Log successful prediction for visibility
        try:
            logger.info(f"Prediction for sensor={sensor_id}: raw={y_pred_raw}, interpreted={y_pred}")
        except Exception:
            pass

        # try to produce a human-friendly label
        try:
            pred_int = int(y_pred)
        except Exception:
            pred_int = None

        label = None
        # prefer a label_map from the loaded pickle if present
        if label_map:
            label = label_map.get(pred_int) if pred_int is not None else None
        # fallback default mapping for common 3-state HMMs
        if label is None:
            DEFAULT_LABEL_MAP = {0: "Healthy", 1: "Degraded", 2: "Faulty"}
            label = DEFAULT_LABEL_MAP.get(pred_int, None)

        # record state with timestamp for RUL computation
        try:
            ts = payload.get("timestamp") or payload.get("ts")
            # if device provided timestamp, try to parse numeric
            if ts is not None:
                try:
                    ts_val = float(ts)
                except Exception:
                    ts_val = None
            else:
                ts_val = None

            # use server time if device didn't provide ts
            now = float(pd.Timestamp.utcnow().timestamp())
            recorded_ts = ts_val if ts_val is not None else now

            hist = SENSOR_STATE_HISTORY.get(sensor_id)
            if hist is None:
                hist = []
                SENSOR_STATE_HISTORY[sensor_id] = hist
            # append (timestamp, state)
            hist.append((recorded_ts, int(pred_int) if pred_int is not None else str(y_pred)))
            # keep history bounded to recent 1000 entries
            if len(hist) > 1000:
                del hist[:-1000]
        except Exception:
            # don't fail prediction on history logging
            pass

        return {"sensor": sensor_id, "prediction": int(pred_int) if pred_int is not None else str(y_pred), "label": label}
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return JSONResponse(status_code=500, content={"error": "Prediction failed"})


# ==============================================
#  MAIN ENTRY
# ==============================================
@app.get("/")
def root():
    return {"status": "ok", "message": "ESP Prediction API running"}


@app.get("/debug")
def debug_info():
    """Return basic info about the loaded model and scaler for diagnostics."""
    model_info = None
    scaler_info = None
    try:
        model_info = {
            "type": str(type(model)),
        }
    except Exception:
        model_info = None
    try:
        if scaler is None:
            scaler_info = None
        else:
            scaler_info = {
                "type": str(type(scaler)),
            }
            if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
                scaler_info["mean_shape"] = np.shape(getattr(scaler, "mean_"))
                scaler_info["scale_shape"] = np.shape(getattr(scaler, "scale_"))
            if hasattr(scaler, "feature_names_in_"):
                scaler_info["feature_names_in_"] = list(getattr(scaler, "feature_names_in_"))
    except Exception:
        scaler_info = {"error": "could not inspect scaler"}

    return {"model": model_info, "scaler": scaler_info, "window_size": WINDOW_SIZE}


@app.get("/label-map")
def label_map_endpoint():
    """Return a label map for model outputs. Frontend fetches this on mount.

    Responds with JSON: { "label_map": { "0": "Healthy", "1": "Degraded", ... } }
    """
    try:
        # Prefer label_map embedded in the model artifact if present
        if label_map and isinstance(label_map, dict):
            # ensure keys are strings for JSON consistency
            return {"label_map": {str(k): v for k, v in label_map.items()}}
        # fallback default mapping
        DEFAULT_LABEL_MAP = {"0": "Healthy", "1": "Degraded", "2": "Faulty"}
        return {"label_map": DEFAULT_LABEL_MAP}
    except Exception as e:
        logger.error(f"label_map endpoint failed: {e}")
        return JSONResponse(status_code=500, content={"error": "Could not retrieve label map"})


@app.get("/rul")
def rul_endpoint(sensor_id: str = None):
    """
    Compute Remaining Useful Life (RUL) estimates based on observed state durations.
    This endpoint uses a simplified, server-side simulation of state transitions
    to demonstrate the RUL calculation. In production, replace the synthetic
    timeline with real stored prediction transitions per sensor.
    """
    try:
        # Build time_state: mapping state -> list of (start_ts, end_ts)
        time_state = {0: [], 1: [], 2: []}
        # Prefer per-sensor timestamped history if available
        hist = None
        if sensor_id:
            hist = SENSOR_STATE_HISTORY.get(sensor_id)

        if hist and len(hist) > 1:
            # hist: list of (timestamp, state)
            previous_state = hist[0][1]
            start_ts = hist[0][0]
            for ts, st in hist[1:]:
                if st != previous_state:
                    # record segment in seconds
                    time_state[int(previous_state)].append((start_ts, ts))
                    start_ts = ts
                    previous_state = st
            # finalize last
            time_state[int(previous_state)].append((start_ts, hist[-1][0]))
            units = "minutes"
            # compute RUL using timestamp differences (seconds -> minutes)
            # convert timestamps (seconds since epoch) to minutes when computing durations
            # The rul_calc expects start/end units consistent; we'll pass minutes
            time_state_in_minutes = {}
            for s, segments in time_state.items():
                time_state_in_minutes[s] = [((seg[0] / 60.0), (seg[1] / 60.0)) for seg in segments]
            rul_upper, rul_mean, rul_lower = rul_calc(time_state_in_minutes, 0.95)
            return {"rul_upper": float(rul_upper), "rul_mean": float(rul_mean), "rul_lower": float(rul_lower), "units": units}
        else:
            # fallback synthetic timeline consistent with training: 10-minute intervals
            time_state = {0: [], 1: [], 2: []}
            previous = None
            start_time = 0
            state_rul = []
            time_rul = []

            # generate timeline in minutes: 0,10,20,... up to 90
            x = np.arange(0, 100, 10)   # minutes
            y = np.random.choice([0, 1, 2], size=len(x), p=[0.5, 0.3, 0.2])

            for time_x, state_y in zip(x, y):
                if previous is None:
                    previous = state_y
                    start_time = time_x

                if previous != state_y:
                    time_state[previous].append((start_time, time_x))
                    start_time = time_x
                    rul_upper, rul_mean, rul_lower = rul_calc(time_state, 0.95)
                    state_rul.append((rul_upper, rul_mean, rul_lower))
                    time_rul.append([time_x, rul_upper, rul_mean, rul_lower])
                    previous = state_y

            # finalize last segment
            if len(x) > 0:
                time_state[previous].append((start_time, x[-1]))
                rul_upper, rul_mean, rul_lower = rul_calc(time_state, 0.95)
                state_rul.append((rul_upper, rul_mean, rul_lower))
                time_rul.append([x[-1], rul_upper, rul_mean, rul_lower])

            if not state_rul:
                return {"rul_upper": 0.0, "rul_mean": 0.0, "rul_lower": 0.0, "units": "minutes"}

            df_rul = pd.DataFrame(state_rul, columns=['RUL UPPER', 'RUL MEAN', 'RUL LOWER'])
            latest = df_rul.iloc[-1].to_dict()
            return {"rul_upper": float(latest['RUL UPPER']), "rul_mean": float(latest['RUL MEAN']), "rul_lower": float(latest['RUL LOWER']), "units": "minutes"}
    except Exception as e:
        logger.error(f"RUL computation failed: {e}")
        return JSONResponse(status_code=500, content={"error": "RUL computation failed"})


def rul_calc(time_state, conf):
    """Compute Remaining Useful Life (RUL) upper/mean/lower bounds.

    time_state: dict mapping state -> list of (start, end) tuples
    conf: confidence multiplier (e.g. 0.95)
    returns: (upper, mean, lower)
    """
    mean_std = {0: [], 1: [], 2: []}
    try:
        for state, times in time_state.items():
            # durations for each recorded segment
            decrease_time = [t[1] - t[0] for t in times if times]
            mean_state = float(np.mean(decrease_time)) if decrease_time else 0.0
            std_state = float(np.std(decrease_time)) if decrease_time else 0.0
            mean_std[state].append((mean_state, std_state))

        for s in mean_std:
            if not mean_std[s]:
                mean_std[s].append((0.0, 0.0))

        # ensure numeric values (avoid nan)
        mean_std[0][0], mean_std[1][0], mean_std[2][0] = (
            np.nan_to_num(mean_std[0][0]),
            np.nan_to_num(mean_std[1][0]),
            np.nan_to_num(mean_std[2][0]),
        )

        rul_upper = sum(m[0] + conf * m[1] for m in [mean_std[0][0], mean_std[1][0], mean_std[2][0]])
        rul_mean = sum(m[0] for m in [mean_std[0][0], mean_std[1][0], mean_std[2][0]])
        rul_lower = sum(m[0] - conf * m[1] for m in [mean_std[0][0], mean_std[1][0], mean_std[2][0]])

        return float(rul_upper), float(rul_mean), float(rul_lower)
    except Exception as e:
        logger.error(f"RUL helper failed: {e}")
        return 0.0, 0.0, 0.0

