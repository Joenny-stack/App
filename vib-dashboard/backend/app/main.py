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

# ==============================================
#  PREDICTION UTILITIES
# ==============================================
BUFFER_LOCK = asyncio.Lock()
SENSOR_BUFFERS = {}
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
    if scaler:
        obs = scaler.transform(obs.reshape(1, -1))[0]
    return obs.reshape(1, -1)


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
        y_pred = model.predict(obs)[0]
        # try to produce a human-friendly label
        try:
            pred_int = int(y_pred)
        except Exception:
            pred_int = None

        label = None
        # prefer a label_map from the loaded pickle if present
        if 'label_map' in globals() and globals().get('label_map'):
            label = globals().get('label_map').get(pred_int) if pred_int is not None else None
        # fallback default mapping for common 3-state HMMs
        if label is None:
            DEFAULT_LABEL_MAP = {0: "Healthy", 1: "Degraded", 2: "Faulty"}
            label = DEFAULT_LABEL_MAP.get(pred_int, None)

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

