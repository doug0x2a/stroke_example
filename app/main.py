from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import joblib
import pandas as pd
from typing import Any, Dict
from pathlib import Path
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------
# Load model once when the module is imported (startup)
# ---------------------------------------------------------------------

LINEAR_REGRESSION_MODEL = joblib.load("models/log_reg_model.joblib")

# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------
def validate_payload(payload: Dict[str, Any]) -> None:
    """
    Validate the input payload for stroke prediction.
    Raises ValueError if something is missing or invalid.
    """
    # expected schema
    expected_fields = {
        "gender": ["Male", "Female", "Other"],
        "age": (0, 120),
        "ever_married": ["Yes", "No"],
        "work_type": ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
        "Residence_type": ["Urban", "Rural"],
        "avg_glucose_level": (0, 400),
        "bmi": (10, 100),
        "smoking_status": ["formerly smoked", "never smoked", "smokes"],
        "hypertension": [0,1],
        "heart_disease": [0,1],
    }

    # check for missing keys
    missing = [k for k in expected_fields if k not in payload]
    if missing:
        raise ValueError(f"Missing keys in payload: {missing}")

    # check for unexpected keys
    extras = [k for k in payload if k not in expected_fields]
    if extras:
        raise ValueError(f"Unexpected keys in payload: {extras}")

    # type and value validation
    validated = {}
    for key, rule in expected_fields.items():
        value = payload[key]
        if isinstance(rule, tuple):  # numeric range
            if not isinstance(value, (int, float)):
                raise ValueError(f"{key} must be numeric, got {type(value).__name__}")
            low, high = rule
            if not (low <= value <= high):
                raise ValueError(f"{key}={value} outside plausible range {rule}")
            validated[key] = float(value)
        elif isinstance(rule, list):  # categorical choices
            if value not in rule:
                raise ValueError(f"{key} must be one of {rule}, got '{value}'")
            validated[key] = value
        else:
            raise ValueError(f"Internal schema error for {key}")


# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------

def get_stroke_prob(model: Pipeline, payload: Dict[str, Any]) -> float:
    """Validate input and return stroke probability."""
    validate_payload(payload)

    X = pd.DataFrame([payload])
    proba = float(model.predict_proba(X)[:, 1][0])
    return proba

# ---------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------
app = FastAPI(title="Stroke Predictor")


@app.get("/", response_class=HTMLResponse)
def home():
    with open("app/static/index.html") as f:
        return f.read()

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: Dict[str, Any]):
    try:
        p = get_stroke_prob(LINEAR_REGRESSION_MODEL, payload)
        return {"stroke_probability": p}
    except ValueError as ve:
        # Validation errors -> 400
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        # Anything else -> 500
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
