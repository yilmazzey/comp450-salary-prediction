"""
Streamlit OCR + salary prediction demo.
 - OCR via Hugging Face DeepSeek OCR pipeline.
 - Allows uploading a CV (PDF/image), extracts text, parses key fields,
   lets the user edit, and then predicts salary using existing model artifacts.
Run: streamlit run apps/ocr_ui/app.py
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image
from pdf2image import convert_from_bytes

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = DATA_DIR / "best_model.joblib"
FEATURE_META_PATH = DATA_DIR / "so_2025_feature_columns.json"

DEFAULT_FEATURES = [
    "Country",
    "EdLevelSimplified",
    "YearsCodeNum",
    "DevTypePrimary",
    "RemoteCategory",
]

COMMON_COUNTRIES = [
    "United States",
    "Germany",
    "United Kingdom",
    "Canada",
    "France",
    "India",
    "Australia",
    "Netherlands",
    "Sweden",
]

REMOTE_CHOICES = [
    "Remote",
    "Hybrid-Flexible",
    "Hybrid-InPerson",
    "Hybrid-Choice",
    "In-person",
]



@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model not found at {MODEL_PATH}")
        st.stop()
    model = joblib.load(MODEL_PATH)
    return model


@lru_cache(maxsize=1)
def load_feature_columns() -> List[str]:
    if FEATURE_META_PATH.exists():
        meta = json.loads(FEATURE_META_PATH.read_text())
        cols = meta.get("feature_columns", [])
        if cols:
            return cols
    # fallback: infer from model if available
    return []


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-ocr")


def ocr_image_with_ollama(img: Image.Image) -> str:
    """
    Run OCR on a single image using an Ollama-hosted DeepSeek-OCR model.
    Expects `ollama run deepseek-ocr` (or a server with that model) to be available.
    """
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": "You are an OCR system. Extract all readable text from this CV and return plain text only.",
        "images": [b64],
        "stream": False,
    }

    try:
        resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=300)
        resp.raise_for_status()
    except Exception as e:
        st.error(f"OCR request to Ollama failed: {e}")
        return ""

    try:
        data = resp.json()
    except ValueError:
        st.error("Failed to parse OCR response from Ollama.")
        return ""

    # Ollama /api/generate returns {"response": "...", ...}
    return str(data.get("response", ""))


def extract_text_from_file(uploaded_file: bytes, mime: str, max_pages: int = 2) -> Tuple[str, List[Image.Image]]:
    images: List[Image.Image] = []
    text_chunks: List[str] = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    if "pdf" in mime:
        try:
            status_text.text("Converting PDF pages to images...")
            # Lower DPI to reduce memory and speed up processing
            pages = convert_from_bytes(
                uploaded_file,
                first_page=1,
                last_page=max_pages,
                dpi=96,
            )
            images = pages
            progress_bar.progress(0.3)
        except Exception as e:
            st.warning(f"PDF to image conversion failed ({e}); unable to process this file.")
            progress_bar.empty()
            status_text.empty()
            return "", []
    else:
        try:
            images = [Image.open(io.BytesIO(uploaded_file))]
        except Exception as e:
            st.error(f"Could not read image: {e}")
            progress_bar.empty()
            status_text.empty()
            return "", []

    if not images:
        progress_bar.empty()
        status_text.empty()
        return "", []

    for i, img in enumerate(images):
        status_text.text(f"Processing page {i + 1}/{len(images)}...")

        # Resize large images to avoid excessive memory usage
        max_size = 1024
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        # Call Ollama-based DeepSeek OCR
        result = ocr_image_with_ollama(img)
        if result:
            text_chunks.append(result)
        else:
            st.warning(f"OCR produced no text for page {i + 1}.")

        progress = 0.3 + 0.7 * (i + 1) / len(images)
        progress_bar.progress(min(progress, 1.0))

    progress_bar.empty()
    status_text.empty()

    return "\n".join(text_chunks), images


def parse_years(text: str) -> float | None:
    # Look for patterns like "X years", "X yrs", numbers near 'experience'
    m = re.search(r"(\\d{1,2})\\s*(?:years|yrs|year)", text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def parse_country(text: str) -> str | None:
    for c in COMMON_COUNTRIES:
        if re.search(rf"\\b{re.escape(c)}\\b", text, re.IGNORECASE):
            return c
    return None


def parse_edlevel(text: str) -> str | None:
    patterns = {
        "Bachelor’s degree": [r"bachelor", r"b\\.sc", r"bs", r"b\\.eng"],
        "Master’s degree": [r"master", r"m\\.sc", r"ms", r"mba"],
        "Post-grad": [r"phd", r"doctor"],
        "Some college": [r"associate", r"college"],
    }
    for label, pats in patterns.items():
        for p in pats:
            if re.search(p, text, re.IGNORECASE):
                return label
    return None


def parse_devtype(text: str) -> str | None:
    pats = {
        "Developer, full-stack": ["full[- ]stack"],
        "Developer, back-end": ["back[- ]end"],
        "Developer, front-end": ["front[- ]end"],
        "Data scientist or machine learning specialist": ["data scientist", "machine learning"],
        "Engineer, data": ["data engineer"],
        "Engineering manager": ["manager", "lead", "director", "vp", "c-suite", "cto", "cio"],
        "Developer, mobile": ["mobile", "ios", "android"],
    }
    for label, substrs in pats.items():
        for s in substrs:
            if re.search(s, text, re.IGNORECASE):
                return label
    return None


def build_feature_vector(form_data: Dict[str, Any], feature_columns: List[str]) -> pd.DataFrame:
    # Start with required fields
    base = {
        "Country": form_data.get("Country") or "United States",
        "EdLevelSimplified": form_data.get("EdLevelSimplified") or "Bachelor’s degree",
        "YearsCodeNum": float(form_data.get("YearsCodeNum") or 3.0),
        "DevTypePrimary": form_data.get("DevTypePrimary") or "Developer, full-stack",
        "RemoteCategory": form_data.get("RemoteCategory") or "Hybrid-Choice",
    }
    df = pd.DataFrame([base])
    encoded = pd.get_dummies(df, columns=["Country", "EdLevelSimplified", "DevTypePrimary", "RemoteCategory"], drop_first=True)

    # Align to training columns
    missing = list(set(feature_columns) - set(encoded.columns))
    if missing:
        zeros_df = pd.DataFrame(0.0, index=encoded.index, columns=missing)
        encoded = pd.concat([encoded, zeros_df], axis=1)

    extra = list(set(encoded.columns) - set(feature_columns))
    if extra:
        encoded = encoded.drop(columns=extra)

    encoded = encoded[feature_columns]
    return encoded


def main():
    st.set_page_config(page_title="OCR + Salary Prediction", layout="wide")
    st.title("OCR CV → Salary Prediction")

    st.markdown(
        "Upload a CV (PDF/image), run DeepSeek OCR, review parsed fields, and predict salary "
        "with the trained StackOverflow salary model."
    )

    with st.sidebar:
        st.header("Upload")
        uploaded = st.file_uploader("CV file", type=["pdf", "png", "jpg", "jpeg", "webp"])
        max_pages = st.number_input("Max PDF pages to OCR", min_value=1, max_value=5, value=2)
        run_ocr = st.button("Run OCR", type="primary")

    if "ocr_text" not in st.session_state:
        st.session_state.ocr_text = None
    if "ocr_images" not in st.session_state:
        st.session_state.ocr_images = None

    if uploaded and run_ocr:
        raw_bytes = uploaded.read()
        text, images = extract_text_from_file(raw_bytes, uploaded.type, max_pages=max_pages)
        if not text.strip():
            st.error("No text extracted.")
        else:
            st.session_state.ocr_text = text
            st.session_state.ocr_images = images

    if st.session_state.ocr_text:
        text = st.session_state.ocr_text
        st.subheader("OCR Extracted Text (truncated)")
        st.text_area("Text", value=text[:5000], height=200)

        # Initial heuristics
        parsed = {
            "Country": parse_country(text),
            "EdLevelSimplified": parse_edlevel(text),
            "YearsCodeNum": parse_years(text),
            "DevTypePrimary": parse_devtype(text),
            "RemoteCategory": None,
        }

        st.subheader("Review & Edit Parsed Fields")
        col1, col2 = st.columns(2)
        with col1:
            country_idx = COMMON_COUNTRIES.index(parsed["Country"]) if parsed["Country"] in COMMON_COUNTRIES else len(COMMON_COUNTRIES)
            country = st.selectbox("Country", options=COMMON_COUNTRIES + ["Other"], index=country_idx)
            edlevel = st.selectbox(
                "Education",
                options=[
                    "Bachelor’s degree",
                    "Master’s degree",
                    "Post-grad",
                    "Some college",
                    "Less than secondary",
                    "Self-taught/other",
                ],
                index=0, # Simplified index selection
            )
            devtype = st.text_input("Primary Dev Type", value=parsed["DevTypePrimary"] or "Developer, full-stack")
        with col2:
            years = st.number_input("Years of experience", min_value=0.0, max_value=60.0, value=float(parsed["YearsCodeNum"] or 3.0), step=0.5)
            remote = st.selectbox("Remote category", options=REMOTE_CHOICES, index=2)

        form_data = {
            "Country": country if country != "Other" else parsed.get("Country") or "United States",
            "EdLevelSimplified": edlevel,
            "YearsCodeNum": years,
            "DevTypePrimary": devtype,
            "RemoteCategory": remote,
        }

        st.subheader("Predict")
        feature_cols = load_feature_columns()
        if not feature_cols:
            st.error("Feature columns metadata missing; cannot build feature vector.")
            return

        X = build_feature_vector(form_data, feature_cols)
        model = load_model()

        if st.button("Predict salary", type="primary"):
            pred = model.predict(X)[0]
            st.success(f"Predicted salary (USD/year): ${pred:,.0f}")
            st.caption("Model: best_model.joblib (StackOverflow salary). This is a point estimate; actual salaries vary.")

        st.subheader("Downloads")
        parsed_df = pd.DataFrame([form_data])
        st.download_button("Download parsed fields (CSV)", data=parsed_df.to_csv(index=False), file_name="parsed_fields.csv")

        # Reuse prediction for download if model is loaded
        current_pred = model.predict(X)[0]
        pred_vec = {"prediction_usd": current_pred}
        st.download_button("Download prediction (JSON)", data=json.dumps(pred_vec, indent=2), file_name="prediction.json")


if __name__ == "__main__":
    main()

