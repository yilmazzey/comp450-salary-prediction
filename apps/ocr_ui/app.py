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
ASSET_IMAGE_PATH = PROJECT_ROOT / "pngegg.png"

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


def inject_global_styles() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
            .stApp {
                background: linear-gradient(180deg, #fff9f3 0%, #ffe7d8 55%, #ffd7b8 100%);
                font-family: 'Poppins', 'Helvetica Neue', Arial, sans-serif;
            }
            .stApp [data-testid="stSidebar"] {
                background-color: rgba(255, 255, 255, 0.75);
                backdrop-filter: blur(6px);
            }
            .top-nav {
                max-width: 960px;
                margin: 0 auto 1.5rem auto;
                position: sticky;
                top: 0;
                z-index: 100;
                background: transparent;
                padding: 0.75rem 1.25rem;
                border-radius: 1.5rem;
                box-shadow: none;
                display: flex;
                gap: 1.5rem;
                align-items: center;
                justify-content: center;
            }
            .top-nav a {
                text-decoration: none;
                font-weight: 600;
                color: #ff7a00;
                transition: color 0.3s ease;
            }
            .top-nav a:hover {
                color: #ff4d00;
            }
            .main-container {
                max-width: 960px;
                margin: 0 auto;
            }
            .hero-card {
                background: transparent;
                padding: 2rem 0 1rem 0;
                border-radius: 1.75rem;
                box-shadow: none;
            }
            .section-anchor {
                position: relative;
                top: -80px;
            }
            .info-pill {
                display: inline-block;
                padding: 0.35rem 0.9rem;
                border-radius: 999px;
                background: rgba(255, 135, 25, 0.18);
                color: #a35514;
                font-weight: 600;
                margin-bottom: 0.75rem;
            }
            .about-card {
                background: transparent;
                padding: 1.5rem 0;
                border-radius: 1.5rem;
                box-shadow: none;
                border: 1px solid rgba(255, 122, 0, 0.35);
            }
            .team-grid {
                display: flex;
                gap: 2rem;
                flex-wrap: wrap;
                justify-content: center;
            }
            .team-card {
                min-width: 210px;
                background: rgba(255, 255, 255, 0.35);
                padding: 1.25rem 1.5rem;
                border-radius: 1.25rem;
                text-align: center;
                box-shadow: 0 6px 18px rgba(255, 122, 0, 0.15);
                backdrop-filter: blur(2px);
            }
            .team-card h4 {
                margin-bottom: 0.35rem;
                font-weight: 600;
            }
            .team-card a {
                display: inline-flex;
                align-items: center;
                gap: 0.4rem;
                text-decoration: none;
                color: #0a66c2;
                font-weight: 500;
            }
            .team-card a:hover {
                color: #084a8f;
            }
            .linkedin-icon {
                width: 20px;
                height: 20px;
            }
            h1, h2, h3, .stMarkdown, .stButton > button, .stTextInput, .stSelectbox, label {
                font-family: 'Poppins', 'Helvetica Neue', Arial, sans-serif;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


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
    inject_global_styles()

    st.markdown(
        """
        <div class="top-nav">
            <a href="#top">Home</a>
            <a href="#predictor">Predictor</a>
            <a href="#about">About Us</a>
        </div>
        <div id="top" class="section-anchor"></div>
        <div class="main-container">
        """,
        unsafe_allow_html=True,
    )

    hero_container = st.container()
    with hero_container:
        st.markdown('<div class="hero-card">', unsafe_allow_html=True)
        cols = st.columns([1.6, 0.9])
        with cols[0]:
            st.markdown('<div class="info-pill">Stack Overflow 2025 Salary Insights</div>', unsafe_allow_html=True)
            st.title("From CV to Salary Insights")
            st.markdown(
                "Upload a resume, let DeepSeek OCR highlight core developer traits, and plug them into our "
                "HistGradientBoostingRegressor (log-target) trained on Stack Overflow’s 2025 survey to estimate annual compensation."
            )
        with cols[1]:
            if ASSET_IMAGE_PATH.exists():
                st.image(Image.open(ASSET_IMAGE_PATH), width=200)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div id="predictor" class="section-anchor"></div>', unsafe_allow_html=True)
    st.header("Predictor")
    st.caption("Run DeepSeek OCR, refine the parsed profile, and request a salary estimate in USD.")

    upload_col, info_col = st.columns([1, 2])
    with upload_col:
        st.subheader("Upload CV")
        uploaded = st.file_uploader("Choose a file", type=["pdf", "png", "jpg", "jpeg", "webp"])
        max_pages = st.number_input("Max PDF pages to OCR", min_value=1, max_value=5, value=2)
        run_ocr = st.button("Run OCR", type="primary")
    with info_col:
        st.subheader("How it works")
        st.markdown(
            "1. Upload a PDF or image CV (up to 5 pages).\n"
            "2. DeepSeek OCR extracts text via an Ollama endpoint.\n"
            "3. We pre-fill developer profile fields, which you can review before predicting salary."
        )

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

    st.markdown('<div id="about" class="section-anchor"></div>', unsafe_allow_html=True)
    st.header("About Us")
    st.caption("Meet the team behind the salary prediction project.")

    linkedin_svg = (
        """
        <svg class="linkedin-icon" viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
            <path d="M20.447 20.452H16.89v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.448-2.136 2.944v5.662H9.337V9h3.412v1.561h.047c.476-.9 1.637-1.852 3.372-1.852 3.605 0 4.27 2.373 4.27 5.463v6.28zM5.337 7.433c-1.1 0-1.99-.892-1.99-1.99 0-1.1.89-1.99 1.99-1.99 1.099 0 1.99.89 1.99 1.99 0 1.098-.891 1.99-1.99 1.99zM6.99 20.452H3.683V9H6.99v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.225 0z"/>
        </svg>
        """
    )

    with st.container():
        st.markdown(
            f"""
            <div class="about-card">
                <div class="team-grid">
                    <div class="team-card">
                        <h4>Zeynep Yılmaz</h4>
                        <a href="https://tr.linkedin.com/in/yilmazzey" target="_blank">{linkedin_svg}LinkedIn</a>
                    </div>
                    <div class="team-card">
                        <h4>Zehra Mert</h4>
                        <a href="https://tr.linkedin.com/in/zehramert8" target="_blank">{linkedin_svg}LinkedIn</a>
                    </div>
                    <div class="team-card">
                        <h4>Onat Sarıbıyık</h4>
                        <a href="https://tr.linkedin.com/in/onat-saribiyik-129671249" target="_blank">{linkedin_svg}LinkedIn</a>
                    </div>
                </div>
                <p style="margin-top:1.25rem; text-align:center;">This project combines modern OCR techniques with gradient boosting to deliver quick salary insights for developers worldwide.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

