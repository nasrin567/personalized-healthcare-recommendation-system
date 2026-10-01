"""
Personalized Healthcare & Medicine Recommendation System — Streamlit dashboard.

Educational / informational prototype. Everything shown here is computed from the
project's own files:

  data/Training.csv          -> symptom feature names (132) + disease target ("prognosis", 41 classes)
  models/best_model.pkl      -> saved SVC disease model (NOT retrained here)
  models/disease_encoder.pkl -> saved LabelEncoder for the 41 disease names
  data/medications.csv       -> disease-wise medication information
  data/Cleaned_Dataset.csv   -> separate patient-profile / risk dataset (never merged with Training.csv)
  data/Medicine_Details.csv  -> medicine-level text used for TF-IDF + cosine similarity

Run with:  streamlit run app.py
"""
from __future__ import annotations

import ast
import html
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Personalized Healthcare",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent

DISCLAIMER = (
    "This system is intended for educational and informational purposes only. It does not "
    "provide medical diagnosis, prescribe medication, or replace advice from a qualified "
    "healthcare professional."
)
SIMILARITY_NOTE = (
    "Similarity score represents textual similarity, not clinical equivalence or interchangeability."
)

# ---------------------------------------------------------------------------
# Disease-name normalisation between Training.csv/encoder and medications.csv.
# Verified in notebooks 03/04: after .strip(), only these two names still differ
# (spelling typos in the source data). "Diabetes " and "Hypertension " are fixed
# by .strip() alone. No other mapping is applied.
# ---------------------------------------------------------------------------
DISEASE_NAME_FIXES = {
    "Peptic ulcer diseae": "Peptic ulcer disease",
    "(vertigo) Paroymsal  Positional Vertigo": "(vertigo) Paroymsal Positional Vertigo",
}


def normalise_disease(name: str) -> str:
    name = str(name).strip()
    return DISEASE_NAME_FIXES.get(name, name)


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
CSS = """
<style>
#MainMenu, footer {visibility:hidden}
.block-container{padding-top:1.6rem;padding-bottom:3rem;max-width:1320px}
:root{--ink:#0f2a43;--teal:#0e9aa7;--blue:#1f4e79;--soft:rgba(14,154,167,.10);--line:rgba(128,128,128,.28)}

.hero{padding:42px 40px;border-radius:22px;margin-bottom:26px;color:#fff;
  background:radial-gradient(1200px 300px at 90% -20%,rgba(14,154,167,.55),transparent),
             linear-gradient(135deg,#0f2a43 0%,#1f4e79 100%);
  box-shadow:0 10px 30px rgba(15,42,67,.18)}
.hero-badge{display:inline-block;padding:6px 14px;border-radius:999px;background:rgba(255,255,255,.15);
  font-size:13px;letter-spacing:.3px;margin-bottom:14px}
.hero h1{font-size:40px;line-height:1.15;margin:0 0 12px 0;color:#fff;padding:0}
.hero p{font-size:17px;line-height:1.6;color:#d9eaf7;max-width:880px;margin:0}

.section-title{font-size:24px;font-weight:700;margin:6px 0 4px 0}
.muted{color:#8b98a5;font-size:14px}

.card{padding:20px 22px;border-radius:16px;border:1px solid var(--line);background:rgba(128,128,128,.06);margin:10px 0}
.card h4{margin:0 0 8px 0;font-size:17px}
.stat{padding:18px 20px;border-radius:16px;border:1px solid var(--line);background:rgba(128,128,128,.06)}
.stat .v{font-size:30px;font-weight:700;line-height:1.1}
.stat .l{font-size:13px;color:#8b98a5;text-transform:uppercase;letter-spacing:.6px;margin-top:4px}
.stat .s{font-size:12px;color:#8b98a5;margin-top:6px}

.result{padding:26px 28px;border-radius:20px;border:1px solid var(--teal);
  background:linear-gradient(135deg,rgba(14,154,167,.14),rgba(31,78,121,.10))}
.result .lab{font-size:13px;text-transform:uppercase;letter-spacing:.8px;color:#8b98a5}
.result .disease{font-size:34px;font-weight:800;margin:4px 0 0 0;line-height:1.15}

.chip{display:inline-block;padding:4px 12px;margin:3px 4px 3px 0;border-radius:999px;font-size:13px;
  background:var(--soft);border:1px solid rgba(14,154,167,.35)}
.chip.med{background:rgba(31,78,121,.12);border-color:rgba(31,78,121,.4);font-size:14px;padding:6px 14px}

.flow{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:8px 0 4px 0}
.flow-step{padding:10px 14px;border-radius:12px;border:1px solid var(--line);background:rgba(128,128,128,.07);
  font-size:14px;font-weight:600}
.flow-arrow{opacity:.55;font-size:18px}

.med-card{padding:18px 20px;border-radius:16px;border:1px solid var(--line);background:rgba(128,128,128,.06);margin:12px 0}
.med-top{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;align-items:baseline}
.med-name{font-size:18px;font-weight:700}
.rank{display:inline-block;min-width:28px;text-align:center;padding:2px 8px;margin-right:8px;border-radius:8px;
  background:var(--blue);color:#fff;font-size:13px}
.score{font-weight:700;color:var(--teal)}
.bar{height:7px;border-radius:99px;background:rgba(128,128,128,.25);margin:8px 0 12px 0;overflow:hidden}
.bar>div{height:100%;background:linear-gradient(90deg,var(--teal),var(--blue))}
.kv{font-size:14px;line-height:1.55;margin:4px 0}
.kv b{display:inline-block;min-width:104px;color:#8b98a5;font-weight:600}
.med-img{float:right;width:84px;height:84px;object-fit:contain;border-radius:10px;margin-left:12px;background:#fff}

.disclaimer{padding:16px 18px;border-radius:14px;border-left:5px solid #d97706;background:rgba(217,119,6,.10);
  margin-top:20px;font-size:14.5px;line-height:1.55}
.footer{text-align:center;color:#8b98a5;padding:28px 0 6px;font-size:13px}

@media (max-width:760px){
  .hero{padding:26px 20px}.hero h1{font-size:28px}.hero p{font-size:15px}
  .result .disease{font-size:26px}.kv b{display:block;min-width:0}
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def esc(value) -> str:
    """HTML-escape any dataset text before putting it inside custom HTML."""
    return html.escape(str(value), quote=True)


def find_file(name: str, subdir: str) -> Path:
    """Prefer <subdir>/<name>; also accept the file next to app.py."""
    for candidate in (BASE_DIR / subdir / name, BASE_DIR / name):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"'{name}' was not found. Expected it at: {BASE_DIR / subdir / name}")


def require_columns(df: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"{source} is missing expected column(s) {missing}. "
            f"Columns found: {list(df.columns)[:12]}{'...' if df.shape[1] > 12 else ''}"
        )


def pretty_symptom(name: str) -> str:
    return re.sub(r"\s+", " ", name.replace("_", " ")).strip().title()


def hero(badge: str, title: str, text: str) -> None:
    st.markdown(
        f'<div class="hero"><div class="hero-badge">{esc(badge)}</div><h1>{title}</h1><p>{esc(text)}</p></div>',
        unsafe_allow_html=True,
    )


def stat_card(value, label: str, sub: str = "") -> str:
    sub_html = f'<div class="s">{esc(sub)}</div>' if sub else ""
    return f'<div class="stat"><div class="v">{esc(value)}</div><div class="l">{esc(label)}</div>{sub_html}</div>'


def flow(steps: list[str]) -> str:
    parts = ['<span class="flow-arrow">→</span>'.join(f'<span class="flow-step">{esc(s)}</span>' for s in steps)]
    return '<div class="flow">' + parts[0] + "</div>"


def disclaimer_box(extra: str = "") -> None:
    extra_html = f"<br>{esc(extra)}" if extra else ""
    st.markdown(f'<div class="disclaimer"><strong>⚠️ Important:</strong> {esc(DISCLAIMER)}{extra_html}</div>',
                unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached loaders (datasets -> cache_data, models / vectorizer -> cache_resource)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_training_summary() -> dict:
    """Read Training.csv once: feature names (all columns except the target) + basic statistics."""
    df = pd.read_csv(find_file("Training.csv", "data"))
    require_columns(df, ["prognosis"], "Training.csv")
    features = [c for c in df.columns if c != "prognosis"]
    X = df[features]
    return {
        "features": features,
        "n_rows": len(df),
        "n_unique_rows": int(len(df.drop_duplicates())),
        "n_classes": int(df["prognosis"].nunique()),
        "never_positive": [c for c in features if X[c].sum() == 0],
        "symptoms_per_row_min": int(X.sum(axis=1).min()),
        "symptoms_per_row_median": float(X.sum(axis=1).median()),
        "symptoms_per_row_max": int(X.sum(axis=1).max()),
        "prognosis_values": sorted(df["prognosis"].unique()),
    }


@st.cache_resource(show_spinner="Loading disease model…")
def load_disease_model():
    """Load the saved model + encoder and verify they match Training.csv. Nothing is retrained."""
    info = load_training_summary()
    features = info["features"]
    model = joblib.load(find_file("best_model.pkl", "models"))
    encoder = joblib.load(find_file("disease_encoder.pkl", "models"))

    if getattr(model, "n_features_in_", len(features)) != len(features):
        raise ValueError(f"Model expects {model.n_features_in_} features but Training.csv has {len(features)}.")
    saved_names = getattr(model, "feature_names_in_", None)
    if saved_names is not None and list(saved_names) != features:
        raise ValueError("Model feature names/order differ from Training.csv columns.")
    if set(encoder.classes_) != set(info["prognosis_values"]):
        raise ValueError("disease_encoder.pkl classes do not match the 'prognosis' values in Training.csv.")
    return model, encoder, features


@st.cache_data(show_spinner=False)
def load_medications() -> pd.DataFrame:
    df = pd.read_csv(find_file("medications.csv", "data"))
    require_columns(df, ["Disease", "Medication"], "medications.csv")
    df["Disease_key"] = df["Disease"].map(normalise_disease)
    return df


@st.cache_data(show_spinner=False)
def load_profile_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(find_file("Cleaned_Dataset.csv", "data"))
    require_columns(
        df, ["disease", "age", "gender", "blood_pressure", "cholesterol_level", "outcome_variable", "risk_level"],
        "Cleaned_Dataset.csv",
    )
    return df, df.drop_duplicates().reset_index(drop=True)


@st.cache_resource(show_spinner="Building TF-IDF index for medicines…")
def load_medicine_index():
    """Same preprocessing as notebook 05: drop duplicate rows, combine 4 text columns, TF-IDF (english stop words)."""
    raw = pd.read_csv(find_file("Medicine_Details.csv", "data"))
    text_cols = ["Medicine Name", "Composition", "Uses", "Side_effects"]
    require_columns(raw, text_cols, "Medicine_Details.csv")
    df = raw.drop_duplicates().reset_index(drop=True).copy()
    for col in text_cols:
        df[col] = df[col].fillna("").astype(str)
    combined = df["Medicine Name"] + " " + df["Composition"] + " " + df["Uses"] + " " + df["Side_effects"]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(combined)
    manufacturer = df["Manufacturer"].fillna("").astype(str) if "Manufacturer" in df.columns else pd.Series([""] * len(df))
    # Some medicine names occur more than once (different manufacturers) -> label includes the manufacturer.
    labels = [f"{n} · {m}" if m else n for n, m in zip(df["Medicine Name"], manufacturer)]
    return df, matrix, labels, len(raw)


def safe(loader):
    """Run a loader; return (result, None) or (None, exception) so one missing file can't break every page."""
    try:
        return loader(), None
    except Exception as exc:  # noqa: BLE001 - shown to the user in a friendly box
        return None, exc


TRAIN, TRAIN_ERR = safe(load_training_summary)
DISEASE, DISEASE_ERR = safe(load_disease_model)
MEDS, MEDS_ERR = safe(load_medications)
PROFILE, PROFILE_ERR = safe(load_profile_data)
MEDICINE, MEDICINE_ERR = safe(load_medicine_index)


def show_load_error(component: str, err: Exception) -> None:
    st.error(f"Could not load the {component}.")
    st.code(f"{type(err).__name__}: {err}")
    st.info("Check that the files are in `data/` and `models/` next to app.py (see README.md), "
            "and that scikit-learn matches requirements.txt (the model was saved with 1.7.2).")


# ---------------------------------------------------------------------------
# Disease prediction + medication lookup
# ---------------------------------------------------------------------------
def run_prediction(model, encoder, features: list[str], selected: list[str]) -> dict:
    """Build the 132-column 0/1 row exactly as in Training.csv, predict, decode."""
    X = pd.DataFrame(0, index=[0], columns=features)
    X.loc[0, selected] = 1
    encoded = model.predict(X)[0]
    disease = encoder.inverse_transform([encoded])[0]

    confidence, top3 = None, []
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        classes = list(model.classes_)
        # Probability of the class that predict() returned (SVC's probabilities are Platt-scaled and
        # can in rare cases rank a different class first, so we report the predicted class's own value).
        confidence = float(probs[classes.index(encoded)])
        for i in np.argsort(probs)[::-1][:3]:
            top3.append((encoder.inverse_transform([classes[i]])[0].strip(), float(probs[i])))
    return {"disease": disease, "confidence": confidence, "top3": top3}


def parse_medication_list(raw: str) -> list[str]:
    """medications.csv stores each list as a Python-list string, e.g. "['A', 'B']"."""
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return [str(raw)]
    return [str(v) for v in value] if isinstance(value, (list, tuple)) else [str(value)]


def lookup_medication(disease: str, meds_df: pd.DataFrame) -> tuple[str, list[str] | None]:
    key = normalise_disease(disease)
    rows = meds_df[meds_df["Disease_key"] == key]
    return key, (parse_medication_list(rows["Medication"].iloc[0]) if not rows.empty else None)


def toggle_symptom(symptom: str) -> None:
    selected = st.session_state.selected_symptoms
    (selected.add if st.session_state.get(f"cb_{symptom}") else selected.discard)(symptom)

def clear_symptoms() -> None:
    st.session_state.selected_symptoms = set()

    for key in list(st.session_state.keys()):
        if key.startswith("cb_"):
            st.session_state[key] = False

def profile_reference(profile_df: pd.DataFrame, age: int, gender: str, bp, chol) -> pd.DataFrame:
    """Descriptive only: records in Cleaned_Dataset.csv with the same gender / BP code / cholesterol code, age ±10."""
    mask = (
        (profile_df["gender"] == gender)
        & (profile_df["blood_pressure"] == bp)
        & (profile_df["cholesterol_level"] == chol)
        & (profile_df["age"].between(age - 10, age + 10))
    )
    return profile_df[mask]


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_diagnosis() -> None:
    hero("🩺 Educational prototype", "Disease Prediction from Symptoms",
         "Select the symptoms, run the saved disease model, and view the disease-wise medication information "
         "stored in the project's medication dataset.")
    if DISEASE_ERR:
        show_load_error("disease model / Training.csv", DISEASE_ERR)
        return
    model, encoder, features = DISEASE

    st.session_state.setdefault("selected_symptoms", set())

    # --- Patient information ------------------------------------------------
    st.markdown('<div class="section-title">1 · Patient information</div>', unsafe_allow_html=True)
    st.caption("These profile fields are **not inputs to the disease model** — the model uses only the "
               f"{len(features)} symptom features below. They are shown with the prediction as context.")
    if PROFILE_ERR:
        gender_options, bp_options, chol_options = ["female", "male"], [0, 1, 2], [0, 1, 2]
        st.warning("Cleaned_Dataset.csv could not be loaded, so default profile options are shown.")
    else:
        pdf = PROFILE[1]
        gender_options = sorted(pdf["gender"].unique())
        bp_options = sorted(pdf["blood_pressure"].unique())
        chol_options = sorted(pdf["cholesterol_level"].unique())
    level = lambda v: f"Level {v}" + {0: " (lowest)", 2: " (highest)"}.get(v, "")  # noqa: E731

    c1, c2, c3, c4 = st.columns(4)
    age = c1.number_input("Age", min_value=1, max_value=120, value=30, step=1)
    gender = c2.selectbox("Gender", gender_options, format_func=str.capitalize)
    bp = c3.selectbox("Blood pressure", bp_options, index=min(1, len(bp_options) - 1), format_func=level,
                      help="Cleaned_Dataset.csv stores blood pressure as ordinal codes 0/1/2 (scaled values rise "
                           "with the code). The project files do not define the clinical meaning of each level.")
    chol = c4.selectbox("Cholesterol", chol_options, index=min(1, len(chol_options) - 1), format_func=level,
                        help="Stored as ordinal codes 0/1/2 in Cleaned_Dataset.csv; clinical thresholds are not "
                             "documented in the project files.")

    # --- Symptom selection --------------------------------------------------
    st.markdown('<div class="section-title">2 · Select symptoms</div>', unsafe_allow_html=True)
    selected = sorted(st.session_state.selected_symptoms)
    if TRAIN:
        st.caption(
            f"Training records contain {TRAIN['symptoms_per_row_min']}–{TRAIN['symptoms_per_row_max']} symptoms "
            f"(median {TRAIN['symptoms_per_row_median']:.0f}); predictions from very few symptoms are less reliable.")

    f1, f2, f3 = st.columns([3, 1.3, 1])
    query = f1.text_input("Search symptoms", placeholder="e.g. fever, cough, headache…", label_visibility="collapsed")
    only_selected = f2.toggle("Show selected only", value=False)
    f3.button("↺ Clear all", on_click=clear_symptoms, width="stretch")

    q = query.strip().lower()
    visible = [s for s in features if (not q or q in s.lower() or q in pretty_symptom(s).lower())
               and (not only_selected or s in st.session_state.selected_symptoms)]
    st.caption(f"Showing {len(visible)} of {len(features)} symptoms · **{len(selected)} selected**")

    with st.container(height=360, border=True):
        if not visible:
            st.write("No symptoms match your search.")
        cols = st.columns(3)
        for i, symptom in enumerate(visible):
            with cols[i % 3]:
                st.checkbox(pretty_symptom(symptom), value=symptom in st.session_state.selected_symptoms,
                            key=f"cb_{symptom}", on_change=toggle_symptom, args=(symptom,))

    selected = sorted(st.session_state.selected_symptoms)
    if selected:
        chips = "".join(f'<span class="chip">{esc(pretty_symptom(s))}</span>' for s in selected)
        st.markdown(f"**Selected symptoms ({len(selected)})**<br>{chips}", unsafe_allow_html=True)
    else:
        st.info("No symptoms selected yet.")

    # --- Prediction ---------------------------------------------------------
    if st.button("🔍 Predict Disease", type="primary", width="stretch"):
        if not selected:
            st.warning("Please select at least one symptom before predicting.")
        else:
            try:
                result = run_prediction(model, encoder, features, selected)
            except Exception as exc:  # noqa: BLE001
                st.error("Prediction failed.")
                st.code(f"{type(exc).__name__}: {exc}")
            else:
                result.update(selected=list(selected), age=age, gender=gender, bp=bp, chol=chol)
                st.session_state.last_prediction = result

    result = st.session_state.get("last_prediction")
    if not result:
        disclaimer_box()
        return

    st.divider()
    st.markdown('<div class="section-title">🧠 AI Prediction Result</div>', unsafe_allow_html=True)
    if result["selected"] != selected:
        st.warning("Your symptom selection has changed since this prediction. Click **Predict Disease** to update it.")

    disease = result["disease"].strip()
    left, mid, right = st.columns([2.2, 1, 1])
    left.markdown(f'<div class="result"><div class="lab">Predicted disease</div><div class="disease">{esc(disease)}</div></div>',
                  unsafe_allow_html=True)
    mid.markdown(stat_card(f"{result['confidence'] * 100:.1f}%" if result["confidence"] is not None else "n/a",
                           "Model confidence", "Probability of predicted class"), unsafe_allow_html=True)
    right.markdown(stat_card(len(result["selected"]), "Symptoms selected", f"of {len(features)} features"),
                   unsafe_allow_html=True)

    if len(result["top3"]) > 1:
        with st.expander("Top-3 model probabilities"):
            top_df = pd.DataFrame(result["top3"], columns=["Disease", "Probability"])
            st.dataframe(top_df, hide_index=True, width="stretch",
                         column_config={"Probability": st.column_config.ProgressColumn(
                             "Probability", format="percent", min_value=0.0, max_value=1.0)})
            st.caption("Probabilities come from the SVC's built-in probability calibration on the provided dataset.")

    st.markdown("##### Patient profile")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Age", result["age"])
    p2.metric("Gender", str(result["gender"]).capitalize())
    p3.metric("Blood pressure", level(result["bp"]))
    p4.metric("Cholesterol", level(result["chol"]))

    if not PROFILE_ERR:
        with st.expander("Reference profile statistics from Cleaned_Dataset.csv (descriptive only)"):
            ref = profile_reference(PROFILE[1], result["age"], result["gender"], result["bp"], result["chol"])
            if ref.empty:
                st.write("No records in the profile dataset share this gender, blood-pressure level, "
                         "cholesterol level and an age within ±10 years.")
            else:
                st.write(f"**{len(ref)}** de-duplicated records share this gender, blood-pressure level, "
                         "cholesterol level and an age within ±10 years.")
                r1, r2 = st.columns(2)
                r1.caption("risk_level in those records")
                r1.bar_chart(ref["risk_level"].value_counts())
                r2.caption("outcome_variable in those records")
                r2.bar_chart(ref["outcome_variable"].value_counts())
            st.caption("This dataset is separate from Training.csv and is not linked to the prediction. "
                       "These counts describe other records — they are not a risk assessment for this patient.")

    # --- Medication information --------------------------------------------
    st.markdown("##### 💊 Medication information")
    if MEDS_ERR:
        show_load_error("medications dataset", MEDS_ERR)
    else:
        key, medications = lookup_medication(disease, MEDS)
        if medications is None:
            st.warning(f"No medication record found in medications.csv for “{key}”.")
        else:
            chips = "".join(f'<span class="chip med">{esc(m)}</span>' for m in medications)
            st.markdown(f'<div class="card"><h4>Disease-wise medication information for {esc(key)}</h4>{chips}</div>',
                        unsafe_allow_html=True)
            if key != disease:
                st.caption(f"Matched using the documented spelling fix “{disease}” → “{key}”.")
        st.caption("Source: medications.csv (disease-wise). It is not a patient-specific prescription, and the "
                   "dataset contains no dosage information.")
    disclaimer_box("The prediction is based on the provided Training.csv dataset. Medication information is "
                   "retrieved from the provided disease-wise dataset and must not be used for self-medication.")


def render_medicine_card(row: pd.Series, rank: int, score: float, show_image: bool) -> str:
    img = ""
    if show_image and "Image URL" in row and str(row["Image URL"]).startswith("http"):
        img = f'<img class="med-img" loading="lazy" src="{esc(row["Image URL"])}" alt="">'
    maker = f'<div class="kv"><b>Manufacturer</b>{esc(row["Manufacturer"])}</div>' if row.get("Manufacturer") else ""
    pct = max(0.0, min(score, 1.0)) * 100
    return (
        f'<div class="med-card">{img}'
        f'<div class="med-top"><span class="med-name"><span class="rank">#{rank}</span>{esc(row["Medicine Name"])}</span>'
        f'<span class="score">{pct:.1f}% text similarity</span></div>'
        f'<div class="bar"><div style="width:{pct:.1f}%"></div></div>'
        f'<div class="kv"><b>Composition</b>{esc(row["Composition"])}</div>'
        f'<div class="kv"><b>Uses</b>{esc(row["Uses"])}</div>'
        f'<div class="kv"><b>Side effects</b>{esc(row["Side_effects"])}</div>{maker}</div>'
    )


def page_similarity() -> None:
    hero("💊 TF-IDF + Cosine Similarity", "Medicine Similarity",
         "Find medicines whose name, composition, uses and side-effect text is similar to a selected medicine.")
    if MEDICINE_ERR:
        show_load_error("Medicine_Details.csv", MEDICINE_ERR)
        return
    df, matrix, labels, n_raw = MEDICINE
    st.info(SIMILARITY_NOTE + " Review percentages in the dataset are not used and no medicine is ranked by effectiveness.")

    c1, c2 = st.columns([3, 1])
    idx = c1.selectbox("Search / select a medicine", options=list(range(len(df))), index=0,
                       format_func=lambda i: labels[i],
                       help="Type to search. Names that occur more than once are shown with their manufacturer.")
    top_n = c2.slider("Number of similar medicines", 3, 10, 5)
    show_images = st.checkbox("Show product images (loaded from the Image URL column on the web)", value=False)

    if st.button("🔎 Find Similar Medicines", type="primary", width="stretch"):
        if not 0 <= idx < len(df):
            st.error("Medicine not found.")
        else:
            # Only the selected row is compared against the matrix (no full N×N similarity matrix).
            scores = cosine_similarity(matrix[idx], matrix).ravel()
            scores[idx] = -1.0
            top = np.argpartition(scores, -top_n)[-top_n:]
            top = top[np.argsort(scores[top])[::-1]]
            st.session_state.sim_result = {"idx": idx, "top": top.tolist(), "scores": scores[top].tolist()}

    res = st.session_state.get("sim_result")
    if not res or res["idx"] >= len(df):
        return
    query_row = df.iloc[res["idx"]]
    st.markdown(f'<div class="section-title">Similar to: {esc(query_row["Medicine Name"])}</div>', unsafe_allow_html=True)
    with st.expander("Selected medicine details"):
        st.write(f"**Composition:** {query_row['Composition']}")
        st.write(f"**Uses:** {query_row['Uses']}")
        st.write(f"**Side effects:** {query_row['Side_effects']}")

    view = st.radio("View", ["Cards", "Table"], horizontal=True, label_visibility="collapsed")
    if view == "Cards":
        st.markdown("".join(render_medicine_card(df.iloc[i], r, s, show_images)
                            for r, (i, s) in enumerate(zip(res["top"], res["scores"]), start=1)),
                    unsafe_allow_html=True)
    else:
        cols = ["Medicine Name", "Composition", "Uses", "Side_effects"] + (["Manufacturer"] if "Manufacturer" in df.columns else [])
        table = df.iloc[res["top"]][cols].copy()
        table.insert(1, "Text similarity", [round(s, 4) for s in res["scores"]])
        st.dataframe(table, hide_index=True, width="stretch",
                     column_config={"Text similarity": st.column_config.ProgressColumn(
                         "Text similarity", format="%.3f", min_value=0.0, max_value=1.0)})
    st.caption(SIMILARITY_NOTE)
    disclaimer_box()


def page_overview() -> None:
    hero("📊 Project overview", "How the system fits together",
         "Live counts and data-quality notes computed from the project's own files.")
    if TRAIN_ERR:
        show_load_error("Training.csv", TRAIN_ERR)
    n_classes = len(DISEASE[1].classes_) if DISEASE else (TRAIN["n_classes"] if TRAIN else "–")
    n_feat = len(DISEASE[2]) if DISEASE else (len(TRAIN["features"]) if TRAIN else "–")
    n_med = f"{len(MEDICINE[0]):,}" if MEDICINE else "–"
    n_prof = f"{len(PROFILE[1]):,}" if PROFILE else "–"

    cols = st.columns(4)
    cols[0].markdown(stat_card(n_classes, "Disease classes", "Training.csv · prognosis"), unsafe_allow_html=True)
    cols[1].markdown(stat_card(n_feat, "Symptom features", "binary 0/1 model inputs"), unsafe_allow_html=True)
    cols[2].markdown(stat_card(n_med, "Medicine records",
                               f"after removing duplicates (raw {MEDICINE[3]:,})" if MEDICINE else ""), unsafe_allow_html=True)
    cols[3].markdown(stat_card(n_prof, "Patient-profile records",
                               f"after removing duplicates (raw {len(PROFILE[0]):,})" if PROFILE else ""), unsafe_allow_html=True)

    st.markdown('<div class="section-title" style="margin-top:22px">🔬 Architecture</div>', unsafe_allow_html=True)
    st.markdown("**Disease prediction → medication information**")
    st.markdown(flow(["Patient symptoms", f"{n_feat} binary features", "Saved disease model", "Predicted disease",
                      "medications.csv", "Medication information"]), unsafe_allow_html=True)
    st.markdown("**Medicine similarity**")
    st.markdown(flow(["Medicine_Details.csv", "Text preprocessing", "TF-IDF", "Cosine similarity",
                      "Similar medicine profiles"]), unsafe_allow_html=True)

    st.markdown('<div class="section-title" style="margin-top:22px">📁 Dataset roles</div>', unsafe_allow_html=True)
    roles = pd.DataFrame({
        "File": ["Training.csv", "medications.csv", "Cleaned_Dataset.csv", "Medicine_Details.csv"],
        "Role": ["Disease prediction (symptoms → prognosis)", "Disease-wise medication information",
                 "Patient-profile / risk-analysis support (separate; not merged with Training.csv)",
                 "Medicine-level text for TF-IDF + cosine similarity"],
        "Used by": ["Saved model inputs & disease labels", "Diagnosis page (lookup)",
                    "Diagnosis page (descriptive reference) · Overview", "Medicine Similarity page"],
        "Records": [f"{TRAIN['n_rows']:,}" if TRAIN else "–", f"{len(MEDS)}" if MEDS is not None else "–",
                    f"{len(PROFILE[0]):,}" if PROFILE else "–", f"{MEDICINE[3]:,}" if MEDICINE else "–"],
    })
    st.dataframe(roles, hide_index=True, width="stretch")

    st.markdown('<div class="section-title" style="margin-top:22px">🤖 Machine-learning components</div>', unsafe_allow_html=True)
    if DISEASE:
        m = DISEASE[0]
        st.write(f"**Disease prediction:** `{type(m).__name__}` (kernel = `{getattr(m, 'kernel', 'n/a')}`, "
                 f"probability estimates = `{getattr(m, 'probability', 'n/a')}`) loaded from `best_model.pkl`; "
                 "labels decoded with `disease_encoder.pkl` (LabelEncoder).")
    st.write("**Medicine similarity:** `TfidfVectorizer(stop_words='english')` over medicine name + composition + uses + "
             "side effects, compared with cosine similarity (same configuration as notebook 05).")


    st.markdown('<div class="section-title" style="margin-top:22px">⚠️ Limitations & data-quality notes</div>', unsafe_allow_html=True)
    notes = ["The reported disease-model accuracy applies only to the provided dataset; it is not real-world diagnostic accuracy.",
             "The medication dataset is disease-wise, not patient-level, so the system retrieves medication information "
             "rather than learning an optimal medicine for an individual.",
             "Medicine similarity is textual similarity, not evidence of clinical equivalence, safety, effectiveness or interchangeability."]
    if TRAIN:
        notes.insert(0, f"Training.csv has {TRAIN['n_rows']:,} rows but only {TRAIN['n_unique_rows']:,} unique rows, so identical "
                        "records are likely to appear in both the training and test splits. This can inflate the "
                        "100% accuracy reported in notebook 04.")
        if TRAIN["never_positive"]:
            notes.append("Symptom feature(s) never set to 1 in Training.csv, so they cannot influence the model: "
                         + ", ".join(pretty_symptom(s) for s in TRAIN["never_positive"]) + ".")
    if PROFILE and DISEASE:
        model_names = {normalise_disease(c).lower() for c in DISEASE[1].classes_}
        shared = {d for d in PROFILE[1]["disease"].unique() if str(d).strip().lower() in model_names}
        notes.append(f"Cleaned_Dataset.csv covers {PROFILE[1]['disease'].nunique()} diseases, but only {len(shared)} share a name "
                     "with the model's classes — one more reason the two datasets are not merged.")
    st.markdown("\n".join(f"- {n}" for n in notes))
    disclaimer_box()


def page_about() -> None:
    hero("ℹ️ About", "Personalized Healthcare & Medicine Recommendation System",
         "An educational machine-learning prototype that connects symptom-based disease prediction, "
         "disease-wise medication information and content-based medicine similarity.")
    a, b = st.columns(2)
    a.markdown("""
<div class="card"><h4>🎯 Objective</h4>Demonstrate an end-to-end data-science workflow: inspect data, train a disease
prediction model, connect its output to medication information, and add a content-based medicine similarity
module — all presented in one interface.</div>
<div class="card"><h4>🧠 Disease prediction</h4>A saved SVC model (<code>best_model.pkl</code>) takes 132 binary symptom
features from <code>Training.csv</code> and predicts one of 41 diseases. Labels are decoded with
<code>disease_encoder.pkl</code>.</div>
<div class="card"><h4>💊 Medication information</h4>The predicted disease is matched to <code>medications.csv</code>,
which lists medications per disease. The dataset has no dosage or patient-level outcome data, so the app only
displays the stored lists.</div>
""", unsafe_allow_html=True)
    b.markdown("""
<div class="card"><h4>🔎 Medicine similarity</h4><code>Medicine_Details.csv</code> text (name, composition, uses,
side effects) is converted to TF-IDF vectors; cosine similarity finds medicines with similar text profiles. It
does not measure clinical equivalence.</div>
<div class="card"><h4>🧾 Patient profile / risk data</h4><code>Cleaned_Dataset.csv</code> is a separate dataset
(disease, a few symptom flags, age, gender, blood pressure, cholesterol, outcome, risk level). It supports
profile/risk exploration and is not used for diagnosis.</div>
<div class="card"><h4>🛠 Technologies</h4>Python · pandas · NumPy · scikit-learn (SVC, LabelEncoder, TfidfVectorizer,
cosine similarity) · joblib · Streamlit · Jupyter notebooks.</div>
""", unsafe_allow_html=True)
    disclaimer_box("Disease-prediction performance is based on the provided dataset; medication information is retrieved "
                   "from the provided dataset; medicine similarity represents textual similarity and does not imply "
                   "clinical equivalence or interchangeability.")


# ---------------------------------------------------------------------------
# Sidebar navigation + router
# ---------------------------------------------------------------------------
PAGES = {
    "🏠 Diagnosis": page_diagnosis,
    "💊 Medicine Similarity": page_similarity,
    "📊 Project Overview": page_overview,
    "ℹ️ About": page_about,
}

with st.sidebar:
    st.markdown("## 🩺 MediCare AI")
    st.caption("Personalized Healthcare & Medicine Recommendation System")
    st.divider()
    page = st.radio("Navigation", list(PAGES), label_visibility="collapsed")
    st.divider()
    st.markdown("**System information**")
    st.write(f"Disease classes: **{len(DISEASE[1].classes_) if DISEASE else '–'}**")
    st.write(f"Symptom features: **{len(DISEASE[2]) if DISEASE else '–'}**")
    st.write(f"Medicine records: **{len(MEDICINE[0]):,}**" if MEDICINE else "Medicine records: **–**")
    if DISEASE and MEDS is not None:
        missing = {normalise_disease(c) for c in DISEASE[1].classes_} - set(MEDS["Disease_key"])
        st.write("Medication coverage: **" + ("all classes ✓" if not missing else f"{len(missing)} missing ✗") + "**")
    st.divider()
    st.caption("Educational / informational prototype. Not a medical diagnosis or prescription system.")

PAGES[page]()

st.markdown('<div class="footer">Personalized Healthcare & Medicine Recommendation System<br>'
            'Machine Learning • TF-IDF • Cosine Similarity • Streamlit</div>', unsafe_allow_html=True)