# ────────────────────────────────────────────────────────────────────
# app.py – Tele‑Health assistant: voice symptom + skin‑image analysis
# Place finAL.xlsx in  data/   and skin_disease_model.h5 next to this file.
# Run with:  streamlit run app.py
# ────────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import whisper
from tempfile import NamedTemporaryFile
import ffmpeg                                  # audio conversion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# image‑model imports
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# ────────────────────────────────────────────────────────────────────
# 1. Load disease table from Excel
# ────────────────────────────────────────────────────────────────────
@st.cache_data
def load_disease_db(path: str = "data/finAL.xlsx"):
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()
    symptom_corpus = df["Symptoms"].str.lower().tolist()
    disease_names  = df["Disease"].tolist()
    return df, symptom_corpus, disease_names

df_db, SYMPTOMS, DISEASES = load_disease_db()

# ────────────────────────────────────────────────────────────────────
# 2. Load Whisper model (once)
# ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_whisper():
    return whisper.load_model("base")   # ~142 MB

# ────────────────────────────────────────────────────────────────────
# 3. Audio → text helper  (accepts wav/mp3/m4a/3gp/aac/ogg/amr)
# ────────────────────────────────────────────────────────────────────
def transcribe(audio_file) -> str:
    with NamedTemporaryFile(delete=False, suffix=".input") as tmp_in:
        tmp_in.write(audio_file.read())
        tmp_in.flush()

    with NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out:
        (
            ffmpeg
            .input(tmp_in.name)
            .output(tmp_out.name, format="wav", ac=1, ar="16000")
            .overwrite_output()
            .run(quiet=True)
        )
        wav_path = tmp_out.name

    result = load_whisper().transcribe(wav_path, fp16=False)
    return result["text"]

# ────────────────────────────────────────────────────────────────────
# 4. Text → likely disease using TF‑IDF
# ────────────────────────────────────────────────────────────────────
def predict_disease(user_text: str):
    corpus = SYMPTOMS + [user_text.lower()]
    vecs   = TfidfVectorizer().fit_transform(corpus)
    sims   = cosine_similarity(vecs[-1], vecs[:-1])
    best_i = sims.argmax()
    score  = sims[0, best_i]
    if score < 0.10:
        return "Condition unclear – consult doctor", "", score
    return DISEASES[best_i], SYMPTOMS[best_i], score

# ────────────────────────────────────────────────────────────────────
# 5.  Image‑model helpers  (auto‑resize to model input)
# ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_skin_model():
    return load_model("skin_disease_model.h5")   # or .keras

def predict_skin(img_file):
    model = load_skin_model()
    _, H, W, _ = model.input_shape            # e.g., (None, 224, 224, 3)
    img = image.load_img(img_file, target_size=(H, W))
    x   = image.img_to_array(img) / 255.0
    x   = np.expand_dims(x, 0)                # shape (1, H, W, 3)
    probs   = model.predict(x, verbose=0)[0]
    classes = ["eczema", "psoriasis", "healthy"]
    idx     = int(np.argmax(probs))
    return classes[idx], float(probs[idx])

# ────────────────────────────────────────────────────────────────────
# 6.  Streamlit UI
# ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Rural Tele‑Health Assistant", layout="centered")
st.title("🩺 Rural Tele‑Health AI Assistant")

tab_voice, tab_image = st.tabs(["🎙️ Voice Input", "📷 Skin Image"])

# ---- Voice tab ----
with tab_voice:
    audio = st.file_uploader(
        "Upload or record your voice (wav/mp3/m4a/3gp/aac/ogg/amr)",
        type=["wav", "mp3", "m4a", "3gp", "aac", "ogg", "amr"]
    )
    if audio:
        st.audio(audio)
        with st.spinner("Transcribing…"):
            spoken = transcribe(audio)
        st.success("You said:  " + spoken)

        disease, matched, score = predict_disease(spoken)
        st.subheader("🔍 Possible Condition")
        st.write(f"**{disease}**  (match score ≈ {score:.2f})")
        if matched:
            st.caption("Matched symptom pattern: " + matched)

        st.markdown(
            "[💬 Click to consult a doctor](https://meet.google.com/test-link)",
            unsafe_allow_html=True
        )

# ---- Image tab ----
with tab_image:
    img = st.file_uploader("Upload a skin image (JPG / PNG)", type=["jpg", "png"])
    if img:
        st.image(img, caption="Uploaded image", use_column_width=True)
        with st.spinner("Analyzing image…"):
            label, conf = predict_skin(img)
        st.success(f"🧪 Prediction: **{label}**  (confidence ≈ {conf:.2f})")

# ────────────────────────────────────────────────────────────────────

