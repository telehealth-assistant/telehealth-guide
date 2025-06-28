# ────────────────────────────────────────────────────────────────────
# app.py  –  Step-6 Streamlit interface that uses finAL.xlsx
#           Place finAL.xlsx in a folder called  data/  next to this file.
#           Run with:  streamlit run app.py
# ────────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import whisper                        # speech-to-text
from tempfile import NamedTemporaryFile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── 1.  Load disease table from your Excel  ─────────────────────────
@st.cache_data
def load_disease_db(path: str = "data/finAL.xlsx"):
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()          # remove leading/trailing spaces
    symptom_corpus = df["Symptoms"].str.lower().tolist()
    disease_names  = df["Disease"].tolist()
    return df, symptom_corpus, disease_names

df_db, SYMPTOMS, DISEASES = load_disease_db()

# ── 2.  Prepare Whisper model  (cached so it loads only once) ───────
@st.cache_resource
def load_whisper():
    return whisper.load_model("base")            # ~142 MB

# ── 3.  Helper: transcribe uploaded audio to text  ──────────────────
# Step A: Save original file temporarily (could be m4a, 3gp, etc.)

from tempfile import NamedTemporaryFile
import ffmpeg

def transcribe(audio_file) -> str:
    with NamedTemporaryFile(delete=False, suffix=".amr") as tmp_input:
        tmp_input.write(audio_file.read())
        tmp_input.flush()

        # Convert to 16kHz mono WAV using ffmpeg
        with NamedTemporaryFile(delete=False, suffix=".wav") as tmp_output:
            try:
                (
                    ffmpeg
                    .input(tmp_input.name)
                    .output(tmp_output.name, format='wav', ac=1, ar='16000')
                    .overwrite_output()
                    .run(quiet=True)
                )
            except Exception as e:
                return f"❌ Audio conversion failed: {e}"

        # Transcribe using Whisper
        result = load_whisper().transcribe(tmp_output.name, fp16=False)
        return result["text"]




# ── 4.  Helper: find the closest disease using TF-IDF  ──────────────
def predict_disease(user_text: str):
    corpus = SYMPTOMS + [user_text.lower()]
    vecs   = TfidfVectorizer().fit_transform(corpus)
    sims   = cosine_similarity(vecs[-1], vecs[:-1])
    best_i = sims.argmax()
    best_score = sims[0, best_i]
    if best_score < 0.10:
        return "Condition unclear – consult doctor", "", best_score
    return DISEASES[best_i], SYMPTOMS[best_i], best_score

# ── 5.  Streamlit layout  ───────────────────────────────────────────
st.set_page_config(page_title="Rural Tele-Health Assistant", layout="centered")
st.title("🩺 Rural Tele-Health AI Assistant")

tab_voice, tab_image = st.tabs(["🎙️  Voice Input", "📷  Image (optional)"])

# ---- Voice tab ----
with tab_voice:
   audio = st.file_uploader("Upload your voice file", type=["wav", "mp3", "m4a", "3gp", "aac", "ogg", "amr"])


if audio:
        st.audio(audio)
        with st.spinner("Transcribing…"):
            spoken_text = transcribe(audio)
        st.success("You said:  " + spoken_text)

        disease, matched_symp, score = predict_disease(spoken_text)
        st.subheader("🔍 Possible Condition")
        st.write(f"**{disease}**  (match score ≈ {score:.2f})")
        if matched_symp:
            st.caption("Matched symptom pattern: " + matched_symp)

        # 👉 replace link with real doctor/PHC link if available
        st.markdown(
            "[💬 Click to consult a doctor](https://meet.google.com/test-link)",
            unsafe_allow_html=True
        )

# ---- Image tab ----
with tab_image:
    img = st.file_uploader("Upload a skin/eye photo (JPG/PNG)", type=["jpg", "png"])
    if img:
        st.image(img, caption="Uploaded image (CV model not integrated yet)")

# ────────────────────────────────────────────────────────────────────
