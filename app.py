import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="HeartStory AI",
    page_icon="💝",
    layout="centered"
)

# ================== CUSTOM CSS ==================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Great+Vibes&family=Cinzel:wght@400;700&family=Baloo+2:wght@500;700&display=swap');

    /* Background */
    .stApp {
        background-color: #ffeaf1;
    }

    /* Logo */
    .logo-container {
        text-align: center;
        margin-top: 10px;
        margin-bottom: 5px;
    }

    .logo-h {
        font-family: 'Great Vibes', cursive;
        font-size: 90px;
        color: #1e3370;
        margin-right: -10px;
    }

    .logo-main {
        font-family: 'Cinzel', serif;
        font-size: 56px;
        font-weight: 700;
        letter-spacing: 3px;
        color: #1e3370;
    }

    .tagline {
        font-family: 'Baloo 2', cursive;
        font-size: 34px;
        font-weight: 700;
        color: #1e3370;
        text-align: center;
        margin-top: -10px;
    }

    .subtitle-text {
        font-family: 'Baloo 2', cursive;
        color: #1e3370;
        font-size: 18px;
        text-align: center;
        margin-bottom: 25px;
    }

    /* Card */
    .card {
        background: white;
        border-radius: 18px;
        padding: 25px;
        box-shadow: 0 10px 25px rgba(30, 51, 112, 0.15);
        margin-bottom: 25px;
    }

    /* Divider */
    .soft-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #ff99b9, transparent);
        margin: 25px 0;
    }

    /* Button */
    .stButton>button {
        background: linear-gradient(135deg, #ff99b9, #ff6f9f);
        color: white;
        border-radius: 30px;
        padding: 0.6em 2.5em;
        font-size: 18px;
        font-weight: 700;
        border: none;
        transition: 0.3s ease;
    }

    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 20px rgba(255, 111, 159, 0.5);
    }

    /* Output */
    .output-box textarea {
        border-radius: 14px !important;
        font-size: 16px !important;
        background-color: #fff7fb;
        border: 2px solid #ff99b9;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ================== LOAD MODEL ==================
MODEL_PATH = "aisyaaahm25/giftbox-model"
device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ================== HEADER ==================
st.markdown(
    """
    <div class="logo-container">
        <span class="logo-h">H</span>
        <span class="logo-main">EARTSTORY AI</span>
    </div>
    <div class="tagline">CRAFTING STORIES IN EVERY GIFT</div>
    <div class="subtitle-text">Buat ucapan personal penuh makna dengan bantuan AI</div>
    """,
    unsafe_allow_html=True
)

# ================== INPUT CARD ==================
st.markdown('<div class="card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    acara = st.selectbox(
        "🎉 Pilih Acara",
        ["Ulang Tahun", "Wisuda", "Hari Ibu", "Hari Ayah", "Natal", "Valentine", "Perpisahan"]
    )
    gaya = st.selectbox("🎨 Gaya Pesan", ["Friendly", "Formal", "Lucu", "Romantis"])

with col2:
    panjang = st.selectbox("📏 Panjang Pesan", ["Pendek", "Sedang", "Panjang"])
    instruksi = st.text_area(
        "✍️ Instruksi Tambahan",
        "Buat ucapan hangat dan natural, maksimal 3 kalimat.",
        height=100
    )

st.markdown('</div>', unsafe_allow_html=True)

# ================== BUTTON ==================
center = st.columns([1, 2, 1])
with center[1]:
    generate = st.button("✨ Generate Pesan")

# ================== PROMPT BUILDER ==================
def build_prompt(acara, gaya, panjang, instruksi=None):
    instruksi_text = instruksi if instruksi else "Buat pesan personal sesuai konteks acara."
    return (
        "### INSTRUCTION:\n"
        f"{instruksi_text}\n\n"
        "### INPUT:\n"
        f"Acara: {acara} | Gaya: {gaya} | Panjang: {panjang}\n\n"
        f"Buat ucapan yang spesifik untuk acara {acara}.\n"
        "### OUTPUT:\n"
    )

# ================== GENERATE ==================
if generate:
    prompt = build_prompt(acara, gaya, panjang, instruksi)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    bad_words = ["bapak", "ibu", "ayah", "anniversary", "menikah", "rumah sakit", "sekolah", "lulus"]
    bad_words_ids = [tokenizer.encode(w, add_special_tokens=False) for w in bad_words]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.6,
            top_p=0.85,
            repetition_penalty=2.0,
            no_repeat_ngram_size=3,
            bad_words_ids=bad_words_ids,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    final_output = text.split("### OUTPUT:")[-1].strip()

    st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="card output-box">', unsafe_allow_html=True)
    st.subheader("💌 HeartStory Message")
    st.text_area("Pesan AI", final_output, height=160)
    st.markdown('</div>', unsafe_allow_html=True)
