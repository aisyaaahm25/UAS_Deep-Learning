import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ======= Custom CSS =======
st.markdown(
    """
    <style>
    /* Background baby pink */
    .stApp {
        background-color: #ffeaf1;
    }

    /* Judul warna pink */
    .title-text {
        color: #ff99b9;
        font-size: 48px;
        font-weight: bold;
        text-align: center;
    }

    /* Subtitle / teks biasa */
    .subtitle-text {
        color: #1e3370;
        font-size: 20px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ======= Load model =======
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

# ======= Title & subtitle =======
st.markdown('<div class="title-text">🎁 Giftbox Personal Message Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Buat ucapan personal sesuai acara dengan AI</div>', unsafe_allow_html=True)

# ======= User inputs =======
acara = st.selectbox(
    "Pilih Acara",
    ["Ulang Tahun", "Wisuda", "Hari Ibu", "Hari Ayah", "Natal", "Valentine", "Perpisahan"]
)

gaya = st.selectbox("Gaya Pesan", ["Friendly", "Formal", "Lucu", "Romantis"])
panjang = st.selectbox("Panjang Pesan", ["Pendek", "Sedang", "Panjang"])
instruksi = st.text_area("Instruksi tambahan", "Buat ucapan hangat dan natural, maksimal 3 kalimat.", height=80)

# ======= Build prompt =======
def build_prompt(acara, gaya, panjang, instruksi=None):
    instruksi_text = instruksi if instruksi else "Buat pesan personal sesuai konteks acara."
    # prompt lebih fokus ke acara
    return (
        "### INSTRUCTION:\n"
        f"{instruksi_text}\n\n"
        "### INPUT:\n"
        f"Acara: {acara} | Gaya: {gaya} | Panjang: {panjang}\n\n"
        f"Buat ucapan yang spesifik untuk acara {acara}.\n"
        "### OUTPUT:\n"
    )

# ======= Generate Pesan =======
if st.button("✨ Generate Pesan"):
    prompt = build_prompt(acara, gaya, panjang, instruksi)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # daftar kata yang ingin dihindari
    bad_words = ["bapak", "ibu", "ayah", "anniversary", "menikah", "rumah sakit", "sekolah", "lulus"]
    bad_words_ids = [tokenizer.encode(w, add_special_tokens=False) for w in bad_words]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=80,           # output lebih pendek
            do_sample=True,
            temperature=0.6,
            top_p=0.85,
            repetition_penalty=2.0,      # kurangi pengulangan
            no_repeat_ngram_size=3,
            bad_words_ids=bad_words_ids,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    final_output = text.split("### OUTPUT:")[-1].strip()

    st.subheader("🎉 Hasil Pesan:")
    st.text_area("Pesan AI", final_output, height=150)
