from deepface import DeepFace
import numpy as np
import streamlit as st
import torch
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import plotly.express as px
import time

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="AI Career Intelligence",
    page_icon="🧠",
    layout="wide"
)

# ---------------------------------------------------
# ELITE + ULTRA UI STYLE (FULL VERSION)
# ---------------------------------------------------

st.markdown("""
<style>

/* ---------- ANIMATED BACKGROUND ---------- */

.stApp {
    background: linear-gradient(-45deg,#0b0f19,#111827,#0f172a,#0b0f19);
    background-size: 400% 400%;
    animation: gradientMove 15s ease infinite;
}

@keyframes gradientMove {
    0% {background-position:0% 50%;}
    50% {background-position:100% 50%;}
    100% {background-position:0% 50%;}
}

/* ---------- FADE IN ---------- */

.fade-in {
    animation: fadeInUp 0.7s ease forwards;
    opacity:0;
}

@keyframes fadeInUp {
    from {opacity:0; transform:translateY(15px);}
    to {opacity:1; transform:translateY(0);}
}

/* ---------- PAGE ---------- */

.block-container {
    padding-top:2rem;
    padding-bottom:2rem;
}

/* ---------- TEXT AREA ---------- */

.stTextArea textarea {
    border-radius:16px;
    border:1px solid #2a2f3a;
    background-color:rgba(17,24,39,0.8);
    color:white;
}

/* ---------- GLASS CARD ---------- */

.result-card {
    padding:20px;
    border-radius:18px;
    background:rgba(255,255,255,0.06);
    border:1px solid rgba(255,255,255,0.12);
    backdrop-filter:blur(12px);
    color:white;
    margin-bottom:16px;
    transition:all 0.35s ease;
    box-shadow:0 8px 30px rgba(0,0,0,0.3);
}

.result-card:hover {
    transform:translateY(-5px);
    border-color:#6366f1;
    box-shadow:0 12px 40px rgba(99,102,241,0.25);
}

/* ---------- BUTTON ---------- */

.stButton button {
    border-radius:12px;
    background:linear-gradient(90deg,#4f46e5,#6366f1);
    color:white;
    font-weight:600;
    border:none;
    transition:0.3s;
}

.stButton button:hover {
    transform:scale(1.03);
}

/* ---------- EXPANDER ---------- */

.streamlit-expanderHeader {
    font-size:1rem;
    font-weight:600;
}

/* ---------- HEADINGS ---------- */

h1,h2,h3 {
    letter-spacing:0.5px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------

@st.cache_resource
def load_personality_model():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "notebooks/results/checkpoint-1534"
    )
    model.eval()
    return tokenizer, model

tokenizer, model = load_personality_model()

reverse_mapping = {
    0:"ENFP",1:"ENTP",2:"INFJ",3:"INFP",
    4:"INTJ",5:"INTP",6:"ISFP",7:"ISTP"
}

mbti_names = {
    "ENFP": "The Campaigner (Creative & Energetic)",
    "ENTP": "The Innovator (Idea Explorer)",
    "INFJ": "The Advocate (Insightful & Purpose-Driven)",
    "INFP": "The Idealist (Thoughtful & Creative)",
    "INTJ": "The Strategist (Analytical Planner)",
    "INTP": "The Thinker (Logical Explorer)",
    "ISFP": "The Artist (Practical & Expressive)",
    "ISTP": "The Builder (Hands-on Problem Solver)"
}

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_embedding_model()

career_df = pd.read_csv("data/career_dataset/career_data.csv")

@st.cache_resource
def get_career_embeddings():
    return embed_model.encode(career_df["description"].tolist())

career_embeddings = get_career_embeddings()

# ---------------------------------------------------
# CORE LOGIC
# ---------------------------------------------------

def predict_personality(text):

    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()

    return reverse_mapping[pred], probs[0]

def recommend_careers(user_text, top_k=5):
    user_embedding = embed_model.encode([user_text])
    similarities = cosine_similarity(user_embedding, career_embeddings)
    top_indices = similarities[0].argsort()[-top_k:][::-1]
    recommendations = career_df.iloc[top_indices]["career"].tolist()
    return recommendations, top_indices

def analyze_image(image_file):

    try:
        result = DeepFace.analyze(
            img_path=image_file,
            actions=["emotion"],
            enforce_detection=False
        )

        emotion = result[0]["dominant_emotion"]

        emotion_map = {
            "happy": [0.25,0.2,0.1,0.15,0.1,0.05,0.1,0.05],
            "neutral":[0.1,0.1,0.15,0.1,0.2,0.2,0.1,0.05],
            "sad":[0.05,0.05,0.25,0.2,0.15,0.15,0.1,0.05],
            "angry":[0.05,0.2,0.05,0.05,0.2,0.2,0.1,0.15]
        }

        probs = emotion_map.get(
            emotion,
            [0.125]*8
        )

        return torch.tensor(probs)

    except:
        return torch.tensor([0.125]*8)
def is_valid_input(text):

    words = text.split()

    # too short
    if len(words) < 20:
        return False

    # low alphabet ratio (random chars)
    letters = sum(c.isalpha() for c in text)
    ratio = letters / max(len(text), 1)

    if ratio < 0.7:
        return False

    return True

def personality_signal_score(text):

    text_lower = text.lower()

    words = text_lower.split()

    # basic quality
    word_count = len(words)
    unique_ratio = len(set(words)) / max(word_count, 1)

    # first-person signals
    personality_words = [
        "i", "my", "me", "myself",
        "feel", "think", "enjoy",
        "prefer", "love", "hate"
    ]

    signal_count = sum(
        1 for w in words if w in personality_words
    )

    score = 0

    # length score
    if word_count > 30:
        score += 1

    # uniqueness score (prevents coding coding coding)
    if unique_ratio > 0.4:
        score += 1

    # personality signal score
    if signal_count >= 3:
        score += 1

    return score

def generate_explanations(user_text, recommendations, personality):

    # personality context (short, not repeated heavily)
    traits = {
        "ENFP":"creative and people-oriented",
        "ENTP":"innovative and idea-driven",
        "INFJ":"insightful and empathetic",
        "INFP":"imaginative and value-driven",
        "INTJ":"strategic and analytical",
        "INTP":"logical and curious",
        "ISFP":"creative and practical",
        "ISTP":"hands-on and technical"
    }

    explanations = []

    for i, career in enumerate(recommendations):

        desc = career_df[
            career_df["career"] == career
        ]["description"].values[0]

        short_desc = " ".join(desc.split()[:18])

        # ⭐ DIFFERENT STYLE PER ITEM
        if i == 0:
            reason = (
                f"**Top match:** {career} fits well because this role focuses on "
                f"{short_desc}... Your {traits[personality]} profile strongly aligns with this kind of work."
            )

        elif i == 1:
            reason = (
                f"This recommendation appears because {career} often requires skills like "
                f"{short_desc}... These tasks match your natural personality strengths."
            )

        elif i == 2:
            reason = (
                f"{career} is suggested since it involves {short_desc}... "
                f"Your personality indicates good compatibility with this environment."
            )

        else:
            reason = (
                f"This career involves {short_desc}... "
                f"Based on your overall profile, it shares overlap with your strengths and working style."
            )

        explanations.append(reason)

    return explanations

def typewriter_effect(text, speed=0.02):
    placeholder = st.empty()
    typed=""
    for word in text.split():
        typed += word+" "
        placeholder.markdown(typed)
        time.sleep(speed)

def visualize(top_indices):

    pca = PCA(n_components=3)
    career_3d = pca.fit_transform(career_embeddings)

    df_plot = pd.DataFrame({
        "x":career_3d[:,0],
        "y":career_3d[:,1],
        "z":career_3d[:,2],
        "career":career_df["career"]
    })

    df_plot["highlight"] = "Other"
    df_plot.loc[top_indices,"highlight"] = "Recommended"

    fig = px.scatter_3d(
        df_plot,
        x="x", y="y", z="z",
        color="highlight",
        hover_name="career",
        title="Interactive 3D Career Space"
    )

    st.plotly_chart(fig,use_container_width=True)

# ---------------------------------------------------
# UI
# ---------------------------------------------------

st.markdown("""
<div class="result-card fade-in">
<h1>🧠 AI Career Intelligence</h1>
<p style="opacity:0.8;">
Multimodal Personality + Career Recommendation System
</p>
</div>
""", unsafe_allow_html=True)

user_text = st.text_area("Enter text about yourself:", height=200)
uploaded_image = st.file_uploader(
    "Upload a profile image (optional)",
    type=["jpg","jpeg","png"]
)

if uploaded_image:
    st.image(uploaded_image, width=250)

if st.button("Analyze"):

    # -----------------------------
    # BASIC VALIDATION
    # -----------------------------
    if user_text.strip() == "":
        st.warning("Please enter some text first.")

    elif not is_valid_input(user_text):

        st.error(
            "⚠️ Please enter a meaningful paragraph (at least 20+ words) describing yourself."
        )

    else:

        # -----------------------------
        # PERSONALITY SIGNAL CHECK
        # -----------------------------
        score = personality_signal_score(user_text)

        if score <= 1:
            st.error(
                "⚠️ Input does not contain enough personality information."
            )
            st.stop()

        # -----------------------------
        # MAIN ANALYSIS
        # -----------------------------
        with st.spinner("🧠 AI is analyzing your personality..."):

            # TEXT PREDICTION
            personality, probs = predict_personality(user_text)

            # IMAGE FUSION
            image_probs = torch.zeros(8)

            if uploaded_image:
                image_probs = analyze_image(uploaded_image)

           # FUSION
            fusion_probs = (
                0.7 * probs +
                0.3 * image_probs
            )

            max_conf = torch.max(fusion_probs).item()

            # -----------------------------
            # LOW CONFIDENCE CHECK FIRST
            # -----------------------------
            if max_conf < 0.25:
                st.warning(
                    "⚠️ Input is unclear for personality prediction. Please write more about yourself."
                )
                st.stop()

            # -----------------------------
            # CONFIDENCE LEVEL UI
            # -----------------------------
            if max_conf < 0.40:
                conf_level = "MEDIUM"
                color = "#f59e0b"   # orange
            else:
                conf_level = "HIGH"
                color = "#22c55e"   # green

            st.markdown(
                f"""
                <div style="
                    padding:12px;
                    border-radius:12px;
                    background:{color};
                    color:white;
                    font-weight:600;
                    margin-bottom:15px;
                ">
                🧠 AI Confidence Level: {conf_level}
                </div>
                """,
                unsafe_allow_html=True
            )

            # FINAL PERSONALITY
            final_idx = torch.argmax(fusion_probs).item()
            personality = reverse_mapping[final_idx]

            st.info("🧠 Final personality derived from Text + Image fusion")

            # -----------------------------
            # UI SECTION
            # -----------------------------
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"""
                <div class="result-card fade-in">
                <h3>Predicted Personality</h3>
               <h1>{mbti_names[personality]}</h1>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.subheader("Model Confidence")

                # ⭐ show only TOP 3 personalities
                top_indices = torch.argsort(
                    fusion_probs,
                    descending=True
                )[:3]

                for idx in top_indices:

                    p = fusion_probs[idx]
                    label = reverse_mapping[int(idx)]
                    percentage = float(p) * 100

                    st.markdown(
                        f"**{mbti_names[label]}** — {percentage:.1f}%"
                    )

                    # animated bar
                    bar = st.progress(0)

                    target = int(percentage)

                    for i in range(target):
                        bar.progress(i + 1)
                        time.sleep(0.005)
                for i in range(target):
                    bar.progress(i + 1)
                    time.sleep(0.01)    

            # -----------------------------
            # CAREER RECOMMENDATIONS
            # -----------------------------
            recommendations, top_indices = recommend_careers(user_text)

            st.subheader("🚀 Recommended Careers")

            cols = st.columns(2)
            for i, career in enumerate(recommendations):
                with cols[i % 2]:
                    st.markdown(f"""
                    <div class="result-card fade-in">
                    <h3>⭐ {career}</h3>
                    </div>
                    """, unsafe_allow_html=True)

            # -----------------------------
            # AI REASONING
            # -----------------------------
            st.subheader("🧠 AI Reasoning")

            explanations = generate_explanations(
                user_text,
                recommendations,
                personality
            )

            for i, exp in enumerate(explanations):
                with st.expander(f"🤖 Why recommendation #{i+1}?"):
                    typewriter_effect(exp)

            st.markdown(
                '<div class="fade-in"><h2>🌍 Explore Career Landscape (Interactive)</h2></div>',
                unsafe_allow_html=True
            )

            visualize(top_indices)