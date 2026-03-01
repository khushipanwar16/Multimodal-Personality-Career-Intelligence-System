import streamlit as st
import torch
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import plotly.express as px
import time   # ⭐ ABSOLUTE GOD MODE

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="AI Career Intelligence",
    page_icon="🧠",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM STYLE
# ---------------------------------------------------

st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.stTextArea textarea {
    border-radius: 12px;
}
.result-card {
    padding: 20px;
    border-radius: 12px;
    background: linear-gradient(135deg,#1f2937,#111827);
    color: white;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD PERSONALITY MODEL
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

# ---------------------------------------------------
# LABEL MAPPING
# ---------------------------------------------------

reverse_mapping = {
    0: "ENFP",
    1: "ENTP",
    2: "INFJ",
    3: "INFP",
    4: "INTJ",
    5: "INTP",
    6: "ISFP",
    7: "ISTP"
}

# ---------------------------------------------------
# LOAD EMBEDDING MODEL
# ---------------------------------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_embedding_model()

# ---------------------------------------------------
# LOAD CAREER DATA
# ---------------------------------------------------

career_df = pd.read_csv("data/career_dataset/career_data.csv")

@st.cache_resource
def get_career_embeddings():
    return embed_model.encode(career_df["description"].tolist())

career_embeddings = get_career_embeddings()

# ---------------------------------------------------
# PERSONALITY PREDICTION
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

# ---------------------------------------------------
# CAREER RECOMMENDATION
# ---------------------------------------------------

def recommend_careers(user_text, top_k=5):

    user_embedding = embed_model.encode([user_text])

    similarities = cosine_similarity(user_embedding, career_embeddings)

    top_indices = similarities[0].argsort()[-top_k:][::-1]

    recommendations = career_df.iloc[top_indices]["career"].tolist()

    return recommendations, top_indices

# ---------------------------------------------------
# AI EXPLANATION GENERATOR
# ---------------------------------------------------

def generate_explanations(user_text, recommendations, personality):

    personality_traits = {
        "ENFP": "creative thinking, adaptability, and strong communication",
        "ENTP": "innovation, debating ideas, and problem-solving",
        "INFJ": "deep insight, empathy, and long-term vision",
        "INFP": "creativity, values-driven thinking, and imagination",
        "INTJ": "strategic planning, independence, and analytical thinking",
        "INTP": "logic, curiosity, and abstract reasoning",
        "ISFP": "practical creativity and personal expression",
        "ISTP": "hands-on problem solving and technical exploration"
    }

    trait_text = personality_traits.get(
        personality,
        "strong analytical and creative abilities"
    )

    explanations = []

    for career in recommendations:
        reason = (
            f"Based on your writing style and detected personality ({personality}), "
            f"you may fit **{career}** because your text reflects {trait_text}. "
            f"This aligns well with the strengths required in this career."
        )
        explanations.append(reason)

    return explanations

# ---------------------------------------------------
# ⭐ CHATGPT STYLE TYPEWRITER EFFECT
# ---------------------------------------------------

def typewriter_effect(text, speed=0.03):

    placeholder = st.empty()
    typed = ""

    for word in text.split():
        typed += word + " "
        placeholder.markdown(typed)
        time.sleep(speed)

# ---------------------------------------------------
# INTERACTIVE 3D VISUALIZATION
# ---------------------------------------------------

def visualize(top_indices):

    pca = PCA(n_components=3)
    career_3d = pca.fit_transform(career_embeddings)

    df_plot = pd.DataFrame({
        "x": career_3d[:,0],
        "y": career_3d[:,1],
        "z": career_3d[:,2],
        "career": career_df["career"]
    })

    df_plot["highlight"] = "Other"
    df_plot.loc[top_indices, "highlight"] = "Recommended"

    fig = px.scatter_3d(
        df_plot,
        x="x",
        y="y",
        z="z",
        color="highlight",
        hover_name="career",
        title="Interactive 3D Career Space"
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------

st.title("🧠 AI Career Intelligence")
st.caption("Multimodal Personality + Career Recommendation System")
st.divider()

user_text = st.text_area("Enter text about yourself:", height=200)

# ---------------------------------------------------
# ANALYZE BUTTON
# ---------------------------------------------------

if st.button("Analyze"):

    if user_text.strip() == "":
        st.warning("Please enter some text first.")
    else:

        with st.spinner("🧠 AI is analyzing your personality..."):

            personality, probs = predict_personality(user_text)

            col1, col2 = st.columns([1,1])

            with col1:
                st.markdown(f"""
                <div class="result-card">
                <h3>Predicted Personality</h3>
                <h1>{personality}</h1>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.subheader("Model Confidence")
                for idx, p in enumerate(probs):
                    st.progress(float(p))
                    st.caption(f"{reverse_mapping[idx]} : {float(p):.2f}")

            recommendations, top_indices = recommend_careers(user_text)

            st.subheader("Recommended Careers")

            for career in recommendations:
                st.markdown(f"""
                <div class="result-card">⭐ {career}</div>
                """, unsafe_allow_html=True)

            # ---------------------------------------------------
            # ABSOLUTE GOD MODE AI REASONING
            # ---------------------------------------------------

            st.subheader("🧠 Why these careers were chosen")

            explanations = generate_explanations(
                user_text,
                recommendations,
                personality
            )

            for exp in explanations:
                st.markdown("🤖 **AI Reasoning:**")
                typewriter_effect(exp)
                st.divider()

            st.subheader("🌍 Interactive 3D Career Space")

            visualize(top_indices)