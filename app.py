import os
import re
import uuid
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Support Ticket Classifier",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# Minimalist, Clean Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    .block-container {
        padding-top: 3rem !important;
        padding-bottom: 3rem !important;
        max-width: 1350px !important;
    }
    
    /* Result card container */
    .result-card {
        background-color: #1e1e24;
        border: 1px solid #2e2e38;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
    }
    .result-label {
        font-size: 0.75rem;
        color: #8b8b9e;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.2rem;
    }
    .result-value {
    font-size: 1.0rem;
    font-weight: 600;
    color: #f1f1f4;
    line-height: 1.35;
    white-space: normal;          
    overflow: visible;
    text-overflow: clip;
    min-height: 2.7rem;           
    display: flex;
    align-items: center;
    }
    
    /* Custom button styling */
    div.stButton > button:first-child {
        background-color: #3b82f6 !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 500 !important;
        border-radius: 6px !important;
        padding: 0.5rem 1rem !important;
    }
    div.stButton > button:first-child:hover {
        background-color: #2563eb !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Lexicons & ML Core
# ---------------------------------------------------------
for res in ["sentiment/vader_lexicon.zip", "vader_lexicon"]:
    try:
        nltk.data.find(res)
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)

MODEL_PATH = os.path.join("model_output", "dense_ticket_classifier.joblib")

@st.cache_resource
def load_triage_core():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Artifact not found at '{MODEL_PATH}'. Please run Notebook 03 first.")
        st.stop()
        
    bundle = joblib.load(MODEL_PATH)
    classifier = bundle["classifier"]
    classes = bundle["classes"]
    model_name = bundle.get("model_name", "all-MiniLM-L6-v2")
    
    embedder = SentenceTransformer(model_name)
    sia = SentimentIntensityAnalyzer()
    return embedder, classifier, classes, sia

embedder, classifier, classes, sia = load_triage_core()

# ---------------------------------------------------------
# Preprocessing & Inference
# ---------------------------------------------------------
CONTRACTION_MAP = {
    "doesn't": "does not", "don't": "do not", "didn't": "did not",
    "can't": "cannot", "won't": "will not", "isn't": "is not",
    "it's": "it is", "i'm": "i am"
}

def clean_input(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    for c, e in CONTRACTION_MAP.items():
        text = text.replace(c, e)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

CRITICAL_TOKENS = {
    "not working", "crashes", "data loss", "locked", "error", 
    "broken", "unauthorized", "fail", "failed", "double charged", "refund"
}

def classify_ticket(raw_text: str):
    cleaned = clean_input(raw_text)
    vector = embedder.encode([cleaned], normalize_embeddings=True)
    
    pred_dept = classifier.predict(vector)[0]
    probabilities = classifier.predict_proba(vector)[0]
    confidence = float(np.max(probabilities))
    
    # SLA priority via VADER
    scores = sia.polarity_scores(raw_text)
    compound = scores["compound"]
    hits = sum(1 for token in CRITICAL_TOKENS if token in raw_text.lower())
    adjusted_compound = compound - (hits * 0.15)
    
    if adjusted_compound <= -0.10 or hits >= 2:
        priority = "Critical (< 1h)"
    elif adjusted_compound < 0.20 or hits >= 1:
        priority = "High (4h)"
    elif adjusted_compound <= 0.50:
        priority = "Medium (12h)"
    else:
        priority = "Low (24h)"
        
    return {
        "department": pred_dept,
        "confidence": confidence,
        "priority": priority,
        "probabilities": probabilities
    }

# ---------------------------------------------------------
# Session Queue
# ---------------------------------------------------------
if "ticket_queue" not in st.session_state:
    st.session_state.ticket_queue = [
        {
            "Ticket ID": "TCK-9401",
            "Timestamp": "11:42",
            "Department": "Finance & Billing",
            "Priority": "Critical (< 1h)",
            "Confidence": 0.998,
            "Complaint": "Card was charged twice on invoice #8042. Please refund immediately."
        },
        {
            "Ticket ID": "TCK-9402",
            "Timestamp": "11:35",
            "Department": "Technical Support",
            "Priority": "High (4h)",
            "Confidence": 0.924,
            "Complaint": "Dell monitor keeps flickering black whenever connecting via DisplayPort."
        }
    ]

# ---------------------------------------------------------
# Clean Header
# ---------------------------------------------------------
st.title("Support Ticket Classifier")
st.markdown(
    "<p style='color: #8b8b9e; margin-top: -12px; margin-bottom: 24px;'>"
    "NLP dispatch pipeline powered by Sentence-BERT (all-MiniLM-L6-v2) and Logistic Regression"
    "</p>", 
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Main 2-Column Work Area
# ---------------------------------------------------------
col_left, col_right = st.columns([0.9, 1.4], gap="large")

with col_left:
    st.markdown("##### Ingest Ticket")
    
    preset = st.selectbox(
        "Load a sample ticket:",
        [
            "Card was charged twice on invoice #8042. Please refund immediately.",
            "Display panel shuts off randomly and fan is making loud screeching noise.",
            "Lost phone and cannot pass 2FA authenticator challenge to sign in.",
            "Enterprise contract renewal: need to cancel membership before end of month.",
            "Does your platform support OAuth2 single sign-on and custom webhooks?",
            "Custom Input..."
        ],
        index=0
    )
    
    default_text = "" if preset == "Custom Input..." else preset
    complaint_body = st.text_area(
        "Ticket Message:",
        value=default_text,
        height=130,
        placeholder="Type customer message here..."
    )
    
    btn_col1, btn_col2 = st.columns([1.4, 1])
    with btn_col1:
        submit = st.button("Dispatch Ticket", use_container_width=True)
    with btn_col2:
        if st.button("Clear Queue", use_container_width=True):
            st.session_state.ticket_queue = []
            st.rerun()

# Run real-time classification
result = classify_ticket(complaint_body if complaint_body.strip() else "Inquiry placeholder")

if submit and complaint_body.strip():
    new_ticket = {
        "Ticket ID": f"TCK-{str(uuid.uuid4().int)[:4]}",
        "Timestamp": datetime.now().strftime("%H:%M"),
        "Department": result["department"],
        "Priority": result["priority"],
        "Confidence": result["confidence"],
        "Complaint": complaint_body.strip()
    }
    st.session_state.ticket_queue.insert(0, new_ticket)
    st.toast(f"Dispatched #{new_ticket['Ticket ID']} to {result['department']}", icon="✅")

with col_right:
    st.markdown("##### Classification Decision")
    
    # Clean 3-part metrics row
    m1, m2, m3 = st.columns([2.0, 1.0, 1.0])
    with m1:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Target Department</div>
            <div class="result-value" style="color: #60a5fa;">{result['department']}</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Confidence</div>
            <div class="result-value" style="color: #34d399;">{result['confidence']*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Priority Tier</div>
            <div class="result-value" style="color: #fb923c;">{result['priority']}</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Probability distribution bar chart
    prob_pairs = sorted(zip(classes, result["probabilities"]), key=lambda x: x[1])
    sorted_depts = [p[0] for p in prob_pairs]
    sorted_probs = [p[1] * 100 for p in prob_pairs]
    bar_colors = ["#3b82f6" if p == max(sorted_probs) else "#2e2e38" for p in sorted_probs]
    
    fig = go.Figure(go.Bar(
        x=sorted_probs,
        y=sorted_depts,
        orientation='h',
        marker=dict(color=bar_colors),
        text=[f"{p:.1f}%" for p in sorted_probs],
        textposition="outside",
        textfont=dict(color="#d4d4d8", size=11)
    ))
    
    fig.update_layout(
        height=190,
        margin=dict(l=0, r=40, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, showticklabels=False, range=[0, 115]),
        yaxis=dict(showgrid=False, tickfont=dict(color="#a1a1aa", size=11)),
        bargap=0.3
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

st.markdown("<hr style='border-color: #27272a; margin: 1.5rem 0;'>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Ticket Queue Table
# ---------------------------------------------------------
st.markdown("##### Dispatched Queue")

if st.session_state.ticket_queue:
    df_queue = pd.DataFrame(st.session_state.ticket_queue)
    
    st.dataframe(
        df_queue,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Ticket ID": st.column_config.TextColumn("Ticket ID", width="small"),
            "Timestamp": st.column_config.TextColumn("Time", width="small"),
            "Department": st.column_config.TextColumn("Assigned Department", width="medium"),
            "Priority": st.column_config.TextColumn("Priority Tier", width="medium"),
            "Confidence": st.column_config.ProgressColumn(
                "Confidence",
                format="%.1f%%",
                min_value=0.0,
                max_value=1.0,
                width="small"
            ),
            "Complaint": st.column_config.TextColumn("Customer Message", width="large")
        }
    )
else:
    st.info("Queue is currently empty.")