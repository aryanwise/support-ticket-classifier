# Support Ticket Classifier

An end-to-end NLP classification and operational triage engine that categorizes inbound customer inquiries across organizational departments, computes sentiment-derived SLA urgency tiers, and manages an interactive dispatch queue.

The system utilizes a denoised weak-supervision pipeline: raw text is encoded into 384-dimensional dense semantic vectors using a Sentence-BERT bi-encoder (`all-MiniLM-L6-v2`) and classified via regularized Multinomial Logistic Regression.

---

## Live Application Demo

The production triage engine is deployed and accessible online:

- **Live App URL:** [support-ticket-classifier.streamlit.app](https://support-ticket-classifier-gisma.streamlit.app/)

### What You Can Test in the Live Demo

- **Real-time Department Routing:** Submit natural language customer complaints to classify them across 5 organizational units (Finance & Billing, Technical Support, Account Access & Security, Customer Retention & Cancellation, and Marketing & Sales).
- **Zero-Shot Semantic Generalization:** Test paraphrased queries with synonyms (e.g., "reimburse deduction" or "misplaced credentials") without relying on exact keyword matching.
- **Automated SLA Assignment:** View real-time urgency tiers derived from sentiment and critical failure tokens.
- **Interactive Triage Queue:** Inspect class probability distributions and watch newly routed tickets populate the operational dispatch log.

---

## Project Overview

Public customer support datasets often contain synthetic noise, template placeholders, and label-target misalignments. This project addresses those data quality challenges directly through programmatic weak supervision and dense representation learning, moving beyond brittle exact-token matching (TF-IDF) to achieve zero-shot semantic generalization.

### About Dataset `Customer Support Ticket Dataset`

The Customer Support Ticket Dataset is a dataset that includes customer support tickets for various tech products. It consists of customer inquiries related to hardware issues, software bugs, network problems, account access, data loss, and other support topics. The dataset provides information about the customer, the product purchased, the ticket type, the ticket channel, the ticket status, and other relevant details...[Read More](https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset/data)

### Core Capabilities

- **Dense Semantic Routing:** Maps free-form text into continuous vector space to classify complaints across 5 functional units:
- Finance & Billing
- Technical Support
- Account Access & Security
- Customer Retention & Cancellation
- Marketing & Sales

- **Zero-Shot Synonym Handling:** Accurately routes paraphrased phrasing (e.g., "reimburse deduction" -> Finance, "credentials" -> Security) without requiring hardcoded regex dictionaries.
- **SLA Urgency Scoring:** Evaluates emotional valence and high-impact failure tokens via VADER sentiment analysis to assign SLA response deadlines (`< 1h`, `4h`, `12h`, `24h`).
- **Real-Time Interactive Interface:** Clean, modern Streamlit operations dashboard featuring class probability distributions and an in-memory ticket queue.

---

## System Architecture

```text
[ Raw Inbound Ticket ]
          │
          ▼
[ Text Normalization ]
  (Contraction expansion, regex cleaning, lemmatization)
          │
          ▼
[ Dense Bi-Encoder: SBERT ]
  (all-MiniLM-L6-v2 -> 384-dimensional normalized embeddings)
          │
          ▼
[ Calibrated Classifier ]
  (Multinomial Logistic Regression with balanced class weights)
          │
          ├──► Target Department + Class Probability Distribution
          │
          ▼
[ Operational SLA Engine ]
  (VADER Sentiment Analysis + Critical Keyword Penalty)
          │
          ▼
[ Dispatched Ticket Queue ]

```

---

## Model Performance & Evaluation

The final classifier was evaluated on an 80/20 stratified hold-out test set derived from programmatically denoised ground truth:

- **Overall Accuracy:** 94.05%
- **Macro Average F1-Score:** 0.94
- **Weighted Average F1-Score:** 0.94

### Generalization on Unseen Complex Complaints

| Test Inbound Message                                                                         | Predicted Department              | Model Confidence |
| -------------------------------------------------------------------------------------------- | --------------------------------- | ---------------- |
| "I was charged twice on my credit card invoice. Please refund the extra money."              | Finance & Billing                 | 99.8%            |
| "The screen went completely black and the device refuses to turn on."                        | Technical Support                 | 76.6%            |
| "I forgot my password and my 2FA authentication code is not sending. Unlock my account."     | Account Access & Security         | 96.7%            |
| "I want to cancel our enterprise membership and terminate our subscription immediately."     | Customer Retention & Cancellation | 99.2%            |
| "Could you provide pricing details and tell me if this accessory is compatible with my Mac?" | Marketing & Sales                 | 97.9%            |

---

## Repository Structure

```text
support-ticket-classifier/
├── dataset/
│   ├── customer_support_tickets.csv           # Raw source dataset
│   └── cleaned_customer_support_tickets.csv   # Denoised, preprocessed dataset
├── model_output/
│   └── dense_ticket_classifier.joblib         # Serialized classifier & metadata
├── src/
│   ├── 01_exploratory_data_analysis.ipynb     # Data audit, leakage & noise identification
│   ├── 02_preprocessing_and_feature_eng.ipynb # Weak supervision, cleaning, SLA derivation
│   └── 03_model_training_and_eval.ipynb       # SBERT embedding generation, training & metrics
├── app.py                                     # Streamlit operations dashboard
├── requirements.txt                           # Project dependencies
├── .gitignore                                 # Git tracking exclusions
└── README.md                                  # Project documentation

```

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/support-ticket-classifier.git
cd support-ticket-classifier

```

### 2. Create and Activate a Virtual Environment

```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate

```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt

```

---

## Running the Application

To launch the local Streamlit dashboard:

```bash
streamlit run app.py

```

The application will launch locally at `http://localhost:8501`.

### Operational Flow in App:

1. Select a pre-configured scenario or enter a custom customer message.
2. Click **Dispatch Ticket** to generate embeddings and run inference.
3. Inspect the real-time target department, confidence rating, SLA urgency tier, and the horizontal probability distribution chart.
4. Review the newly assigned ticket in the **Dispatched Queue** table.

---

## Tech Stack

- **Language:** Python 3.10+
- **NLP & Representation Learning:** Sentence-Transformers (`all-MiniLM-L6-v2`), NLTK (VADER), Scikit-Learn
- **Data Processing:** Pandas, NumPy
- **Visualization:** Plotly Graph Objects, Seaborn, Matplotlib
- **Deployment & UI:** Streamlit, Joblib
