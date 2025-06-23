
# 🧬 Biomedical Abbreviation Detection using BioBERT

This project focuses on detecting biomedical abbreviations and their corresponding long forms in scientific text using token classification techniques. The system identifies abbreviations and expansions using transformer models and serves them through an interactive chatbot.

---

## Problem Statement

Biomedical texts often contain complex abbreviations that vary in context and meaning. The goal of this project is to build a model that accurately tags abbreviations (AC) and long forms (LF) using BIO tagging format:
- `B-AC` – Beginning of an abbreviation
- `B-LF` – Beginning of a long form
- `I-LF` – Inside of a long form
- `O` – Outside (not a named entity)

---

## Model Evaluation

We tested two models using the [PLOD-CW-25](https://huggingface.co/datasets/surrey-nlp/PLOD-CW-25) dataset:

| Model       | F1 Score | Recall  | Precision | Accuracy |
|-------------|----------|---------|-----------|----------|
| BioBERT     | 0.8464   | 0.8847  | 0.8113    | 93.11%   |
| DistilBERT  | 0.8130   | 0.8733  | 0.7604    | 92.94%   |

**BioBERT** was chosen for deployment due to its superior performance and domain-specific training.

---

## Additional Experiments

- **Data Augmentation:** Trained BioBERT with increasing subsets (5%, 10%, 15%) of the [PLODv2-filtered](https://huggingface.co/datasets/surrey-nlp/PLODv2-filtered) dataset for better generalization.
- **Loss Functions Tested:**
  - Cross-Entropy Loss (CE)
  - Weighted Cross-Entropy (WCE)
  - Focal Loss (FL)  
  Focal Loss showed best recall and stable training on imbalanced data.

---

## Deployment

### 🔗 Hosted Model
Deployed on Hugging Face Hub:  
👉 [`Lohit20/biobert-v1.2-base-cased-v1.2-ner`](https://huggingface.co/Lohit20/biobert-v1.2-base-cased-v1.2-ner)

### 🖥️ Run Locally (Flask App)

```bash
git clone https://github.com/Lohit20/biomedical-token-tagging.git
cd biomedical-token-tagging
pip install -r requirements.txt
python app.py
````

Visit: `http://127.0.0.1:5000` in your browser.

---

## Chatbot Interface

The chatbot lets users input biomedical text and returns real-time token classification results via:

* Flask + HTML (Jinja2 templates)
* CSS-styled chat interface
* JavaScript with AJAX for real-time responses

---

## Logging System

All predictions are logged into `token_predictions.json` in structured JSON format.

```json
{
  "timestamp": "2025-05-15T15:44:24.617326",
  "bio_ner_tags": [
    {"token": "hc", "label": "B-LF", "entity_type": "Long Form"},
    {"token": "c", "label": "B-AC", "entity_type": "Abbreviation"}
  ]
}
```

Each log includes:

* Timestamp
* Tokens with predicted BIO labels
* Human-readable entity type (e.g., Abbreviation, Long Form)

---

## Model Loading Example

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("Lohit20/biobert-v1.2-base-cased-v1.2-ner")
model = AutoModelForTokenClassification.from_pretrained("Lohit20/biobert-v1.2-base-cased-v1.2-ner")
```

---

## Limitations & Recommendations

* Flask is single-threaded; not ideal for concurrent users.
* Use `gunicorn` or `FastAPI` for production deployment.
* Add GPU support or use Hugging Face Inference API for scalability.
* Use Redis queues + Celery for async request handling in production.


```

Let me know if you'd like help auto-generating a `requirements.txt`, setting up a `.gitignore`, or structuring your repo for deployment (e.g., `/templates`, `/static`, `/logs`, `/models`).
```
