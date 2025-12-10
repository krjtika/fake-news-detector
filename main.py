from flask import Flask, request, render_template_string, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import joblib
import os

MODEL_PATH = "ts_model.joblib"

data = [
    ("Breaking: Scientists discovered cure for rare disease, buy this miracle pill now!", "fake"),
    ("Government announces new scholarship scheme for students in 2025 academic year", "real"),
    ("Study shows coffee linked to improved focus — peer-reviewed research", "real"),
    ("Celebrity endorses crypto coin guaranteed 1000% returns", "fake"),
    ("Local hospital to open free eye camp next week for senior citizens", "real"),
    ("Secret method to get rich in 7 days — click to learn", "fake"),
    ("City council passes new safety regulations for food vendors", "real"),
    ("Shocking: eat this fruit and lose 10kg in 3 days, doctors hate it", "fake"),
    ("University publishes dataset of climate observations for regional study", "real"),
    ("You won't believe what this politician said — viral clip manipulates facts", "fake")
]

def build_dataframe(data_list):
    df = pd.DataFrame(data_list, columns=["text", "label"])
    return df

def train_and_save_model(df, path=MODEL_PATH):
    X = df["text"]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    pipeline = make_pipeline(
        TfidfVectorizer(ngram_range=(1,2), max_features=4000),
        LogisticRegression(solver="liblinear", C=1.0)
    )
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print("Model trained — test accuracy:", round(acc, 3))
    print(classification_report(y_test, preds))
    joblib.dump(pipeline, path)
    print("Saved model to", path)
    return pipeline

def load_model(path=MODEL_PATH):
    if os.path.exists(path):
        model = joblib.load(path)
        print("Loaded model from", path)
        return model
    return None

df = build_dataframe(data)
model = load_model()
if model is None:
    model = train_and_save_model(df)

app = Flask(__name__)

INDEX_HTML = """
<!doctype html>
<title>Trust & Safety — Fake News Demo</title>
<h2>Trust & Safety — Fake News Detector (Demo)</h2>
<form method="post" action="/predict">
  <textarea name="text" rows="6" cols="80" placeholder="Paste an article or headline here..."></textarea><br>
  <button type="submit">Check</button>
</form>
{% if result %}
  <h3>Prediction: {{ result.label }}</h3>
  <p>Confidence: {{ result.confidence }}%</p>
  <h4>Details</h4>
  <pre>{{ result.details }}</pre>
{% endif %}
<hr>
<p style="font-size:0.9em;color:gray">This is a demo aligned with the AI for Trust & Safety PPT. For better accuracy use a larger labeled dataset and more preprocessing.</p>
"""

def simple_predict(model, text):
    probs = model.predict_proba([text])[0]
    classes = model.classes_
    idx = probs.argmax()
    label = classes[idx]
    confidence = float(probs[idx]) * 100
    detail_lines = []
    for c, p in sorted(zip(classes, probs), key=lambda x: -x[1]):
        detail_lines.append(f"{c}: {round(p*100,2)}%")
    details = "\n".join(detail_lines)
    return {"label": label, "confidence": round(confidence, 2), "details": details}

@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML)

@app.route("/predict", methods=["POST"])
def predict_form():
    text = request.form.get("text", "")
    if not text.strip():
        return render_template_string(INDEX_HTML, result={"label":"(no text)","confidence":0,"details":""})
    res = simple_predict(model, text)
    return render_template_string(INDEX_HTML, result=res)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(force=True, silent=True) or {}
    text = payload.get("text", "")
    if not text:
        return jsonify({"error":"No text provided"}), 400
    res = simple_predict(model, text)
    return jsonify(res)

if __name__ == "__main__":
    print("Starting Trust & Safety demo app on http://127.0.0.1:5000")
    app.run(debug=True)
