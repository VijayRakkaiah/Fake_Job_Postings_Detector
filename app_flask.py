# app_flask.py
import os
import re, html, traceback
from typing import Optional, List, Dict, Any
from flask import Flask, request, jsonify, render_template
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.special import expit
import joblib
import numpy as np

# Optional explainability libs (LIME/SHAP)
from lime.lime_text import LimeTextExplainer
import shap

# ---------- CONFIG ----------
MODEL_PATH = "best_model_xgb.joblib"
VECT_PATH  = "tfidf_vectorizer.joblib"
TEXT_COLS = ['title', 'company_profile', 'description', 'requirements', 'benefits']

# ---------- FLASK SETUP ----------
app = Flask(__name__)

# ---------- LOAD MODEL & VECTORIZER ----------
if not os.path.exists(MODEL_PATH) or not os.path.exists(VECT_PATH):
    raise FileNotFoundError(f"Please ensure {MODEL_PATH} and {VECT_PATH} exist in the working directory.")

best_model = joblib.load(MODEL_PATH)
tfidf = joblib.load(VECT_PATH)

# ---------- PREPROCESSING ----------
_stop_words = set(stopwords.words('english'))
_lemmatizer = WordNetLemmatizer()

def combine_text_from_obj(obj: Dict[str, Any]) -> str:
    parts = []
    for k in TEXT_COLS:
        v = obj.get(k, "")
        if v is None: continue
        s = str(v).strip()
        if s != "" and s.lower() != "nan":
            parts.append(s)
    return " \n ".join(parts)

def clean_text(text: Optional[str]) -> str:
    if text is None: text = ""
    if not isinstance(text, str): text = str(text)
    text = text.lower()
    text = html.unescape(text)
    text = re.sub(r'http\S+|www\S+|\S+@\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [_lemmatizer.lemmatize(tok) for tok in text.split() if tok not in _stop_words and len(tok) > 1]
    return " ".join(tokens)

# ---------- PREDICTION HELPERS ----------
def predict_proba_from_raw_texts(texts: List[str]) -> np.ndarray:
    cleaned = [clean_text(t) for t in texts]
    X = tfidf.transform(cleaned)
    if hasattr(best_model, "predict_proba"):
        return best_model.predict_proba(X)
    else:
        if hasattr(best_model, "decision_function"):
            df = best_model.decision_function(X)
            if df.ndim == 1:
                prob_pos = expit(df)
                return np.vstack([1-prob_pos, prob_pos]).T
            else:
                probs = expit(df)
                probs = probs / probs.sum(axis=1, keepdims=True)
                return probs
        else:
            raise RuntimeError("Model lacks predict_proba and decision_function")

def predict_single(raw_text: str, threshold: float = 0.5) -> Dict[str, Any]:
    probs = predict_proba_from_raw_texts([raw_text])
    if probs.shape[1] == 2:
        prob_fake = float(probs[0,1])
    else:
        try:
            idx = list(best_model.classes_).index(1)
            prob_fake = float(probs[0, idx])
        except Exception:
            prob_fake = float(probs[0].max())
    label = 1 if prob_fake >= threshold else 0
    return {"label": int(label), "prob_fake": prob_fake, "raw_text": raw_text}

# LIME explainer (instantiate once)
_lime_explainer = LimeTextExplainer(class_names=['real','fake'])
def lime_explain(raw_text: str, num_features: int = 8):
    def predict_fn(texts: List[str]):
        return predict_proba_from_raw_texts(texts)
    exp = _lime_explainer.explain_instance(raw_text, predict_fn, num_features=num_features)
    return exp.as_list()

# SHAP explanation (simple single-instance top features)
def shap_explain_single(raw_text: str, topn: int = 10) -> Dict[str, Any]:
    try:
        explainer = shap.Explainer(best_model)
        X = tfidf.transform([clean_text(raw_text)])
        shap_vals = explainer(X)
        vals = shap_vals.values
        if vals.ndim == 3:
            vals = vals[:,1,:]
        arr = vals.ravel()
        feat_names = tfidf.get_feature_names_out()
        idx = np.argsort(-np.abs(arr))[:topn]
        return {"top_features": [{"feature": feat_names[i], "shap_value": float(arr[i])} for i in idx]}
    except Exception as e:
        return {"error": f"SHAP failed: {str(e)}"}

# ---------- ROUTES ----------
@app.route("/", methods=["GET"])
def home():
    # Renders templates/index.html
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    try:
        data = request.get_json(force=True)
        if data is None:
            return jsonify({"error":"No JSON body found"}), 400

        raw_text = data.get("raw_text") or combine_text_from_obj(data)
        if raw_text.strip() == "":
            return jsonify({"error":"Empty text. Provide raw_text or fill text fields."}), 400

        threshold = float(data.get("threshold", 0.5))
        result = predict_single(raw_text, threshold=threshold)

        explain = data.get("explain", None)
        if explain == "lime":
            try:
                result["lime_explanation"] = lime_explain(raw_text, num_features=int(data.get("num_features", 8)))
            except Exception as e:
                result["lime_error"] = str(e)
        elif explain == "shap":
            result["shap_explanation"] = shap_explain_single(raw_text, topn=int(data.get("num_features", 10)))

        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

# ---------- RUN ----------
if __name__ == "__main__":
    # DEV server — for production use gunicorn/uvicorn
    app.run(host="0.0.0.0", port=8000, debug=True)
