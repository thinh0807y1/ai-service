from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from flask import Flask, jsonify, request
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
CHATBOT_MODEL_FILE = MODEL_DIR / "chatbot_intent.joblib"
RECOMMENDER_MODEL_FILE = MODEL_DIR / "recommender.joblib"


def load_models() -> tuple[Any, dict[str, Any]]:
    if not CHATBOT_MODEL_FILE.exists():
        raise FileNotFoundError(f"Không tìm thấy model chatbot: {CHATBOT_MODEL_FILE}")
    if not RECOMMENDER_MODEL_FILE.exists():
        raise FileNotFoundError(f"Không tìm thấy model gợi ý: {RECOMMENDER_MODEL_FILE}")

    # Chỉ load các file model do chính chủ dự án tạo/train trên Kaggle.
    chatbot = joblib.load(CHATBOT_MODEL_FILE)
    recommender = joblib.load(RECOMMENDER_MODEL_FILE)

    required_keys = {"vectorizer", "book_matrix", "book_ids"}
    if not isinstance(recommender, dict) or not required_keys.issubset(recommender):
        missing = required_keys - set(recommender.keys() if isinstance(recommender, dict) else [])
        raise ValueError("Model recommender thiếu dữ liệu: " + ", ".join(sorted(missing)))

    return chatbot, recommender


chatbot_model, recommender_model = load_models()
recommender_vectorizer = recommender_model["vectorizer"]
book_matrix = recommender_model["book_matrix"]
book_ids = [int(book_id) for book_id in recommender_model["book_ids"]]
book_titles = recommender_model.get("book_titles", [])
trained_version = str(recommender_model.get("scikit_learn_version", "1.6.1"))
trained_at = str(recommender_model.get("trained_at", ""))

app = Flask(__name__)


def require_service_key() -> bool:
    """Cho phép cấu hình khóa khi deploy host; local mặc định không cần khóa."""
    configured_key = os.environ.get("AI_SERVICE_KEY", "").strip()
    if not configured_key:
        return True
    supplied_key = request.headers.get("X-AI-Service-Key", "").strip()
    return supplied_key == configured_key


@app.before_request
def protect_api():
    if request.path in {"/", "/health"}:
        return None
    if not require_service_key():
        return jsonify({"ok": False, "message": "AI service key không hợp lệ."}), 401
    return None


@app.get("/")
def index():
    return jsonify({
        "ok": True,
        "service": "Manga World AI Service",
        "message": "Dịch vụ mô hình đã train đang hoạt động.",
        "endpoints": ["/health", "/predict-intent", "/recommend"],
    })


@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "service": "Manga World AI Service",
        "chatbot_model": CHATBOT_MODEL_FILE.name,
        "recommender_model": RECOMMENDER_MODEL_FILE.name,
        "number_of_books": len(book_ids),
        "number_of_intents": len(getattr(chatbot_model, "classes_", [])),
        "trained_scikit_learn_version": trained_version,
        "trained_at": trained_at,
    })


@app.post("/predict-intent")
def predict_intent():
    body = request.get_json(silent=True) or {}
    message = str(body.get("message", "")).strip()

    if not message:
        return jsonify({"ok": False, "message": "Nội dung câu hỏi đang trống."}), 400

    probabilities = chatbot_model.predict_proba([message])[0]
    classes = chatbot_model.classes_
    ranked = np.argsort(probabilities)[::-1]

    best_index = int(ranked[0])
    second_index = int(ranked[1]) if len(ranked) > 1 else best_index
    predicted_intent = str(classes[best_index])
    confidence = float(probabilities[best_index])
    second_confidence = float(probabilities[second_index])
    margin = confidence - second_confidence

    # Tập train nhỏ có xác suất tuyệt đối không cao. Kết hợp ngưỡng và khoảng cách
    # top-1/top-2 để từ chối câu quá lạ, nhưng vẫn nhận đúng các câu đã học.
    accepted = predicted_intent == "out_of_scope" or (
        confidence >= 0.11 and margin >= 0.012
    )
    if not accepted:
        predicted_intent = "out_of_scope"

    return jsonify({
        "ok": True,
        "message": message,
        "intent": predicted_intent,
        "raw_intent": str(classes[best_index]),
        "confidence": round(confidence, 4),
        "margin": round(margin, 4),
        "accepted": bool(accepted),
        "model": "TF-IDF ký tự + Logistic Regression",
        "trained": True,
    })


@app.post("/recommend")
def recommend():
    body = request.get_json(silent=True) or {}
    query = str(body.get("query", "")).strip()

    try:
        limit = int(body.get("limit", 8))
    except (TypeError, ValueError):
        limit = 8
    limit = max(1, min(limit, 20))

    effective_query = query or "truyện manga nổi bật hành động phiêu lưu"
    query_vector = recommender_vectorizer.transform([effective_query])
    similarity_scores = cosine_similarity(query_vector, book_matrix).flatten()
    ranked_indexes = np.argsort(similarity_scores)[::-1][:limit]

    suggestions = []
    for index in ranked_indexes:
        index = int(index)
        item = {
            "idsach": int(book_ids[index]),
            "ai_score": round(float(similarity_scores[index]), 4),
        }
        if index < len(book_titles):
            item["trained_title"] = str(book_titles[index])
        suggestions.append(item)

    return jsonify({
        "ok": True,
        "query": query,
        "effective_query": effective_query,
        "model": "Content-Based TF-IDF + Cosine Similarity",
        "trained": True,
        "suggestions": suggestions,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
