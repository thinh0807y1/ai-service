from __future__ import annotations

import os
import re
import unicodedata
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


def plain(text: str) -> str:
    text = (text or "").strip().lower().replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^a-z0-9#.,/\-\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def money_value(number: str, unit: str | None = None) -> float:
    raw = (number or "").strip().lower()
    # 50.000 / 50,000 -> 50000; 49,5k -> 49500
    if unit:
        raw = raw.replace(",", ".")
        try:
            val = float(raw)
        except ValueError:
            return 0.0
        u = plain(unit)
        if u in {"k", "nghin", "ngan"}:
            return val * 1000
        if u in {"trieu", "m"}:
            return val * 1_000_000
        return val

    compact = raw.replace(" ", "")
    if re.fullmatch(r"\d{1,3}([.,]\d{3})+", compact):
        return float(re.sub(r"[.,]", "", compact))
    compact = compact.replace(",", ".")
    try:
        return float(compact)
    except ValueError:
        return 0.0


def first_money(text: str) -> float | None:
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*(k|nghin|ngan|trieu|m)\b", text)
    if m:
        return money_value(m.group(1), m.group(2))
    m = re.search(r"\b(\d{4,}(?:[.,]\d{3})*)\b", text)
    if m:
        return money_value(m.group(1))
    return None


STOPWORDS = {
    "tim", "kiem", "truyen", "sach", "manga", "cho", "toi", "minh", "em", "anh", "chi",
    "cac", "nhung", "mot", "vai", "giup", "hay", "muon", "can", "co", "khong", "nao", "voi",
    "ve", "de", "duoc", "cua", "shop", "mangaworld", "xem", "gia", "khoang", "tu", "den", "duoi",
    "tren", "toi", "da", "dang", "con", "hang", "ban", "nhat", "moi", "phu", "hop", "goi", "y",
    "danh", "gia", "sao", "top", "gia", "re", "dat", "nhieu", "it", "hon", "tro", "len", "va",
    "hoac", "nhieu", "nhieu", "thu", "the", "loai", "tac", "gia", "noi", "dung", "noi", "bat",
    "dau", "vao", "ra", "nay", "kia", "a", "nhe", "nha", "oi", "duoi", "khong", "qua",
    "la", "ai", "so", "sanh", "bao", "may", "don", "phi", "giao", "nhanh", "dung"
}


def extract_terms(text: str) -> list[str]:
    clean = plain(text)
    clean = re.sub(r"\b\d+(?:[.,]\d+)?\s*(?:k|nghin|ngan|trieu|m|sao)?\b", " ", clean)
    clean = re.sub(r"#\d+", " ", clean)
    tokens = re.findall(r"[a-z0-9]+", clean)
    terms: list[str] = []
    for token in tokens:
        if len(token) < 2 or token in STOPWORDS:
            continue
        if token not in terms:
            terms.append(token)
        if len(terms) >= 8:
            break
    return terms


def parse_entities(message: str) -> dict[str, Any]:
    clean = plain(message)
    entities: dict[str, Any] = {
        "terms": extract_terms(message),
        "limit": 5,
        "sort": "relevance",
        "in_stock": False,
    }

    m = re.search(r"\btop\s*(\d{1,2})\b", clean) or re.search(r"\b(\d{1,2})\s*(?:truyen|sach|manga)\b", clean)
    if m:
        entities["limit"] = max(1, min(int(m.group(1)), 10))

    # Khoảng giá "từ 30k đến 60k"
    range_m = re.search(
        r"\btu\s*(\d+(?:[.,]\d+)?)\s*(k|nghin|ngan|trieu|m)?\s*(?:den|toi|-)\s*(\d+(?:[.,]\d+)?)\s*(k|nghin|ngan|trieu|m)?",
        clean,
    )
    if range_m:
        unit1 = range_m.group(2) or range_m.group(4)
        unit2 = range_m.group(4) or range_m.group(2)
        entities["min_price"] = money_value(range_m.group(1), unit1)
        entities["max_price"] = money_value(range_m.group(3), unit2)
    else:
        max_m = re.search(r"\b(?:duoi|khong qua|toi da|nho hon|re hon)\s*(\d+(?:[.,]\d+)?)\s*(k|nghin|ngan|trieu|m)?", clean)
        min_m = re.search(r"\b(?:tren|it nhat|toi thieu|lon hon)\s*(\d+(?:[.,]\d+)?)\s*(k|nghin|ngan|trieu|m)?", clean)
        if max_m:
            entities["max_price"] = money_value(max_m.group(1), max_m.group(2))
        if min_m:
            entities["min_price"] = money_value(min_m.group(1), min_m.group(2))

    rating_m = re.search(r"(?:danh gia|rating|tu)?\s*(\d(?:[.,]\d)?)\s*sao(?:\s*(?:tro len|trở lên))?", clean)
    if rating_m:
        try:
            entities["min_rating"] = max(0.0, min(5.0, float(rating_m.group(1).replace(",", "."))))
        except ValueError:
            pass

    entities["in_stock"] = any(k in clean for k in ["con hang", "co hang", "san hang", "mua duoc"])
    if any(k in clean for k in ["sap het hang", "sap het", "ton kho thap"]):
        entities["max_stock"] = 5

    if any(k in clean for k in ["re nhat", "gia thap nhat", "gia tang dan"]):
        entities["sort"] = "price_asc"
    elif any(k in clean for k in ["dat nhat", "gia cao nhat", "gia giam dan"]):
        entities["sort"] = "price_desc"
    elif any(k in clean for k in ["danh gia cao", "nhieu sao", "rating cao"]):
        entities["sort"] = "rating_desc"
    elif any(k in clean for k in ["ban chay", "pho bien", "nhieu nguoi mua", "hot nhat"]):
        entities["sort"] = "sold_desc"
    elif any(k in clean for k in ["moi nhat", "moi ra", "sach moi", "truyen moi"]):
        entities["sort"] = "newest"

    # Đơn hàng
    order_m = re.search(r"#\s*(\d+)", clean) or re.search(r"\bdon(?:\s*hang)?\s+(?:ma\s+)?(\d+)\b", clean)
    if order_m:
        entities["order_id"] = int(order_m.group(1))
    entities["latest_order"] = any(k in clean for k in ["don gan nhat", "don moi nhat", "don vua dat"])
    entities["count_request"] = any(k in clean for k in ["bao nhieu don", "co may don", "may don"])

    status_map = {
        "cho thanh toan": "Chờ thanh toán",
        "cho xac nhan": "Chờ xác nhận",
        "da xac nhan": "Đã xác nhận",
        "dang giao": "Đang giao",
        "hoan thanh": "Hoàn thành",
        "da huy": "Đã hủy",
        "cho hoan tien": "Chờ hoàn tiền",
        "da hoan tien": "Đã hoàn tiền",
    }
    for key, value in status_map.items():
        if key in clean:
            entities["order_status"] = value
            break

    # Giao hàng
    if any(k in clean for k in ["giao nhanh", "ship nhanh", "hoa toc"]):
        entities["shipping_method"] = "fast"
    elif any(k in clean for k in ["giao tieu chuan", "ship thuong", "giao thuong"]):
        entities["shipping_method"] = "standard"

    if any(k in clean for k in ["phi ship", "phi van chuyen", "giao hang", "giao nhanh", "giao tieu chuan", "ship"]):
        addr_m = re.search(r"(?:ve|den|toi)\s+([a-z0-9\s.\-/]+?)(?:\s+(?:cho|voi|don|gia|het|bao nhieu)|$)", clean)
        if addr_m:
            address = addr_m.group(1).strip()
            if 2 <= len(address) <= 120:
                entities["address"] = address
        # tổng tiền hàng nếu người dùng nói "đơn 200k"
        subtotal_m = re.search(r"(?:don|gio|tien hang|tong)\s*(?:hang)?\s*(\d+(?:[.,]\d+)?)\s*(k|nghin|ngan|trieu|m)?", clean)
        if subtotal_m:
            entities["subtotal"] = money_value(subtotal_m.group(1), subtotal_m.group(2))

    # Voucher theo giá trị đơn
    if "voucher" in clean or "ma giam" in clean or "khuyen mai" in clean:
        subtotal = first_money(clean)
        if subtotal:
            entities["subtotal"] = subtotal

    # So sánh hai tên truyện: "so sánh A và B"
    compare_m = re.search(r"so sanh\s+(.+?)\s+(?:voi|va)\s+(.+)$", clean)
    if compare_m:
        entities["compare"] = [compare_m.group(1).strip(), compare_m.group(2).strip()]

    return entities


def rule_intent(message: str, entities: dict[str, Any]) -> str | None:
    clean = plain(message)
    if not clean:
        return "welcome"
    if "compare" in entities:
        return "compare_books"
    if any(k in clean for k in ["xin chao", "chao bot", "hello", "hi bot", "chao shop"]):
        return "greeting"
    if any(k in clean for k in ["hoan tien", "refund"]):
        return "refund_info"
    if any(k in clean for k in ["huy don", "huy don hang"]):
        return "cancel_order"
    if any(k in clean for k in ["voucher", "ma giam", "khuyen mai"]):
        return "voucher_info"
    if any(k in clean for k in ["phi ship", "phi van chuyen", "ship", "giao hang bao lau", "giao nhanh", "giao tieu chuan"]):
        return "shipping_info"
    if any(k in clean for k in ["cod", "thanh toan khi nhan"]):
        return "cod_info"
    if any(k in clean for k in ["thanh toan online", "vietqr", "qr", "mangapay"]):
        return "online_payment"
    if entities.get("count_request") or entities.get("order_status") or any(k in clean for k in ["don hang", "don #", "don gan nhat", "don moi nhat", "theo doi don", "trang thai don"]):
        return "order_status"
    if any(k in clean for k in ["ban chay", "pho bien", "nhieu nguoi mua", "hot nhat"]):
        return "best_seller"
    if any(k in clean for k in ["moi nhat", "moi ra", "sach moi", "truyen moi"]):
        return "new_books"
    if any(k in clean for k in ["duoi ", "tren ", "re nhat", "dat nhat", "gia thap", "gia cao"]) or "min_price" in entities or "max_price" in entities:
        return "price_filter"
    if any(k in clean for k in ["tim ", "truyen", "sach", "manga", "tac gia", "the loai", "con hang", "danh gia cao"]):
        return "search_book"
    return None


def predict_payload(message: str) -> dict[str, Any]:
    probabilities = chatbot_model.predict_proba([message])[0]
    classes = chatbot_model.classes_
    ranked = np.argsort(probabilities)[::-1]
    best_index = int(ranked[0])
    second_index = int(ranked[1]) if len(ranked) > 1 else best_index
    raw_intent = str(classes[best_index])
    confidence = float(probabilities[best_index])
    margin = confidence - float(probabilities[second_index])

    accepted = raw_intent == "out_of_scope" or (confidence >= 0.11 and margin >= 0.012)
    ml_intent = raw_intent if accepted else "out_of_scope"
    entities = parse_entities(message)
    rule = rule_intent(message, entities)

    # Luật chỉ override với tín hiệu nghiệp vụ rõ ràng. Với câu khác giữ classifier đã train.
    final_intent = rule or ml_intent
    return {
        "ok": True,
        "message": message,
        "intent": final_intent,
        "raw_intent": raw_intent,
        "ml_intent": ml_intent,
        "rule_intent": rule,
        "confidence": round(confidence, 4),
        "margin": round(margin, 4),
        "accepted": bool(accepted),
        "entities": entities,
        "model": "Hybrid NLU: TF-IDF + Logistic Regression + entity parser",
        "trained": True,
    }


@app.get("/")
def index():
    return jsonify({
        "ok": True,
        "service": "Manga World AI Service",
        "message": "Dịch vụ phân tích ngôn ngữ MangaBot đang hoạt động.",
        "endpoints": ["/health", "/predict-intent", "/analyze", "/recommend"],
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
        "nlu_version": "2026-08-10-hybrid-v2",
    })


@app.post("/predict-intent")
def predict_intent():
    body = request.get_json(silent=True) or {}
    message = str(body.get("message", "")).strip()
    if not message:
        return jsonify({"ok": False, "message": "Nội dung câu hỏi đang trống."}), 400
    return jsonify(predict_payload(message))


@app.post("/analyze")
def analyze():
    body = request.get_json(silent=True) or {}
    message = str(body.get("message", "")).strip()
    if not message:
        return jsonify({"ok": False, "message": "Nội dung câu hỏi đang trống."}), 400
    return jsonify(predict_payload(message))


@app.post("/recommend")
def recommend():
    """Giữ endpoint để tương thích bản cũ. Website mới không còn trang AI gợi ý riêng."""
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
        item = {"idsach": int(book_ids[index]), "ai_score": round(float(similarity_scores[index]), 4)}
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
