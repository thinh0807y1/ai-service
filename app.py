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
    semantic_terms: list[str] = []
    aliases = {
        "danh nhau": ["hanh", "dong"], "chien dau": ["hanh", "dong"], "combat": ["hanh", "dong"],
        "pha an": ["trinh", "tham"], "tham tu": ["trinh", "tham"], "dieu tra": ["trinh", "tham"],
        "lang man": ["tinh", "cam"], "romance": ["tinh", "cam"],
        "phieu luu": ["phieu", "luu"], "hai huoc": ["hai", "huoc"],
    }
    for phrase, mapped in aliases.items():
        if phrase in clean:
            for token in mapped:
                if token not in semantic_terms:
                    semantic_terms.append(token)
    clean = re.sub(r"\b\d+(?:[.,]\d+)?\s*(?:k|nghin|ngan|trieu|m|sao)?\b", " ", clean)
    clean = re.sub(r"#\d+", " ", clean)
    tokens = re.findall(r"[a-z0-9]+", clean)
    terms: list[str] = list(semantic_terms)
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
    if any(k in clean for k in ["cam on", "cám on", "thanks", "thank you"]):
        return "thanks"
    if any(k in clean for k in ["tam biet", "bye", "hen gap lai"]):
        return "goodbye"
    if any(k in clean for k in ["ban la ai", "ban lam duoc gi", "bot lam duoc gi", "co the giup gi", "hoi duoc gi", "mangabot lam gi"]):
        return "capabilities"
    if any(k in clean for k in ["xin chao", "chao bot", "hello", "hi bot", "chao shop"]):
        return "greeting"
    if any(k in clean for k in ["so sanh cod", "cod va online", "online va cod", "nen chon cod", "nen thanh toan online"]):
        return "payment_compare"
    if any(k in clean for k in ["cach dat hang", "dat hang nhu the nao", "mua hang nhu the nao", "lam sao de mua", "huong dan dat hang"]):
        return "how_to_order"
    if any(k in clean for k in ["doi mat khau", "cap nhat tai khoan", "cap nhat dia chi", "sua thong tin tai khoan", "thong tin ca nhan"]):
        return "account_help"
    if any(k in clean for k in ["cach danh gia", "danh gia san pham nhu the nao", "binh luan san pham", "viet danh gia"]):
        return "review_info"
    if any(k in clean for k in ["manga la gi", "truyen manga la gi"]):
        return "manga_info"
    if any(k in clean for k in ["anime la gi", "phim anime la gi"]):
        return "anime_info"
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
    if any(k in clean for k in ["goi y cho toi", "tu van truyen", "de xuat cho toi", "phu hop voi toi", "nen doc truyen", "nen mua truyen", "toi thich", "minh thich", "khong biet doc gi"]):
        return "recommend_personal"
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


YES_WORDS = {
    "co", "co nhe", "duoc", "duoc do", "ok", "oke", "okie", "dong y", "uh", "u", "yes",
    "lam di", "goi y di", "tiep di", "chon di"
}
NO_WORDS = {"khong", "khong can", "thoi", "de sau", "no", "khong nhe"}


def _as_int_list(value: Any, limit: int = 20) -> list[int]:
    if not isinstance(value, list):
        return []
    out: list[int] = []
    for item in value:
        try:
            iv = int(item)
        except (TypeError, ValueError):
            continue
        if iv > 0 and iv not in out:
            out.append(iv)
        if len(out) >= limit:
            break
    return out


def _safe_context(body: dict[str, Any]) -> dict[str, Any]:
    raw = body.get("context")
    if not isinstance(raw, dict):
        raw = {}
    history = body.get("history")
    clean_history: list[dict[str, str]] = []
    if isinstance(history, list):
        for item in history[-12:]:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).strip().lower()
            content = str(item.get("content", item.get("text", ""))).strip()
            if role not in {"user", "assistant"} or not content:
                continue
            clean_history.append({"role": role, "content": content[:1200]})
    return {
        "last_intent": str(raw.get("last_intent", ""))[:80],
        "last_entities": raw.get("last_entities") if isinstance(raw.get("last_entities"), dict) else {},
        "last_result_ids": _as_int_list(raw.get("last_result_ids")),
        "last_result_titles": [str(x)[:180] for x in raw.get("last_result_titles", [])[:20]] if isinstance(raw.get("last_result_titles"), list) else [],
        "last_focus_book_id": int(raw.get("last_focus_book_id", 0) or 0),
        "pending_action": str(raw.get("pending_action", ""))[:100],
        "pending_limit": max(1, min(int(raw.get("pending_limit", 3) or 3), 10)),
        "last_follow_up_question": str(raw.get("last_follow_up_question", ""))[:500],
        "history": clean_history,
    }


def _ordinal_number(clean: str) -> int | None:
    m = re.search(r"(?:thu|so)\s*(\d{1,2})\b", clean)
    if m:
        return int(m.group(1))
    if re.search(r"\b(?:dau tien|cuon dau|truyen dau|sach dau)\b", clean):
        return 1
    words = {
        "nhat": 1, "hai": 2, "ba": 3, "bon": 4, "tu": 4, "nam": 5,
        "sau": 6, "bay": 7, "tam": 8, "chin": 9, "muoi": 10,
    }
    for word, number in words.items():
        if re.search(rf"\bthu\s+{re.escape(word)}\b", clean):
            return number
    return None


def _is_yes(clean: str) -> bool:
    return clean in YES_WORDS or any(clean.startswith(x + " ") for x in ["co", "duoc", "ok", "dong y", "yes"])


def _is_no(clean: str) -> bool:
    return clean in NO_WORDS or clean.startswith("khong can")


def _context_reference(clean: str) -> bool:
    return any(k in clean for k in [
        "tu do", "trong so do", "trong danh sach do", "cac truyen tren", "nhung truyen tren",
        "may truyen do", "cac cuon do", "nhung cuon do", "danh sach tren", "trong so tren",
        "truyen vua neu", "sach vua neu", "ket qua tren"
    ])


def contextualize(message: str, intent: str, entities: dict[str, Any], context: dict[str, Any]) -> tuple[str, dict[str, Any], bool, str]:
    """Giải tham chiếu hội thoại mà không cần LLM bên ngoài.

    Trả về (intent, entities, context_used, reason). Các id trong restrict_ids chỉ là
    id kết quả do website gửi từ lượt trước; AI service không tự truy cập MySQL.
    """
    clean = plain(message)
    e = dict(entities)
    last_ids = context.get("last_result_ids", [])
    pending_action = context.get("pending_action", "")

    # Người dùng trả lời Có/OK cho câu hỏi gợi ý trước đó.
    if _is_yes(clean) and pending_action and last_ids:
        if pending_action == "top_rated_from_last_results":
            e.update({
                "restrict_ids": last_ids,
                "terms": [],
                "sort": "rating_desc",
                "limit": context.get("pending_limit", 3),
            })
            return "search_book", e, True, "accepted_previous_follow_up"
        if pending_action == "cheapest_from_last_results":
            e.update({"restrict_ids": last_ids, "terms": [], "sort": "price_asc", "limit": 3})
            return "search_book", e, True, "accepted_previous_follow_up"
    if _is_yes(clean) and pending_action == "compare_payment_methods":
        return "payment_compare", e, True, "accepted_previous_follow_up"
    if _is_yes(clean) and pending_action == "explain_refund":
        return "refund_info", e, True, "accepted_previous_follow_up"
    if _is_yes(clean) and pending_action == "check_latest_order":
        e["latest_order"] = True
        return "order_status", e, True, "accepted_previous_follow_up"
    if _is_yes(clean) and pending_action in {"start_recommendation", "find_budget_books", "collect_recommendation_preferences"}:
        return "recommend_personal", e, True, "accepted_previous_follow_up"

    if _is_no(clean) and pending_action:
        return "conversation_decline", e, True, "declined_previous_follow_up"

    if pending_action == "collect_recommendation_preferences" and clean:
        previous = context.get("last_entities", {})
        if isinstance(previous, dict):
            for key in ["min_price", "max_price", "min_rating", "max_stock", "in_stock", "sort"]:
                if key not in e and key in previous:
                    e[key] = previous[key]
            old_terms = previous.get("terms", []) if isinstance(previous.get("terms"), list) else []
            new_terms = e.get("terms", []) if isinstance(e.get("terms"), list) else []
            e["terms"] = list(dict.fromkeys([*old_terms, *new_terms]))[:8]
        return "search_book", e, True, "continued_recommendation_preferences"

    # Câu ngắn như "dưới 50k thôi", "còn hàng", "cái nào hay nhất?" được hiểu là lọc tiếp danh sách trước.
    if last_ids and context.get("last_intent") in {"search_book", "price_filter", "best_seller", "new_books", "recommend_personal"}:
        has_new_filter = any(k in e for k in ["min_price", "max_price", "min_rating", "max_stock"]) or bool(e.get("in_stock"))
        has_sort_request = any(k in clean for k in ["re nhat", "dat nhat", "danh gia cao", "tot nhat", "hay nhat", "ban chay", "moi nhat", "nhieu sao"])
        content_terms = [x for x in e.get("terms", []) if x not in {"thoi", "cai", "hay", "tot"}]
        if (has_new_filter and not content_terms) or (has_sort_request and any(k in clean for k in ["cai nao", "truyen nao", "cuon nao", "sach nao", "nhat"])):
            e["restrict_ids"] = last_ids
            e["terms"] = []
            if any(k in clean for k in ["tot nhat", "hay nhat", "danh gia cao", "nhieu sao"]): e["sort"] = "rating_desc"
            elif "re nhat" in clean: e["sort"] = "price_asc"
            elif "dat nhat" in clean: e["sort"] = "price_desc"
            elif "ban chay" in clean: e["sort"] = "sold_desc"
            elif "moi nhat" in clean: e["sort"] = "newest"
            return "search_book", e, True, "implicit_continuation_of_previous_results"

    # Các câu nối tiếp ngắn theo chủ đề vừa trả lời, không cần tra CSDL.
    last_intent = context.get("last_intent", "")
    if last_intent in {"cod_info", "online_payment", "payment_compare"} and any(k in clean for k in ["cai nao tot hon", "nen chon cai nao", "loai nao tot hon", "khac nhau the nao", "so sanh"]):
        return "payment_compare", e, True, "continued_payment_topic"
    if last_intent in {"refund_info", "cancel_order"} and any(k in clean for k in ["vay can lam gi", "lam the nao", "tiep theo lam gi", "roi sao"]):
        return last_intent, e, True, "continued_policy_topic"

    # "cuốn thứ 2", "truyện thứ 3"... dùng danh sách bot vừa trả.
    ordinal = _ordinal_number(clean)
    if ordinal is not None and last_ids and 1 <= ordinal <= len(last_ids):
        e["focus_book_id"] = int(last_ids[ordinal - 1])
        e["terms"] = []
        return "book_detail", e, True, "ordinal_reference"

    # "cuốn đó / nó" sau khi bot đã tập trung vào một sách cụ thể.
    if context.get("last_focus_book_id") and any(k in clean for k in ["cuon do", "truyen do", "sach do", "no ", " no", "cai do"]):
        e["focus_book_id"] = int(context["last_focus_book_id"])
        e["terms"] = []
        return "book_detail", e, True, "pronoun_reference"

    # "so sánh 2 cuốn đầu" trong danh sách vừa trả.
    if last_ids and any(k in clean for k in ["so sanh 2 cuon dau", "so sanh hai cuon dau", "so sanh 2 truyen dau", "so sanh hai truyen dau"]):
        e["compare_ids"] = last_ids[:2]
        e["terms"] = []
        return "compare_context", e, True, "compare_previous_results"

    # "gợi ý 3 truyện tốt nhất từ đó", "rẻ nhất trong số đó"...
    if last_ids and _context_reference(clean):
        e["restrict_ids"] = last_ids
        e["terms"] = []
        if any(k in clean for k in ["tot nhat", "hay nhat", "danh gia tot", "danh gia cao", "nhieu sao"]):
            e["sort"] = "rating_desc"
        elif any(k in clean for k in ["re nhat", "gia thap nhat"]):
            e["sort"] = "price_asc"
        elif any(k in clean for k in ["ban chay", "pho bien", "nhieu nguoi mua"]):
            e["sort"] = "sold_desc"
        elif any(k in clean for k in ["moi nhat", "moi ra"]):
            e["sort"] = "newest"
        if intent in {"unknown", "recommend_personal", "out_of_scope"}:
            intent = "search_book"
        return intent, e, True, "reference_to_previous_results"

    # "3 truyện tốt nhất" không có tham chiếu -> hiểu là top rating toàn cửa hàng.
    if any(k in clean for k in ["truyen tot nhat", "sach tot nhat", "danh gia tot nhat", "nhieu sao nhat"]):
        e["terms"] = []
        e["sort"] = "rating_desc"
        return "search_book", e, False, "global_top_rating"

    return intent, e, False, ""


DIRECT_RESPONSES: dict[str, str] = {
    "greeting": "Chào bạn! Mình là MangaBot. Mình có thể trò chuyện theo ngữ cảnh và hỗ trợ tìm truyện, so sánh sách, giải thích cách mua hàng, thanh toán, giao hàng, voucher, hủy/hoàn tiền và kiểm tra đơn của bạn khi cần.",
    "capabilities": "Mình có thể giúp bạn chọn truyện theo sở thích/ngân sách, giải thích các chức năng của Manga World, so sánh COD với thanh toán online, hướng dẫn đặt hàng và chỉ tra dữ liệu MySQL khi câu hỏi cần giá, tồn kho, đánh giá, voucher hoặc trạng thái đơn thực tế.",
    "thanks": "Rất vui vì đã giúp được bạn. Nếu muốn, bạn có thể tiếp tục hỏi dựa trên những truyện hoặc thông tin mình vừa nêu; mình sẽ giữ ngữ cảnh cuộc trò chuyện.",
    "goodbye": "Chào bạn, hẹn gặp lại! Khi cần tìm truyện hoặc kiểm tra đơn hàng, cứ mở MangaBot nhé.",
    "cod_info": "COD là thanh toán khi nhận hàng. Bạn đặt đơn trước, cửa hàng xử lý và giao hàng; khi nhận hàng bạn thanh toán cho đơn vị giao hàng theo quy trình của Manga World.",
    "online_payment": "Thanh toán online của bản demo Manga World dùng luồng VietQR/MangaPay Sandbox. Sau khi thanh toán mô phỏng thành công, đơn được ghi nhận và chuyển sang chờ cửa hàng xác nhận.",
    "payment_compare": "COD phù hợp khi bạn muốn nhận hàng rồi mới thanh toán. Online thuận tiện khi muốn hoàn tất thanh toán ngay; với đơn online đã thanh toán mà bị hủy, hệ thống có quy trình cung cấp thông tin nhận hoàn tiền và nhân viên xác nhận hoàn tiền.",
    "cancel_order": "Bạn có thể hủy đơn khi đơn vẫn ở trạng thái cho phép hủy. Hệ thống hoàn lại tồn kho/voucher; nếu đơn online đã thanh toán bị hủy thì chuyển sang quy trình chờ hoàn tiền.",
    "refund_info": "Với đơn online đã thanh toán nhưng bị hủy, khách cung cấp Ngân hàng, Số tài khoản và Tên chủ tài khoản trong mục Đơn hàng. Khi thông tin đầy đủ, nhân viên mới xác nhận đã hoàn tiền.",
    "how_to_order": "Để đặt hàng: tìm hoặc mở chi tiết truyện → thêm vào giỏ → kiểm tra số lượng → sang thanh toán → nhập/kiểm tra số điện thoại và địa chỉ → chọn COD hoặc Online → xác nhận đặt hàng. Sau đó bạn theo dõi trạng thái trong mục Đơn hàng.",
    "account_help": "Trong phần tài khoản, bạn có thể cập nhật thông tin cá nhân như số điện thoại và địa chỉ. Những dữ liệu này có thể được tự điền ở bước thanh toán nhưng bạn vẫn được sửa cho từng đơn hàng.",
    "review_info": "Ở chi tiết truyện, bạn có thể chọn trực tiếp từ 1 đến 5 sao. Khi chọn sao thứ N, các sao từ 1 đến N được tô nổi; nội dung đánh giá phải thỏa các ràng buộc của form trước khi gửi.",
    "manga_info": "Manga là truyện tranh có nguồn gốc và phong cách phát triển mạnh tại Nhật Bản. Manga có nhiều thể loại và nhóm độc giả khác nhau; trên Manga World bạn có thể tìm theo tên, tác giả, thể loại, giá và đánh giá.",
    "anime_info": "Anime thường chỉ hoạt hình theo phong cách Nhật Bản. Một số anime được chuyển thể từ manga, nhưng manga là truyện tranh còn anime là tác phẩm hoạt hình; chúng có thể khác nhau về nội dung, nhịp kể và cách thể hiện.",
    "conversation_decline": "Được, mình sẽ không áp dụng gợi ý đó. Bạn có thể đổi tiêu chí hoặc hỏi tiếp dựa trên danh sách trước.",
}


def response_mode(intent: str, entities: dict[str, Any], context_used: bool) -> tuple[str, bool]:
    database_intents = {
        "search_book", "price_filter", "best_seller", "new_books", "book_detail",
        "compare_books", "compare_context", "order_status", "voucher_info", "recommend_personal"
    }
    if intent in database_intents:
        return "database", True
    if intent == "refund_info" and (entities.get("order_id") or entities.get("latest_order")):
        return "database", True
    if intent == "shipping_info":
        # Có thể dùng địa chỉ đã lưu trong tài khoản, nhưng không cần dữ liệu sách.
        return "business_rule", False
    if intent in DIRECT_RESPONSES:
        return "knowledge" if intent not in {"greeting", "thanks", "goodbye", "conversation_decline"} else "conversation", False
    if intent in {"welcome"}:
        return "conversation", False
    if context_used:
        return "conversation", False
    return "clarify", False


def follow_up_plan(intent: str, entities: dict[str, Any], context: dict[str, Any]) -> dict[str, Any] | None:
    if intent in {"search_book", "price_filter", "best_seller", "new_books"}:
        if entities.get("sort") != "rating_desc":
            return {
                "when": "after_results",
                "action": "top_rated_from_last_results",
                "limit": 3,
                "question": "Bạn có muốn mình chọn 3 truyện có đánh giá tốt nhất trong danh sách này không?",
                "quick": "Có, chọn 3 truyện đánh giá tốt nhất",
            }
        return {
            "when": "after_results",
            "action": "cheapest_from_last_results",
            "limit": 3,
            "question": "Bạn có muốn mình lọc tiếp các truyện có giá dễ mua nhất trong danh sách này không?",
            "quick": "Có, chọn truyện rẻ nhất trong danh sách",
        }
    if intent == "cod_info":
        return {"when": "now", "action": "compare_payment_methods", "question": "Bạn có muốn mình so sánh COD với thanh toán online không?", "quick": "So sánh COD và thanh toán online"}
    if intent == "online_payment":
        return {"when": "now", "action": "explain_refund", "question": "Bạn muốn mình giải thích thêm quy trình hoàn tiền khi đơn online bị hủy không?", "quick": "Quy trình hoàn tiền online"}
    if intent == "how_to_order":
        return {"when": "now", "action": "find_budget_books", "question": "Bạn muốn mình tìm thử vài truyện theo ngân sách của bạn không?", "quick": "Tìm truyện dưới 50k còn hàng"}
    if intent == "capabilities":
        return {"when": "now", "action": "start_recommendation", "question": "Bạn muốn bắt đầu bằng thể loại yêu thích, mức giá hay sách bán chạy?", "quick": "Sách bán chạy"}
    if intent == "refund_info":
        return {"when": "now", "action": "check_latest_order", "question": "Bạn có muốn mình kiểm tra đơn gần nhất của bạn đang ở trạng thái nào không?", "quick": "Đơn gần nhất của tôi"}
    if intent == "recommend_personal":
        return {"when": "now", "action": "collect_recommendation_preferences", "question": "Bạn có thể trả lời ngắn như “hành động dưới 50k” hoặc “trinh thám đánh giá cao”.", "quick": "Hành động dưới 50k"}
    return None


def analyze_payload(message: str, body: dict[str, Any]) -> dict[str, Any]:
    base = predict_payload(message)
    context = _safe_context(body)
    intent, entities, context_used, context_reason = contextualize(
        message,
        str(base.get("intent", "out_of_scope")),
        dict(base.get("entities", {})),
        context,
    )

    # Nếu người dùng chỉ nói "gợi ý cho tôi" mà chưa đưa tiêu chí và chưa có danh sách trước,
    # hỏi thêm thay vì truy vấn MySQL một cách mơ hồ.
    clean = plain(message)
    clarify_answer = ""
    if intent == "recommend_personal" and not context_used:
        meaningful_terms = [x for x in entities.get("terms", []) if x not in {"tot", "hay", "phu", "hop"}]
        has_filters = any(k in entities for k in ["min_price", "max_price", "min_rating", "max_stock"])
        if not meaningful_terms and not has_filters:
            clarify_answer = "Được. Bạn thích thể loại nào hoặc muốn mức giá khoảng bao nhiêu? Ví dụ: hành động dưới 50k, trinh thám đánh giá cao, hoặc truyện nhẹ nhàng còn hàng."

    mode, uses_database = response_mode(intent, entities, context_used)
    direct_answer = DIRECT_RESPONSES.get(intent, "")
    if clarify_answer:
        mode, uses_database, direct_answer = "clarify", False, clarify_answer
    elif mode == "clarify" and not direct_answer:
        direct_answer = "Mình chưa chắc bạn đang muốn tìm dữ liệu cửa hàng hay chỉ cần tư vấn. Bạn có thể nói rõ hơn một chút, ví dụ thể loại, ngân sách, tên truyện hoặc vấn đề cần hỗ trợ."

    follow_up = follow_up_plan(intent, entities, context)
    base.update({
        "intent": intent,
        "entities": entities,
        "context_used": context_used,
        "context_reason": context_reason,
        "response_mode": mode,
        "uses_database": uses_database,
        "direct_answer": direct_answer,
        "follow_up": follow_up,
        "conversation_memory": {
            "history_items_received": len(context.get("history", [])),
            "previous_result_count": len(context.get("last_result_ids", [])),
            "pending_action": context.get("pending_action", ""),
        },
        "model": "Contextual Hybrid NLU: TF-IDF + Logistic Regression + entity parser + conversation resolver",
    })
    return base


@app.get("/")
def index():
    return jsonify({
        "ok": True,
        "service": "Manga World AI Service",
        "message": "Dịch vụ phân tích ngôn ngữ MangaBot đang hoạt động.",
        "endpoints": ["/health", "/predict-intent", "/analyze", "/recommend"],
        "conversation": "POST /analyze accepts history + context and resolves follow-up references.",
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
        "nlu_version": "2026-08-10-contextual-hybrid-v3.1",
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
    return jsonify(analyze_payload(message, body))


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
