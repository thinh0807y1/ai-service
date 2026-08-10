# Manga World AI Service - Hybrid NLU v2

Bản này nâng cấp AI Service cho MangaBot. AI Service không trực tiếp trả dữ liệu sách; nó phân tích ý định và điều kiện trong câu hỏi, còn website PHP truy vấn MySQL để tạo câu trả lời cuối cùng.

## Endpoints
- `GET /health`
- `POST /predict-intent`
- `POST /analyze` (endpoint chính cho website mới)
- `POST /recommend` (giữ để tương thích bản cũ)

## Ví dụ `/analyze`
Input:
```json
{"message":"cho tôi 3 truyện trinh thám dưới 50k còn hàng đánh giá từ 4 sao"}
```

Kết quả chứa intent và entities như `limit`, `max_price`, `min_rating`, `in_stock`, `terms`, `sort`.

## Lưu ý deploy
- Giữ `requirements.txt` hiện tại, trong đó scikit-learn được pin 1.6.1 để phù hợp model đã lưu.
- Start command: `gunicorn app:app --bind 0.0.0.0:$PORT`
- Health check: `/health`
