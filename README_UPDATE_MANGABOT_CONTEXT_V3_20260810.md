# MangaBot Contextual Hybrid NLU v3.1

Ngày cập nhật: 2026-08-10

## Mục tiêu

- MangaBot giữ ngữ cảnh hội thoại trong phiên chat.
- Hiểu các câu nối tiếp như: `từ đó`, `trong số đó`, `cái nào hay nhất?`, `dưới 30k thôi`, `cuốn thứ 2`, hoặc chỉ `có` sau một câu hỏi gợi ý của bot.
- Chỉ yêu cầu website truy vấn MySQL khi câu hỏi thực sự cần dữ liệu hiện tại như sách, giá, tồn kho, đánh giá, voucher hoặc đơn hàng.
- Các câu hỏi hướng dẫn/kiến thức như COD, thanh toán online, cách đặt hàng, Manga/Anime... có thể được trả lời trực tiếp theo ngữ cảnh mà không cần truy vấn dữ liệu sách.
- Sau khi có danh sách truyện, bot có thể chủ động hỏi tiếp một câu hợp lý, ví dụ: `Bạn có muốn mình chọn 3 truyện có đánh giá tốt nhất trong danh sách này không?`

## Endpoint

### GET /health
Kiểm tra service. Phiên bản mới trả:

`nlu_version = 2026-08-10-contextual-hybrid-v3.1`

### POST /analyze
Body mới hỗ trợ:

```json
{
  "message": "gợi ý cho tôi 3 truyện tốt nhất từ đó",
  "history": [
    {"role": "user", "content": "truyện anime giá dưới 40000"},
    {"role": "assistant", "content": "Mình tìm thấy 5 truyện..."}
  ],
  "context": {
    "last_intent": "price_filter",
    "last_entities": {"max_price": 40000},
    "last_result_ids": [1, 2, 3, 4, 5],
    "last_result_titles": ["..."],
    "pending_action": "top_rated_from_last_results",
    "pending_limit": 3
  }
}
```

Service có thể trả thêm:

- `context_used`
- `context_reason`
- `response_mode`
- `uses_database`
- `direct_answer`
- `follow_up`
- `entities.restrict_ids`
- `entities.focus_book_id`
- `entities.compare_ids`

## Ví dụ hội thoại

### Ví dụ 1: lọc tiếp danh sách trước

1. User: `truyện anime giá dưới 40000`
2. Website trả 5 truyện từ MySQL.
3. Bot hỏi: `Bạn có muốn mình chọn 3 truyện có đánh giá tốt nhất trong danh sách này không?`
4. User: `có`
5. `/analyze` trả `restrict_ids` là 5 truyện trước, `sort=rating_desc`, `limit=3`.
6. Website chỉ xếp hạng 3 truyện tốt nhất trong chính danh sách đó.

User cũng có thể nói trực tiếp:

`gợi ý cho tôi 3 truyện tốt nhất từ đó`

### Ví dụ 2: tham chiếu thứ tự

Sau khi bot vừa trả 5 truyện:

`Tác giả cuốn thứ 2 là ai?`

Service trả `intent=book_detail` và `focus_book_id` là id của truyện thứ 2 trong kết quả trước.

### Ví dụ 3: câu nối tiếp ngắn

Sau danh sách trước:

- `dưới 30k thôi`
- `cái nào hay nhất?`
- `rẻ nhất trong số đó`
- `so sánh 2 cuốn đầu`

đều được hiểu dựa trên ngữ cảnh trước.

### Ví dụ 4: không cần MySQL

- `COD là gì?`
- `so sánh COD và thanh toán online`
- `Manga là gì?`
- `Anime là gì?`
- `cách đặt hàng`

Service trả `uses_database=false` và `direct_answer`.

## Lưu ý quan trọng

AI Service không trực tiếp kết nối MySQL. Website PHP giữ context/result ids trong session và gửi context lên `/analyze`. Sau đó PHP mới quyết định có truy vấn MySQL hay không.

Vì vậy để tính năng nhớ ngữ cảnh hoạt động đầy đủ, cần cập nhật đồng thời:

- AI Service trên Render.
- `php/api/mangabot.php` trên InfinityFree.
- `react/assets/app.js` + HTML version để giao diện hiển thị đúng trạng thái `Có/Không dùng CSDL`.
