# NeoRAG Cup 2026 — S2B Knowledge Assistant

<div align="center">

<img src="docs/assets/neorag-banner.png" alt="NeoRAG Cup 2026 Banner" width="750">

<p align="center">
  <b>Hệ thống Retrieval-Augmented Generation (RAG) tối ưu hóa truy xuất và phản hồi thông tin cộng đồng S2B</b><br>
  <i>(CLB Lập Trình PTIT · CLB Multimedia PTIT · CLB Nhà Sáng Tạo Game PTIT)</i>
</p>

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![LLM](https://img.shields.io/badge/LLM-claude--haiku--5--5-blueviolet.svg)](https://proptit.dekthe.dev/)
[![Embedding](https://img.shields.io/badge/Embedding-bge--m3-orange.svg)](https://proptit.dekthe.dev/)
[![Retrieval Score](https://img.shields.io/badge/Retrieval%20Score-88.38%25-brightgreen.svg)](#kết-quả-đánh-giá)
[![Abstention](https://img.shields.io/badge/Abstention-100%25-success.svg)](#kết-quả-đánh-giá)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Thí sinh:** Lê Thị Thủy (Team AI – CLB Lập Trình PTIT)  
**Tài khoản:** `lethithuy` | **Mã nguồn:** [GitHub Repository](https://github.com/LeThuyyy/NeoRAG-Cup-2026-Sub)

</div>

---

## 📌 Mục lục
- [1. Giới thiệu tổng quan](#1-giới-thiệu-tổng-quan)
- [2. Kiến trúc giải pháp (System Architecture)](#2-kiến-trúc-giải-pháp-system-architecture)
- [3. Các đột phá kỹ thuật chính](#3-các-đột-phá-kỹ-thuật-chính)
- [4. Kết quả đánh giá (Benchmark Results)](#4-kết-quả-đánh-giá-benchmark-results)
- [5. Cấu trúc thư mục](#5-cấu-trúc-thư-mục)
- [6. Hướng dẫn cài đặt & Chạy thực nghiệm](#6-hướng-dẫn-cài-đặt--chạy-thực-nghiệm)
- [7. Chatbot Demo (Product Showcase)](#7-chatbot-demo-product-showcase)

---

## 1. Giới thiệu tổng quan

**NeoRAG Cup 2026** là cuộc thi thiết kế giải pháp RAG chuyên sâu nhằm giải quyết bài toán hỏi đáp thông tin chính xác, đa chiều về cộng đồng **S2B** tại Học viện Công nghệ Bưu chính Viễn thông (gồm 3 CLB: **ProPTIT**, **MPC**, **GCC**).

### Thách thức chính của bài toán:
1. **Dữ liệu phân tán & đa thực thể:** Kho tài liệu gồm 118 đoạn văn bản với nhiều chủ đề đan xen (lịch tuyển sinh, cơ cấu nhân sự, lộ trình kỹ thuật, giải thưởng, quy chế nội bộ).
2. **Đặc thù câu hỏi sinh viên:** Người dùng thường gõ tắt, teencode (*clb lt, khmt, bcn, ttv...*), gõ không dấu, hoặc đặt câu hỏi so sánh chéo giữa các CLB.
3. **Hiện tượng "Ảo giác" & Câu hỏi ngoài phạm vi:** Khoảng 7–9% câu hỏi không có dữ liệu trong tài liệu, yêu cầu hệ thống phải từ chối chính xác tuyệt đối mà không bị ảo giác dữ kiện.
4. **Hạn chế tài nguyên & Gateway:** Ngân sách LLM cố định ($5), mô hình giới hạn duy nhất `claude-haiku-5-5` với `temperature=None`.

---

## 2. Kiến trúc giải pháp (System Architecture)

Giải pháp của chúng tôi xây dựng một **End-to-End Hybrid RAG Pipeline** hoàn chỉnh với 4 tầng xử lý:

```
[ CÂU HỎI NGƯỜI DÙNG ]
        │
        ▼
┌────────────────────────────────────────────────────────────────────────┐
│ TẦNG 1: QUERY PREPROCESSING & EXPANSION                                │
│  • Chuẩn hóa regex: Mở rộng từ viết tắt, teencode, ban ngành          │
│  • Bổ sung unaccented tokens (hỗ trợ câu hỏi gõ không dấu)             │
│  • Phân tách truy vấn đa thực thể (Multi-Entity Decomposition)         │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
        ┌──────────────────────────┴──────────────────────────┐
        ▼                                                     ▼
┌───────────────────────────────┐     ┌───────────────────────────────────┐
│ TẦNG 2A: DENSE RETRIEVAL      │     │ TẦNG 2B: DUAL-INDEX BM25          │
│  • Vector Embedding BGE-M3    │     │  • Unigram + Bigram Tiếng Việt   │
│  • Metadata Enrichment        │     │  • Song song: Có dấu + Không dấu │
│    (f"{club} - {topic}: {t}") │     │  • Khử nhiễu TF-IDF tối ưu        │
└───────────────┬───────────────┘     └─────────────────┬─────────────────┘
                │                                       │
                └──────────────────┬────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ TẦNG 3: FUSION, CLUB ROUTING & DYNAMIC CONTEXT                         │
│  • Reciprocal Rank Fusion (RRF, k=20) kết hợp trọng số Dense & Sparse  │
│  • Soft Club Routing: Nhân hệ số 1.25x cho CLB đích, 0.85x CLB khác   │
│  • Round-robin Interleaving: Xen kẽ kết quả cho câu hỏi so sánh        │
│  • Dynamic Context Window:                                             │
│      - Câu hỏi so sánh/đa CLB ──> Giữ Top 10 documents                 │
│      - Câu hỏi đơn CLB        ──> Co gọn Top 7 documents (chống nhiễu) │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ TẦNG 4: CONSTRAINED GENERATION (CLAUDE HAIKU 5.5)                      │
│  • Súc tích & Trực diện: 1-3 câu, nêu rõ chủ thể, không lan man       │
│  • Bảo toàn 100% dữ kiện số liệu, mốc thời gian, liên kết URL          │
│  • Phân định rạch ròi:                                                 │
│      - Thiếu dữ kiện con: Nêu rõ chưa có và trích dữ kiện liên quan    │
│      - Hoàn toàn ngoài tài liệu: Từ chối chuẩn (Abstention = 100%)    │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
                           [ CÂU TRẢ LỜI CUỐI ]
```

---

## 3. Các đột phá kỹ thuật chính

### 1. Dual-Index BM25 (Hỗ trợ câu hỏi không dấu & teencode)
- Xây dựng chỉ mục BM25Okapi thuần Python tối ưu cho tiếng Việt với cả **unigram** và **bigram**.
- Tích hợp hàm `strip_accents` tạo thêm luồng chỉ mục không dấu song song. Với những câu hỏi người dùng gõ không dấu (như `train-0109`), tài liệu trúng đích vẫn đạt điểm BM25 cao mà không bị trôi thứ hạng.
- Bộ luật Regex chuẩn hóa tự động hơn 40 từ khóa viết tắt (*dpt, khmt, ttv, sv, hv, bcn, bởn code, bên vẽ, người đứng lớp...*).

### 2. Dynamic Context Window (Bảo toàn dữ liệu mà vẫn chống nhiễu)
- **Vấn đề giải quyết:** Nếu cắt cố định ngữ cảnh xuống 3-4 tài liệu sẽ đánh mất dữ kiện ở các câu hỏi so sánh phức tạp; ngược lại nếu luôn nhồi 10 tài liệu thì câu hỏi đơn CLB sẽ bị loãng thông tin.
- **Giải pháp:** 
  - Hệ thống luôn trả về đủ **10 `retrieved_ids`** cho bộ chấm để tối đa hóa `Recall@5`, `MRR@5`, `NDCG@5`.
  - Phân tích câu hỏi theo thời gian thực: Với câu hỏi so sánh 3 CLB, đưa toàn bộ **Top 10** vào ngữ cảnh; với câu hỏi đơn lẻ, co gọn xuống **Top 7** có độ tương quan cao nhất.

### 3. Tinh chỉnh Prompt chống ảo giác & Tối ưu Relevancy
- **Trọng tâm (Conciseness):** Cắt bỏ hoàn toàn các câu mở đầu rào đón, loại bỏ các chi tiết phụ không được hỏi $\rightarrow$ Tăng vọt độ chính xác $F_1$ cho **ROUGE-L** (từ 57.89% lên 65.86%).
- **Xử lý câu hỏi thiếu dữ kiện (Partial Data):** Phân biệt rõ giữa *câu hỏi thiếu thuộc tính con* (như hỏi thời gian tuyển sinh năm 2027) và *câu hỏi hoàn toàn lạc đề*, giúp giữ trọn điểm String Presence và ROUGE-L ở các case khó mà vẫn bảo toàn 100 điểm **Abstention**.

---

## 4. Kết quả đánh giá (Benchmark Results)

### Bảng so sánh Before vs After (Trên tập Train - 126 câu hỏi chuẩn hóa):

| Phân nhóm | Chỉ số (Metric) | Baseline ban đầu | Sau tối ưu (Ours) | Mức tăng trưởng ($\Delta$) |
| :--- | :--- | :---: | :---: | :---: |
| **Retrieval** | **Retrieval Score** | 86.01% | **88.38%** | **+2.37%** 🚀 |
| | **Recall@5** | 87.94% | **92.83%** | **+4.89%** 🚀 |
| | **Hit@5** | 97.54% | **99.18%** | **+1.64%** |
| | **Hit@7** | 99.18% | **100.00%** | **+0.82%** ⭐ (100% trúng đích) |
| | **NDCG@5** | 82.18% | **85.28%** | **+3.10%** 🚀 |
| | **MRR@5** | 87.90% | **87.05%** | Giữ vững thứ hạng cao |
| **Generation** | **ROUGE-L** | 57.89% | **65.86%** | **+7.97%** 🔥 (Tăng vọt) |
| | **BLEU-4** | 44.42% | **51.40%** | **+6.98%** 🚀 |
| | **String Presence** | 88.46% | **88.06%** | Duy trì độ phủ từ khóa |
| | **Groundedness** | 99.21% | **99.15%** | Gần như tuyệt đối |
| | **Abstention** | 100.00% | **100.00%** | **100%** (Không ảo giác) |
| **Hiệu năng** | **Median Latency** | ~3.0s | **3.0s** | Phản hồi cực nhanh |

---

## 5. Cấu trúc thư mục

```
NeoRAG-Cup-2026-Sub/
├── app/
│   └── chat.py                     # Ứng dụng Web Chatbot tương tác (Streamlit)
├── configs/
│   └── eval.toml                   # Cấu hình thang điểm và trọng số đánh giá
├── data/
│   ├── corpus.jsonl                # Kho 118 đoạn văn tri thức về cộng đồng S2B
│   ├── train/questions.jsonl       # 126 câu hỏi phát triển kèm Ground Truth
│   ├── public_test/questions.jsonl # 42 câu hỏi Public Test
│   └── private_test/questions.jsonl# 42 câu hỏi Private Test chính thức
├── docs/                           # Tài liệu quy chế, hướng dẫn, gateway
├── outputs/                        # File bài nộp (.jsonl, .zip) và báo cáo đánh giá
├── scripts/                        # Các công cụ hỗ trợ tự động hóa, kiểm thử
├── src/neorag/
│   ├── pipelines/
│   │   ├── base.py                 # Lớp RAGPipeline cơ sở
│   │   └── baseline.py             # Pipeline tối ưu hóa chính thức của đội thi
│   ├── embeddings.py               # Module vector embedding (BGE-M3)
│   ├── llm.py                      # Client kết nối Gateway Claude Haiku có retry/cache
│   ├── predict.py & evaluate.py    # Bộ máy dự đoán và đánh giá chính thức
│   └── text.py                     # Xử lý ngôn ngữ tự nhiên tiếng Việt
└── README.md                       # Báo cáo kỹ thuật tổng quan dự án
```

---

## 6. Hướng dẫn cài đặt & Chạy thực nghiệm

### 1. Khởi tạo môi trường
```powershell
# Tạo và kích hoạt môi trường ảo (Windows PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Cài đặt thư viện phụ thuộc
python scripts/install.py

# Cấu hình API key cuộc thi
copy .env.example .env
# Điền OPENAI_API_KEY vào file .env
```

### 2. Chạy pipeline và đánh giá
```bash
# Đánh giá nhanh trên tập train (Retrieval + Lexical, không tốn token Judge)
neorag run train --no-judge

# Đánh giá toàn diện có LLM Judge trên tập train
neorag run train

# Sinh bài nộp cho Public Test (tạo outputs/public_test_submission.zip)
neorag run public_test

# Sinh bài nộp cho Private Test (tạo outputs/private_test_submission.zip)
neorag run private_test
```

---

## 7. Chatbot Demo (Product Showcase)

Dự án cung cấp giao diện Chatbot hỏi đáp trực quan được xây dựng bằng **Streamlit**, cho phép người dùng hỏi đáp theo thời gian thực với đầy đủ trích dẫn tài liệu:

```powershell
# Chạy ứng dụng Chatbot
python -m streamlit run app/chat.py
```

Chatbot hỗ trợ:
- Trả lời thông minh các câu hỏi về quy chế, tuyển sinh, sự kiện của ProPTIT, MPC, GCC.
- Trực quan hóa nguồn tài liệu trích dẫn (`retrieved_ids`) kèm độ trễ phản hồi (latency).
- Phản hồi súc tích, đáng tin cậy và nói không với thông tin sai lệch ngoài tài liệu.

---

<div align="center">
  <b>NeoRAG Cup 2026 — Team AI, CLB Lập Trình PTIT (ProPTIT)</b><br>
  <i>"Chia sẻ để cùng nhau phát triển"</i>
</div>
