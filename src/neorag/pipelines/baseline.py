from __future__ import annotations

import math
import os
import re
import unicodedata
import numpy as np

from neorag.data import Document
from neorag.llm import LLMClient
from neorag.pipelines.base import RAGOutput, RAGPipeline
from neorag.text import tokenize

SYSTEM_PROMPT = (
    "Bạn là trợ lý AI chuyên nghiệp trả lời các câu hỏi về cộng đồng S2B "
    "(bao gồm CLB Lập Trình PTIT - ProPTIT, CLB Multimedia PTIT - MPC, và CLB Nhà Sáng Tạo Game PTIT - GCC) "
    "tại Học viện Công nghệ Bưu chính Viễn thông (PTIT)."
)

PROMPT = """Hãy đọc kỹ phần TÀI LIỆU và trả lời CÂU HỎI một cách chính xác, đúng trọng tâm, súc tích và bám sát tài liệu.

QUY TẮC BẮT BUỘC:
1. TRẢ LỜI ĐÚNG TRỌNG TÂM VÀ SÚC TÍCH:
   - Chỉ trả lời đúng những ý mà CÂU HỎI yêu cầu, ngắn gọn (thường từ 1-3 câu), đi thẳng vào nội dung chính.
   - Nêu rõ tên CLB hoặc chủ thể được hỏi một cách tự nhiên để câu trả lời độc lập và trọn vẹn ngữ nghĩa.
   - Nếu câu hỏi hỏi "Có... không?", "Được... không?", "Đã có ai... chưa?", hãy mở đầu rõ ràng bằng câu khẳng định/phủ định ("Có.", "Không.", "Được.", "Chưa chắc.") rồi giải thích ngắn gọn nguyên do.
   - Tuyệt đối KHÔNG tự ý suy diễn, KHÔNG sao chép lan man toàn bộ tài liệu, KHÔNG liệt kê thêm các công cụ, phần mềm, giải thưởng hay chi tiết phụ ngoài lề không được hỏi trong câu hỏi.
   - Tuyệt đối KHÔNG trích dẫn số thứ tự tài liệu (như "[1]", "[2]") và KHÔNG thêm lời khuyên ngoài lề.

2. ĐẢM BẢO CHÍNH XÁC DỮ KIỆN:
   - Giữ nguyên vẹn và chuẩn xác 100% tất cả các con số, mốc thời gian, ngày tháng năm, tên riêng người, sự kiện, ban ngành, đường link URL và thuật ngữ chuyên ngành có trong tài liệu.

3. XỬ LÝ KHI THIẾU DỮ KIỆN HOẶC CÂU HỎI NGOÀI TÀI LIỆU:
   - Nếu tài liệu có nhắc đến chủ đề/bối cảnh được hỏi nhưng KHÔNG ghi dữ kiện cụ thể (ví dụ: hỏi chức vụ không có tên trong danh sách, hỏi năm tương lai mà tài liệu ghi chỉ áp dụng cho năm 2026, hỏi thời hạn mà tài liệu chỉ nêu ngày bắt đầu):
     Hãy nói rõ tài liệu không ghi dữ kiện đó / chưa có thông tin thời điểm đó, và nêu ngắn gọn thông tin liên quan mà tài liệu thực tế có ghi.
   - Nếu tài liệu HOÀN TOÀN không có bất kỳ thông tin nào liên quan đến câu hỏi:
     Chỉ trả lời một câu: "Xin lỗi, tôi không tìm thấy thông tin này trong tài liệu."
   - Nếu câu hỏi gồm nhiều ý/nhiều CLB mà tài liệu chỉ có thông tin một phần:
     Trả lời phần có thông tin, còn phần thiếu nói rõ: "Về [phần thiếu], tài liệu hiện không có thông tin."

TÀI LIỆU:
{context}

CÂU HỎI: {query}
TRẢ LỜI:"""

# PROMPT CHUẨN HÓA & PHÂN TÁCH TRUY VẤN ĐA THỰC THỂ CHO RETRIEVAL
REWRITE_SYSTEM = (
    "Bạn là trợ lý chuẩn hóa và tối ưu truy vấn tìm kiếm cho cộng đồng S2B tại Học viện Công nghệ Bưu chính Viễn thông (PTIT), "
    "gồm 3 câu lạc bộ:\n"
    "1. CLB Lập Trình PTIT (ProPTIT)\n"
    "2. CLB Multimedia PTIT (MPC)\n"
    "3. CLB Nhà Sáng Tạo Game PTIT (GCC)\n\n"
    "Nhiệm vụ: Viết lại câu hỏi thành câu truy vấn tìm kiếm tiếng Việt rõ ràng, ĐẦY ĐỦ DẤU, tập trung vào từ khóa quan trọng "
    "(tên người, số liệu, năm, sự kiện, giải thưởng, công nghệ, mốc thời gian, quy định, từ đồng nghĩa liên quan).\n\n"
    "QUY TẮC BẮT BUỘC:\n"
    "- GIỮ NGUYÊN VẸN Ý NGHĨA VÀ TỪ KHÓA GỐC: Đặc biệt là các từ chỉ thời gian, tần suất (bao lâu 1 lần, khi nào, mấy buổi, định kỳ), "
    "đối tượng (ai, chức vụ gì), số liệu. Không được tự ý suy diễn hoặc thay thế bằng từ khóa khác làm lệch hướng tìm kiếm.\n"
    "- Nếu câu hỏi so sánh hoặc hỏi chung nhiều CLB trong cộng đồng S2B (ví dụ: slogan của mỗi CLB, quy trình tuyển của 3 CLB, "
    "link đăng ký của các bên, CLB nào lâu đời nhất, trẻ nhất), hãy TÁCH thành các dòng truy vấn riêng biệt cho từng CLB liên quan, "
    "mỗi dòng đúng 1 truy vấn.\n"
    "- Nếu câu hỏi chỉ hỏi về 1 CLB hoặc 1 chủ đề cụ thể (ví dụ chỉ hỏi về CLB Multimedia, hoặc chỉ hỏi về ProPTIT), "
    "TUYỆT ĐỐI CHỈ TRẢ VỀ ĐÚNG 1 DÒNG TRUY VẤN DUY NHẤT, không được tự ý tách ra các CLB khác.\n"
    "- Tuyệt đối không thêm lời giải thích, không thêm lời chào, không đánh số thứ tự đầu dòng, không dùng dấu ngoặc kép."
)

VIETNAMESE_DIACRITICS = set(
    "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
    "ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ"
)


def strip_accents(text: str) -> str:
    """Loại bỏ dấu tiếng Việt để hỗ trợ tìm kiếm không dấu."""
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d").replace("Đ", "D")


# Danh sách mở rộng từ viết tắt / teencode phổ biến (hỗ trợ cả có dấu và không dấu)
ABBREVIATION_RULES: list[tuple[str, str]] = [
    # ProPTIT
    (r"\bclb\s+lt\s+ptit\b", "CLB Lập Trình PTIT (ProPTIT)"),
    (r"\bclb\s+lt\b", "CLB Lập Trình PTIT (ProPTIT)"),
    (r"\bclb\s+l[aà]p\s+tr[iì]nh\s+ptit\b", "CLB Lập Trình PTIT (ProPTIT)"),
    (r"\bclb\s+l[aà]p\s+tr[iì]nh\b", "CLB Lập Trình PTIT (ProPTIT)"),
    (r"\bclb\s+code\b", "CLB Lập Trình PTIT (ProPTIT)"),
    (r"\bproptit\b", "CLB Lập Trình PTIT (ProPTIT)"),
    # MPC / Multimedia
    (r"\bclb\s+mm\s+ptit\b", "CLB Multimedia PTIT (MPC)"),
    (r"\bclb\s+mm\b", "CLB Multimedia PTIT (MPC)"),
    (r"\bclb\s+đ[aà]\s+ph[uư][oơ]ng\s+ti[eệ]n\s+ptit\b", "CLB Multimedia PTIT (MPC)"),
    (r"\bclb\s+da\s+phuong\s+tien\s+ptit\b", "CLB Multimedia PTIT (MPC)"),
    (r"\bclb\s+đ[aà]\s+ph[uư][oơ]ng\s+ti[eệ]n\b", "CLB Multimedia PTIT (MPC)"),
    (r"\bclb\s+da\s+phuong\s+tien\b", "CLB Multimedia PTIT (MPC)"),
    (r"\bmpc\b", "CLB Multimedia PTIT (MPC)"),
    # GCC / Game
    (r"\bclb\s+game\s+ptit\b", "CLB Nhà Sáng Tạo Game PTIT (GCC)"),
    (r"\bclb\s+game\b", "CLB Nhà Sáng Tạo Game PTIT (GCC)"),
    (r"\bclb\s+l[aà]m\s+game\s+ptit\b", "CLB Nhà Sáng Tạo Game PTIT (GCC)"),
    (r"\bclb\s+l[aà]m\s+game\b", "CLB Nhà Sáng Tạo Game PTIT (GCC)"),
    (r"\bclb\s+nh[aà]\s+s[aá]ng\s+t[aạ]o\s+game\s+ptit\b", "CLB Nhà Sáng Tạo Game PTIT (GCC)"),
    (r"\bclb\s+nh[aà]\s+s[aá]ng\s+t[aạ]o\s+game\b", "CLB Nhà Sáng Tạo Game PTIT (GCC)"),
    (r"\bgcc\b", "CLB Nhà Sáng Tạo Game PTIT (GCC)"),
    # Ban / Nhánh
    (r"\b(b[eê]n\s+code|nh[aá]nh\s+code)\b", "nhánh Dev (lập trình)"),
    (r"\b(b[eê]n\s+v[eẽ]|nh[aá]nh\s+v[eẽ]|b[eê]n\s+art|nh[aá]nh\s+art)\b", "nhánh Art (thiết kế)"),
    (r"\b(b[eê]n\s+game|nh[aá]nh\s+game)\b", "nhánh Game"),
    # Quy chế & vai trò
    (r"\bng[uư][oơ]i\s+d[uù]ng\s+l[oó]p\b", "người đứng lớp (người chủ trì buổi học)"),
    (r"\b(chuy[eể]n\s+l[iì]ch|đ[oổ]i\s+l[iì]ch)\b", "đổi lịch (thông báo lùi lịch)"),
    # Thuật ngữ sinh viên / chức vụ
    (r"\bdpt\b", "đa phương tiện"),
    (r"\bkhmt\b", "khoa học máy tính"),
    (r"\bttv\b", "tuyển thành viên"),
    (r"\bt[aâ]n\s+sv\b", "tân sinh viên"),
    (r"\bsv\b", "sinh viên"),
    (r"\bhv\b", "Học viện"),
    (r"\btv\b", "thành viên"),
    (r"\bnk\b", "nhiệm kỳ"),
    (r"\bcn\b", "chủ nhiệm"),
    (r"\bph[oó]\s+cn\b", "phó chủ nhiệm"),
    (r"\bbcn\b", "ban chủ nhiệm"),
    (r"\bdk\b", "đăng ký"),
    (r"\bdki\b", "đăng ký"),
    (r"\bdc\b", "được"),
    (r"\bđc\b", "được"),
    (r"\bko\b", "không"),
    (r"\bvs\b", "với"),
    (r"\bj\b", "gì"),
    (r"\bjz\b", "gì vậy"),
    (r"\bv\b", "vậy"),
    (r"\be\b", "em"),
]

TEENCODE_CHECK = re.compile(
    r"\b(clb\s+lt|clb\s+mm|clb\s+code|clb\s+game|clb\s+l[aà]m\s+game|dpt|khmt|ttv|t[aâ]n\s+sv|sv|hv|tv|nk|ph[oó]\s+cn|bcn|dk|dki|dc|đc|ko|vs|jz|b[eê]n\s+code|b[eê]n\s+v[eẽ]|b[eê]n\s+art|ng[uư][oơ]i\s+d[uù]ng\s+l[oó]p)\b",
    re.IGNORECASE,
)

# Nhận diện câu hỏi hỏi về nhiều CLB, so sánh hoặc hỏi chung cộng đồng S2B
MULTI_CLUB_CHECK = re.compile(
    r"(?:"
    r"\bs2b\b|\b3\s*clb\b|\bba\s*clb\b|\bcả\s*3\b|\bcả\s*ba\b|\bmỗi\s*clb\b|\bcác\s*clb\b|\bmấy\s*clb\b|"
    r"lâu\s*đời\s*nhất|trẻ\s*nhất|ít\s*nhất|nhiều\s*nhất|giống\s*nhau|khác\s*nhau|"
    r"(?:proptit|lập\s*trình).*(?:multimedia|mpc|game|gcc)|"
    r"(?:multimedia|mpc).*(?:proptit|lập\s*trình|game|gcc)|"
    r"(?:game|gcc).*(?:proptit|lập\s*trình|multimedia|mpc)"
    r")",
    re.IGNORECASE,
)

# Nhận diện các câu hỏi chứa khái niệm đặc thù cần mở rộng từ đồng nghĩa (synonyms)
DOMAIN_SYNONYMS_CHECK = re.compile(
    r"\b(y\s*tế|sức\s*khỏe|bệnh|sân\s*chơi|cuộc\s*thi|học\s*phí|chi\s*phí|quỹ|tiền|mày\s*mò|dạy\s*lại)\b",
    re.IGNORECASE,
)


def expand_abbreviations(text: str) -> str:
    """Thay thế nhanh các từ viết tắt và teencode phổ biến."""
    res = text
    for pattern, replacement in ABBREVIATION_RULES:
        res = re.sub(pattern, replacement, res, flags=re.IGNORECASE)
    return res


def get_query_club(query: str) -> str | None:
    """Xác định CLB mục tiêu duy nhất của câu hỏi (nếu có) để tăng độ chuẩn xác định tuyến."""
    if MULTI_CLUB_CHECK.search(query):
        return None  # Câu hỏi nhiều CLB, không phân định đơn lẻ

    is_pro = bool(re.search(r"\b(proptit|lập\s*trình)\b", query, re.IGNORECASE))
    is_mpc = bool(re.search(r"\b(mpc|multimedia|đa\s*phương\s*tiện)\b", query, re.IGNORECASE))
    is_gcc = bool(re.search(r"\b(gcc|nhà\s*sáng\s*tạo\s*game|làm\s*game|sweet\s*battle|iai)\b", query, re.IGNORECASE))

    if is_pro and not (is_mpc or is_gcc):
        return "CLB Lập Trình PTIT"
    if is_mpc and not (is_pro or is_gcc):
        return "CLB Multimedia PTIT"
    if is_gcc and not (is_pro or is_mpc):
        return "CLB Nhà Sáng Tạo Game PTIT (GCC)"
    return None


def needs_rewrite(query: str) -> bool:
    """Xác định câu hỏi cần được khôi phục dấu, chuẩn hóa hoặc phân tách đa thực thể."""
    words = query.strip().split()
    # 1. Câu hỏi không dấu
    if len(words) >= 2 and not any(ch in VIETNAMESE_DIACRITICS for ch in query):
        return True
    # 2. Câu hỏi chứa teencode
    if TEENCODE_CHECK.search(query):
        return True
    # 3. Câu hỏi so sánh hoặc hỏi chung nhiều CLB
    if MULTI_CLUB_CHECK.search(query):
        return True
    # 4. Câu hỏi chứa từ khóa cần mở rộng đồng nghĩa
    if DOMAIN_SYNONYMS_CHECK.search(query):
        return True
    return False


class BM25Index:
    """Chỉ mục BM25Okapi thuần Python, tối ưu cho tiếng Việt với unigram và bigram, hỗ trợ cả có dấu và không dấu."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_len: list[int] = []
        self.avg_doc_len: float = 0.0
        self.doc_freqs: list[dict[str, int]] = []
        self.idf: dict[str, float] = {}
        self.n_docs: int = 0

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        words = tokenize(text)
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]
        base_tokens = words + bigrams

        # Thêm tokens không dấu để hỗ trợ câu hỏi gõ không dấu
        unaccented_text = strip_accents(text)
        unaccented_words = tokenize(unaccented_text)
        unaccented_bigrams = [
            f"{unaccented_words[i]}_{unaccented_words[i+1]}"
            for i in range(len(unaccented_words) - 1)
        ]
        return base_tokens + unaccented_words + unaccented_bigrams

    def fit(self, docs: list[str]) -> None:
        self.n_docs = len(docs)
        self.doc_freqs = []
        self.doc_len = []
        df: dict[str, int] = {}

        for doc in docs:
            tokens = self._tokenize(doc)
            self.doc_len.append(len(tokens))
            freqs: dict[str, int] = {}
            for token in tokens:
                freqs[token] = freqs.get(token, 0) + 1
            self.doc_freqs.append(freqs)
            for token in freqs:
                df[token] = df.get(token, 0) + 1

        self.avg_doc_len = sum(self.doc_len) / max(self.n_docs, 1)

        for token, freq in df.items():
            self.idf[token] = math.log(1.0 + (self.n_docs - freq + 0.5) / (freq + 0.5))

    def get_scores(self, query: str) -> np.ndarray:
        q_tokens = self._tokenize(query)
        scores = np.zeros(self.n_docs, dtype=np.float32)

        for token in q_tokens:
            if token not in self.idf:
                continue
            idf_val = self.idf[token]
            for i, freqs in enumerate(self.doc_freqs):
                if token in freqs:
                    tf = freqs[token]
                    denom = tf + self.k1 * (
                        1.0 - self.b + self.b * (self.doc_len[i] / self.avg_doc_len)
                    )
                    scores[i] += idf_val * (tf * (self.k1 + 1.0)) / denom
        return scores


class GatewayEmbedder:
    """Embedding qua gateway (cùng key với LLM)."""

    def __init__(self, llm: LLMClient, model_name: str):
        self.llm, self.model_name = llm, model_name

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.asarray(self.llm.embed(texts, model=self.model_name), dtype=np.float32)
        norm = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norm, 1e-9)


class BaselinePipeline(RAGPipeline):
    def __init__(
        self,
        top_k: int = 10,
        n_retrieved: int = 10,
        llm_model: str | None = None,
        embedding_model: str | None = None,
        llm: LLMClient | None = None,
        embedder=None,
    ):
        self.top_k = top_k  # số doc tối đa có thể đưa vào context cho LLM đọc
        self.n_retrieved = n_retrieved  # số doc trả về để chấm retrieval (tối đa 10 theo eval.toml)
        self.llm_model = llm_model or os.getenv("LLM_MODEL", "claude-haiku-5-5")
        self.llm = llm or LLMClient(cache_path=".cache/pipeline.sqlite")
        self.embedder = embedder or GatewayEmbedder(
            self.llm, embedding_model or os.getenv("EMBEDDING_MODEL", "bge-m3")
        )

    def index(self, corpus: list[Document]) -> None:
        self.docs = corpus

        # 1. Làm giàu văn bản với Metadata (CLB & Chủ đề) cho cả BM25 và BGE-M3
        enriched_texts = [f"{d.club} - {d.topic}: {d.text}" for d in corpus]

        # 2. Xây dựng chỉ mục BM25
        self.bm25 = BM25Index()
        self.bm25.fit(enriched_texts)

        # 3. Lập vector embedding với bge-m3
        self.doc_vectors = self.embedder.embed(enriched_texts)

    def rewrite_query(self, query: str) -> str:
        """Dùng Claude Haiku khôi phục dấu và tách truy vấn con cho từng CLB."""
        preprocessed = expand_abbreviations(query)
        messages = [
            {"role": "system", "content": REWRITE_SYSTEM},
            {"role": "user", "content": f"Câu hỏi gốc: {preprocessed}\nTruy vấn chuẩn hóa:"},
        ]
        try:
            rewritten = self.llm.chat(messages, model=self.llm_model).strip()
            rewritten = rewritten.strip('"\'')
            return rewritten if len(rewritten) > 3 else preprocessed
        except Exception:
            return preprocessed

    def get_search_queries(self, query: str) -> list[str]:
        """Tạo danh sách các truy vấn tìm kiếm (chỉ phân tách đa thực thể khi câu hỏi hỏi nhiều CLB)."""
        preprocessed = expand_abbreviations(query)
        is_multi_club = bool(MULTI_CLUB_CHECK.search(query))

        if needs_rewrite(query):
            rewritten_text = self.rewrite_query(query)
            lines = [line.strip().strip('"\'') for line in rewritten_text.splitlines() if line.strip()]
            cleaned_lines = []
            for line in lines:
                cleaned = re.sub(r"^\s*(?:\d+[.)]|[-*•+])\s*", "", line).strip()
                if len(cleaned) > 3:
                    cleaned_lines.append(cleaned)
            if cleaned_lines:
                # CHỈ phân tách nhiều truy vấn khi câu hỏi thực sự hỏi nhiều CLB / so sánh
                if is_multi_club:
                    return cleaned_lines
                else:
                    return [cleaned_lines[0]]
        return [preprocessed]

    def rank_single_query(
        self, search_query: str, original_query: str = "", target_club: str | None = None
    ) -> np.ndarray:
        """Xếp hạng tài liệu cho một truy vấn đơn lẻ bằng RRF (k=20) kết hợp định tuyến CLB."""
        # 1. Dense retrieval (bge-m3)
        q_vec = self.embedder.embed([search_query])[0]
        dense_scores = self.doc_vectors @ q_vec
        dense_ranking = np.argsort(-dense_scores)
        dense_rank_map = {doc_idx: rank for rank, doc_idx in enumerate(dense_ranking)}

        # 2. BM25 retrieval
        bm25_scores = self.bm25.get_scores(search_query)
        if original_query and original_query != search_query:
            bm25_scores = bm25_scores + 0.5 * self.bm25.get_scores(original_query)

        matched_bm25 = np.where(bm25_scores > 0)[0]
        sorted_matched_bm25 = matched_bm25[np.argsort(-bm25_scores[matched_bm25])]
        bm25_rank_map = {doc_idx: rank for rank, doc_idx in enumerate(sorted_matched_bm25)}

        # 3. RRF kết hợp thứ hạng (k=20 tối ưu cho kho 118 tài liệu)
        k_rrf = 20
        rrf_scores = np.zeros(len(self.docs), dtype=np.float32)
        for i, doc in enumerate(self.docs):
            score = 1.0 / (k_rrf + dense_rank_map[i])
            if i in bm25_rank_map:
                score += 1.3 / (k_rrf + bm25_rank_map[i])

            # Định tuyến mềm theo CLB: nếu câu hỏi hướng tới 1 CLB cụ thể, ưu tiên tài liệu của CLB đó và S2B chung
            if target_club:
                if doc.club == target_club or doc.club == "Cộng đồng S2B":
                    score *= 1.25
                else:
                    score *= 0.85

            rrf_scores[i] = score

        return np.argsort(-rrf_scores)

    def run(self, query: str) -> RAGOutput:
        # 1. Phân tích và sinh các truy vấn tìm kiếm
        preprocessed = expand_abbreviations(query)
        queries = self.get_search_queries(query)

        # 2. Truy xuất tài liệu
        if len(queries) == 1:
            target_club = get_query_club(preprocessed) or get_query_club(queries[0])
            ranking = self.rank_single_query(
                queries[0], original_query=preprocessed, target_club=target_club
            )[: self.n_retrieved]
        else:
            # Truy vấn từng thực thể và xen kẽ kết quả (round-robin interleave)
            # Với mỗi sub-query, xác định target_club riêng của sub-query đó
            sub_rankings = [
                self.rank_single_query(sq, original_query="", target_club=get_query_club(sq))
                for sq in queries
            ]
            interleaved: list[int] = []
            seen: set[int] = set()
            max_len = max(len(r) for r in sub_rankings)
            for step in range(max_len):
                for r in sub_rankings:
                    if step < len(r):
                        doc_idx = int(r[step])
                        if doc_idx not in seen:
                            seen.add(doc_idx)
                            interleaved.append(doc_idx)
                            if len(interleaved) >= self.n_retrieved:
                                break
                if len(interleaved) >= self.n_retrieved:
                    break
            ranking = np.array(interleaved[: self.n_retrieved], dtype=int)

        docs = [self.docs[i] for i in ranking]

        # 3. Dynamic context: câu hỏi nhiều CLB cần nhiều doc (10), câu hỏi đơn 1 CLB chỉ cần top 7 để tránh nhiễu
        is_multi_club = bool(MULTI_CLUB_CHECK.search(query)) or len(queries) > 1
        effective_top_k = self.top_k if is_multi_club else min(7, self.top_k)
        context_docs = docs[:effective_top_k]

        # Ghép nối các đoạn văn làm ngữ cảnh (context) có kèm nhãn CLB và Chủ đề
        context = "\n\n".join(
            [f"[{i+1}] ({d.club} - {d.topic})\n{d.text}" for i, d in enumerate(context_docs)]
        )

        # 4. Sinh câu trả lời chuẩn mực bám sát ngữ cảnh
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": PROMPT.format(context=context, query=query)},
        ]

        answer = self.llm.chat(
            messages,
            model=self.llm_model,
        )

        return RAGOutput(answer=answer, retrieved_ids=[d.id for d in docs])
