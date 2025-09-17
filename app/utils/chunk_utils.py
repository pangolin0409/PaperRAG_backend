from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_text(pages: List[Dict], chunk_size: int = 700, chunk_overlap: int = 100) -> List[Dict]:
    """
    - 先合併全文再切，確保語意完整
    - 利用頁碼位置回填每個 chunk 的來源範圍
    pages: [{"text": "page 1 content", "page": 1}, {"text": "page 2 content", "page": 2}, ...]
    """
    # --- Step 1: 建立全文與頁碼 index ---
    full_text = ""
    page_ranges = []  # [(start_char, end_char, page_number)]
    cursor = 0

    for p in pages:
        text = p["text"]
        start = cursor
        end = cursor + len(text)
        page_ranges.append((start, end, p["page"]))
        full_text += text + "\n"
        cursor = end + 1  # +1 for the added newline

    # --- Step 2: chunking ---
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    docs = splitter.create_documents([full_text])

    # --- Step 3: 回填頁碼範圍 ---
    chunks = []
    cursor = 0
    for i, doc in enumerate(docs):
        text = doc.page_content
        start = cursor
        end = cursor + len(text)
        cursor = end

        # 找出 chunk 對應的頁碼範圍
        pages_in_chunk = [r[2] for r in page_ranges if not (r[1] < start or r[0] > end)]

        if pages_in_chunk:
            page_start, page_end = min(pages_in_chunk), max(pages_in_chunk)
        else:
            page_start, page_end =-1, -1  # 無法對應頁碼

        chunks.append({
            "text": text,
            "chunk_id": i,
            "page_start": page_start,
            "page_end": page_end
        })

    return chunks
