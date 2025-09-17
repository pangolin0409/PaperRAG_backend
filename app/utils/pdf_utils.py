from langchain.document_loaders import PyPDFLoader
import re
from habanero import Crossref
import arxiv
import unicodedata

def extract_text_from_pdf(path_to_pdf: str) -> str:
    """Extract and clean text from PDF using PyMuPDF (fitz)."""
    loader = PyPDFLoader(path_to_pdf)
    docs = loader.load()
    results = []
    for page in docs:
        text = page.page_content
        # --- Step 1: 去除換行造成的斷詞 & 過多換行 ---
        text = re.sub(r'-\s*\n\s*', '', text)  # 把 "deep-\nfake" → "deepfake"
        text = re.sub(r'\n+', ' ', text)       # 多個換行 → 空格

        # --- Step 2: 移除頁眉/頁腳噪音 ---
        text = re.sub(r'arXiv:\d+\.\d+(v\d+)?', ' ', text)
        text = re.sub(r'Page \d+/\d+', ' ', text)
        text = re.sub(r'©.*?(\d{4})', ' ', text)

        # --- Step 3: Unicode 正規化 (全形→半形, 引號統一等) ---
        text = unicodedata.normalize("NFKC", text)

        # --- Step 4: 刪掉孤立數字 (多半是頁碼/圖號) ---
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)

        # --- Step 5: (選擇性) 小寫化 ---
        # 👉 建議只對 query lowercase，這裡我保留原始大小寫
        # text = text.lower()

        # --- Step 6: 符號清理 (去掉沒語意的分隔線) ---
        text = re.sub(r'[-=*_]{3,}', ' ', text)

        # --- 其他專用 pattern (數學符號/公式/引用/圖表) ---
        patterns = [
            r'\b\d+/\d+\b',                 # 分數 (12/34)
            r'\b\d+\^\d+\b',                # 次方 (2^10)
            r'\b(?:sin|cos|tan|log|ln|exp)\b',  # 常見函數
            r'\[[0-9,\-\s]+\]',             # 引用 [12], [1-3]
            r'\bFig\.?\s*\d+|\bTable\s*\d+' # 圖表
            r'\(\d+\)',
        ]
        # (A) 去掉整段公式樣式
        formula_patterns = [
            r'\$.*?\$',                # LaTeX inline formula: $...$
            r'\$\$.*?\$\$',            # LaTeX block formula: $$...$$
            r'\\\(.+?\\\)',            # \( ... \)
            r'\\\[.+?\\\]',            # \[ ... \]
            r'[A-Za-z0-9\s]*=[^,.;]+', # 等號後的長表達式
        ]

        # (B) 去掉數學專用符號
        symbol_patterns = [
            r'[≈≥≤±⊗∑∫∂∞∇]',           # 常見特殊符號
            r'[α-ωΑ-Ω]',               # 希臘字母
            r'[\^_][{]?[A-Za-z0-9]+[}]?', # 上標/下標
        ]

        # (C) 去掉純數字公式
        numeric_patterns = [
            r'\d+\s*[\+\-\*/^]\s*\d+',   # 2 + 2, 3*5, 2^10
            r'\d+\.\d+\s*%',             # 百分比 30.4%
            r'∥[^∥]+∥',
            r'ˆ\s*\w*',
            r'\s*[,.:;]\s*[,.:;]+',
            r'^[,.:;]\s*|\s*[,.:;]$',
        ]
        for p in patterns+formula_patterns+symbol_patterns+numeric_patterns:
            text = re.sub(p, " ", text)

        # --- 最後: 移除多餘空白 ---
        text = re.sub(r'\s+', ' ', text).strip()
        results.append({'text': text, 'page': page.metadata.get('page', -1)})
    return results

# ---------- STEP 1: DOI ----------
def fetch_metadata_from_doi(doi: str) -> dict:
    cr = Crossref()
    try:
        result = cr.works(ids=doi)
        msg = result.get("message", {})
        title = msg.get("title", [None])[0]
        authors = [f"{a.get('given', '')} {a.get('family', '')}".strip()
                   for a in msg.get("author", [])]
        year = None
        if "published-print" in msg:
            year = msg["published-print"]["date-parts"][0][0]
        elif "issued" in msg:
            year = msg["issued"]["date-parts"][0][0]

        abstract = msg.get("abstract")
        if abstract:
            # CrossRef abstract 可能帶有 <jats:p> 標籤，要清掉
            abstract = re.sub(r"<[^>]+>", "", abstract).strip()

        return {
            "doi": doi,
            "title": title,
            "authors": authors,
            "year": year,
            "abstract": abstract,
            "source": "crossref"
        }
    except Exception as e:
        return {"doi": doi, "source": "crossref_error", "error": str(e)}

# ---------- STEP 2: arXiv ----------
def fetch_metadata_from_arxiv(arxiv_id: str) -> dict:
    try:
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(search.results())
        return {
            "arxiv_id": arxiv_id,
            "title": paper.title,
            "authors": [str(a) for a in paper.authors],
            "year": paper.published.year,
            "abstract": paper.summary.strip(),
            "source": "arxiv"
        }
    except Exception as e:
        return {"arxiv_id": arxiv_id, "source": "arxiv_error", "error": str(e)}

# ---------- STEP 3: Heuristic fallback ----------
def extract_basic_metadata_from_pdf(file_path: str) -> dict:
    loader = PyPDFLoader(file_path)  # 確保檔案可讀
    docs = loader.load()
    first_page_text = docs[0].page_content

    # 嘗試抓 Title: 假設第一行是 title
    lines = [line.strip() for line in first_page_text.split("\n") if line.strip()]
    title = lines[0] if lines else None

    # 嘗試抓 Authors: 假設第二、三行內含 email 或大寫縮寫
    authors = None
    for line in lines[1:5]:
        if "@" in line or re.search(r"[A-Z]\.\s*[A-Z]", line):
            authors = line
            break

    # 嘗試抓年份: 從 References 或頁腳
    year_match = re.search(r"(19|20)\d{2}", first_page_text)
    year = int(year_match.group(0)) if year_match else None

    # 嘗試抓 Abstract: 找 "Abstract" 開頭段落
    abstract_match = re.search(r"(?i)abstract[:\s]*(.+?)(?=\n\s*[1I]\.|\n\s*Keywords|\n\s*Index|\Z)", 
                               first_page_text, re.S)
    abstract = abstract_match.group(1).strip() if abstract_match else None

    return {
        "title": title,
        "authors": authors,
        "year": year,
        "abstract": abstract,
        "source": "pdf_heuristic"
    }

# ---------- MASTER FUNCTION ----------
def extract_pdf_metadata(file_path: str) -> dict:
    loader = PyPDFLoader(file_path)  # 確保檔案可讀
    docs = loader.load()

    first_page_text = docs[0].page_content

    # Step 1: Try DOI
    doi_match = re.search(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", first_page_text, re.I)
    if doi_match:
        doi = doi_match.group(0)
        return fetch_metadata_from_doi(doi)

    # Step 2: Try arXiv ID
    arxiv_match = re.search(r"arXiv:\d{4}\.\d{4,5}(v\d+)?", first_page_text)
    if arxiv_match:
        arxiv_id = arxiv_match.group(0).replace("arXiv:", "")
        return fetch_metadata_from_arxiv(arxiv_id)

    # Step 3: Fallback Heuristic
    return extract_basic_metadata_from_pdf(file_path)
