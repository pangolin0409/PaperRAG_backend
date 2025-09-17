# Paper RAG

## 專案概述 (Project Overview)

這個專案旨在解決學術研究者、學生或任何需要從下載過的論文中快速找印象中論文片段。

本專案透過 RAG (Retrieval-Augmented Generation) 技術，讓使用者能用自然語言與上傳的 PDF 文件進行互動式問答。使用者不再需要通讀全文，而是可以直接提問，系統會根據使用的問題搜尋最高度相關的論文片段，並產生出對應的答案。

## 軟體架構

*   **核心架構設計與開發**：使用 **Python** 與 **FastAPI** 框架，設計並實作了整個後端服務的 RESTful API，確保系統具備高擴展性與穩定性。
*   **RAG 流程建構**：
    *   整合 **ChromaDB** 作為向量資料庫，負責儲存論文內容的向量化表示 (Embeddings)。
    *   開發文件處理模組，實現 PDF 內容解析、文本切割 (Chunking) 與向量化，並確保文件來源的唯一性 (透過 Hash)。
    *   串接 **{您使用的 LLM，例如 OpenAI GPT-4, Google Gemini}**，實現高品質的自然語言生成，為使用者的提問提供精準答案。
*   **API 端點開發**：負責開發 `/upload`、`/search` 等核心 API，實現文件上傳、處理、以及與論文的即時問答功能。
*   **系統優化**：設計了非同步處理流程來處理耗時的文件解析與向量化任務，避免 API 請求超時，提升使用者體驗。

## 技術棧 (Tech Stack)

| 類別 | 技術/工具 |
| :--- | :--- |
| **後端 (Backend)** | Python, FastAPI |
| **資料庫 (Database)** | ChromaDB (向量資料庫), SQLite |
| **AI / LLM** | {您使用的 LLM，例如 OpenAI, Gemini}, LangChain |
| **核心演算法** | Retrieval-Augmented Generation (RAG) |
| **工具 (Tools)** | Git, Docker, Uvicorn |


## 安裝與使用方式 (Installation & Usage)

請依照以下步驟在本地端運行此專案：

1.  **Clone 專案**
    ```bash
    git clone {您的專案 Git URL}
    cd {專案名稱}
    ```

2.  **安裝依賴套件**
    ```bash
    pip install -r requirements.txt
    ```

3.  **設定環境變數**
    *   建立一個 `.env` 檔案。
    *   在檔案中加入您的 LLM API 金鑰，例如：
        ```
        OPENAI_API_KEY="sk-..."
        ```

4.  **啟動應用程式**
    ```bash
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    ```

5.  **使用 API**
    *   **上傳論文**: 使用 `POST /papers/upload` 端點並附上 PDF 檔案。
    *   **進行問答**: 使用 `POST /papers/{paper_id}/search` 端點並在請求主體中附上您的問題。


## 未來改進與學習 (Future Improvements & Lessons Learned)

*   **未來改進方向**:
    *   **支援多文件問答**：讓使用者能同時上傳多篇論文，進行跨文件的整合性提問。
    *   **引入快取機制**：對於常見問題或已處理過的區塊，引入快取以加速回應。
    *   **前端介面開發**：建立一個簡單的前端介面，讓非技術人員也能輕鬆使用。

*   **最大的收穫**:
    *   透過這個專案，我深入學習了 RAG 的完整工作流程，從資料處理到模型應用都有了第一手經驗。
    *   我更加熟練地運用 FastAPI 的進階功能（如背景任務）來解決真實世界的工程問題。
    *   我體會到 Prompt Engineering 在控制 LLM 行為上的重要性，並學習到如何設計有效的提示詞。
