import chromadb
from typing import List, Dict, Any
from app.services import embedding, llm_service
from app.utils import pdf_utils, chunk_utils
from app.utils.logger import get_logger
from langchain.document_loaders import PyPDFLoader


logger = get_logger(__name__)

CHROMA_PATH = "chroma_db"

client = chromadb.PersistentClient(path=CHROMA_PATH)

# Paper-level collection (abstract + metadata)
papers_collection = client.get_or_create_collection(
    name="new_academic_papers",
    metadata={"hnsw:space": "cosine"}
)

# Chunk-level collection (sections/sentences)
chunks_collection = client.get_or_create_collection(
    name="new_paper_chunks",
    metadata={"hnsw:space": "cosine"}
)


def add_paper(file_path: str, file_hash: str, filename: str) -> int:
    """
    Adds a paper into vector DB with two collections:
    - papers: meta + abstract
    - chunks: chunked text with embeddings
    """
    try:
        # ---------- Step 1: Extract metadata ----------
        meta = pdf_utils.extract_pdf_metadata(file_path)
        if isinstance(meta.get("authors"), list):
            meta["authors"] = ", ".join(meta["authors"])

        # 如果沒有 abstract，就 fallback 從 PDF 內文第一段抓
        abstract = meta.get("abstract")
        if not abstract:
            pdfloader = PyPDFLoader(file_path)  # 確保檔案可讀
            docs = pdfloader.load()
            text_first_page = docs[0].page_content
            abstract = text_first_page.split("\n\n")[1] if "\n\n" in text_first_page else text_first_page[:1000]
            meta["abstract"] = abstract

        # ---------- Step 2: Store in papers ----------
        abstract_embedding = embedding.get_embeddings([abstract])[0]
        papers_collection.add(
            ids=[file_hash],   # paper_id = file_hash
            embeddings=[abstract_embedding],
            metadatas=[{
                "paper_id": file_hash,
                "filename": filename,
                "hash": file_hash,
                **meta
            }],
            documents=[abstract]
        )

        # ---------- Step 3: Chunking ----------
        text = pdf_utils.extract_text_from_pdf(file_path)
        chunks = chunk_utils.chunk_text(text)  # [{"text":..., "chunk_id":...}, ...]
        # ---------- Step 4: Store chunks ----------
        embeddings = embedding.get_embeddings([c["text"] for c in chunks])
        chunk_ids = [f"{file_hash}_{c['chunk_id']}" for c in chunks]

        chunks_collection.add(
            ids=chunk_ids,
            embeddings=embeddings,
            metadatas=[{
                "paper_id": file_hash,
                "filename": filename,
                "chunk_id": c["chunk_id"],
                "page_start": c["page_start"],
                "page_end": c["page_end"],
            } for c in chunks],
            documents=[c["text"] for c in chunks]
        )

        logger.info(f"Added paper {filename} with {len(chunks)} chunks + abstract")
        return len(chunks)

    except Exception as e:
        logger.error(f"Error adding paper {filename}: {e}", exc_info=True)
        raise e

def delete_paper(paper_id: str) -> None:
    """Deletes a paper and its chunks from the vector database."""
    try:
        # 刪掉 paper-level
        papers_collection.delete(where={"paper_id": paper_id})
        # 刪掉 chunk-level
        chunks_collection.delete(where={"paper_id": paper_id})

        logger.info(f"Deleted paper and chunks with paper_id {paper_id}")
    except Exception as e:
        logger.error(f"Error deleting paper {paper_id}: {e}", exc_info=True)
        raise


def list_papers() -> List[Dict[str, Any]]:
    """Lists all papers in the database (from papers_collection)."""
    try:
        results = papers_collection.get(include=["metadatas"])
        all_metadatas = results.get("metadatas", [])

        papers = []
        for meta in all_metadatas:
            logger.info(f"Paper metadata: {meta}")  # debug
            paper = {
                "hash": meta.get("hash"),
                "filename": meta.get("filename"),
                "title": meta.get("title"),
                "authors": meta.get("authors"),
                "year": meta.get("year"),
                "doi": meta.get("doi"),
                "arxiv_id": meta.get("arxiv_id"),
                "status": "completed",
            }
            # authors 格式清理
            if isinstance(paper["authors"], list):
                paper["authors"] = ", ".join(paper["authors"])
            papers.append(paper)

        return papers
    except Exception as e:
        logger.error(f"Error listing papers: {e}", exc_info=True)
        raise e

def search(query: str, 
           k: int, 
           model: str, 
           provider: str, 
           api_key: str, 
           prompt_mode: str, 
           custom_prompt: str = "") -> Dict[str, Any]:
    """
    RAG 搜尋與生成回應
    - provider: Local | OpenAI | Google | Anthropic
    - 二階段檢索：先 abstract，再 chunk
    """
    try:
        # --- Query rewrite (if too short) ---
        # if len(query.split()) <= 2:
        #     try:
        #         rewrite_prompt = f"""Rewrite the query '{query}' for better document search.
        #         Keep it short and focused.
        #         Preserve the original action (combine/use/integrate) without changing the meaning.
        #         Do NOT turn it into general comparisons or analyses.
        #         """

        #         query = generate_answer(provider, model, api_key, rewrite_prompt)
        #         logger.info(f"Rewritten query: {query}")
        #     except Exception as e:
        #         logger.warning(f"Query rewrite failed: {e}. Using original query.")

        # --- Step 1: 對 query 做 embedding ---
        query_embedding = embedding.get_embeddings([query])[0]

        # --- Step 2: 在 paper-level (abstract) 搜索 ---
        paper_results = papers_collection.query(
            query_embeddings=[query_embedding],
            n_results=min(5, k),   # 先縮小候選集合
            include=["documents", "metadatas", "distances"]
        )

        # 取回候選 paper_id
        candidate_papers = []
        for meta, dist in zip(paper_results["metadatas"][0], paper_results["distances"][0]):
            similarity = 1 - dist
            candidate_papers.append((meta.get("hash"), similarity))

        if not candidate_papers:
            return {"query": query, "sources": [], "answer": "No relevant papers found."}

        # --- Step 3: 在 chunks_collection 搜索 ---
        results = chunks_collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where={"paper_id": {"$in": [pid for pid, _ in candidate_papers]}},
            include=["documents", "metadatas", "distances"]
        )

        chunks = results["documents"][0]
        chunk_matas = results["metadatas"][0]
        metas = paper_results["metadatas"][0]
        scores = results["distances"][0]

        # --- Step 4: sources 結構化 ---
        sources = []
        for chunk, chunk_mata, meta, score in zip(chunks, chunk_matas, metas, scores):
            similarity = 1 - score  # cosine distance → similarity
            sources.append({
                "chunk_text": chunk,
                "chunk_id": chunk_mata.get("chunk_id", ""),
                "paper_id": meta.get("paper_id", ""),
                "paper_title": meta.get("title", ""),
                "paper_year": meta.get("year", ""),
                "page_start": chunk_mata.get("page_start", ""),
                "page_end": chunk_mata.get("page_end", ""),
                "score": round(similarity * 100, 2)
            })

        # --- Step 5: 組合 context ---
        context = "\n\n".join(
            [f"{s['chunk_text']}" for s in sources]
        )

        if prompt_mode == "custom":
            prompt = custom_prompt.format(context=context, query=query)
        else:
            prompts = {
                "summary": f"""
                    You are an academic research assistant.
                    Summarize the following text **only in relation to the query**: "{query}".

                    Guidelines:
                    - Focus strictly on information that answers the query.
                    - Highlight not just *what* the authors say, but also *why it matters* (motivation, implications).
                    - Ignore irrelevant details.
                    - If the context does not contain enough information, say so explicitly.

                    Context:
                    {context}
                """,
                "tech": f"""
                    You are a technical expert.
                    Provide a detailed, structured explanation to answer the query: "{query}".

                    Guidelines:
                    - Use the provided context as your primary source.
                    - Start with a concise **direct answer**.
                    - Then break down the explanation into sections (e.g., Definitions, Methodology, Results, Implications).
                    - Include equations, numbers, or examples if they appear in the context.
                    - If the context is insufficient, acknowledge the gap instead of inventing details.

                    Context:
                    {context}
                """,
                "citation": f"""
                    You are an academic citation assistant.
                    Find the most relevant citations from the text to support the query: "{query}".

                    Guidelines:
                    - Extract direct references, author names, or publication details exactly as given.
                    - Present results in the format: [Author, Year] or closest possible.
                    - If no clear citation exists, state: "No relevant citation found in the context."

                    Context:
                    {context}
                """
            }
            prompt = prompts.get(
                prompt_mode,
                f"Answer the question '{query}' using this context:\n{context}"
            )

        logger.info(f"Final prompt to LLM: {prompt}")
        answer = generate_answer(provider, model, api_key, prompt)

        return {
            "query": query,
            "sources": sources,
            "answer": answer,
            "model_used": model,
            "prompt_mode": prompt_mode
        }

    except Exception as e:
        logger.error(f"Error during search: {e}", exc_info=True)
        raise e


def generate_answer(provider: str, model: str, api_key: str, prompt: str) -> str:
    # --- Step 6: 呼叫 LLM ---
    if provider == "Local":
        answer = llm_service.generate_completion_local(model, prompt)
    elif provider == "OpenAI":
        answer = llm_service.generate_completion_openai(model, api_key, prompt)
    elif provider == "Google":
        answer = llm_service.generate_completion_gemini(model, api_key, prompt)
    elif provider == "Anthropic":
        answer = llm_service.generate_completion_claude(model, api_key, prompt)
    else:
        logger.error(f"Unsupported provider: {provider}")
        raise ValueError(f"Unsupported provider: {provider}")
    return answer


def clear_all_data():
    """
    Clears all data from both collections by deleting and recreating them.
    """
    global papers_collection, chunks_collection
    try:
        client.delete_collection(name="new_academic_papers")
        client.delete_collection(name="new_paper_chunks")
        
        papers_collection = client.get_or_create_collection(
            name="new_academic_papers",
            metadata={"hnsw:space": "cosine"}
        )
        
        chunks_collection = client.get_or_create_collection(
            name="new_paper_chunks",
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info("Cleared and recreated all collections.")
    except Exception as e:
        logger.error(f"Error clearing all data: {e}", exc_info=True)
        # If collections don't exist, deletion will fail. Try to create them anyway.
        try:
            papers_collection = client.get_or_create_collection(
                name="new_academic_papers",
                metadata={"hnsw:space": "cosine"}
            )
            chunks_collection = client.get_or_create_collection(
                name="new_paper_chunks",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Recreated collections after deletion error.")
        except Exception as create_e:
            logger.error(f"Failed to recreate collections after error: {create_e}", exc_info=True)
            raise create_e
