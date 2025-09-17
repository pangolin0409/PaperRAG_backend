from fastapi import APIRouter, UploadFile, File, HTTPException
import os
import shutil
from app.services import rag_service
from app.utils import hash_utils
from app.utils.logger import get_logger

router = APIRouter()
PAPERS_DIR = "papers/"
logger = get_logger(__name__)

@router.post("/upload")
def upload_papers(files: list[UploadFile] = File(...)):
    results = []
    for file in files:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file name provided")

        file_path = os.path.join(PAPERS_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_hash = hash_utils.get_file_hash(file_path)

        existing_papers = rag_service.list_papers()
        if any(p['hash'] == file_hash for p in existing_papers):
            os.remove(file_path)
            raise HTTPException(status_code=409, detail=f"Paper with hash {file_hash} already exists.")

        try:
            num_chunks = rag_service.add_paper(file_path, file_hash, file.filename)
            results.append({
                "filename": file.filename,
                "hash": file_hash,
                "chunks": num_chunks,
                "status": "success"
            })
        except Exception as e:
            os.remove(file_path)
            logger.error(f"Error processing file {file.filename}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to process and add paper: {e}")

    return {"uploaded": results}

@router.get("/list_papers")
def list_papers():
    return {"papers": rag_service.list_papers()}

@router.delete("/{paper_hash}")
def delete_paper(paper_hash: str):
    try:
        rag_service.delete_paper(paper_hash)
        # Also delete the file from the papers directory
        # Note: This requires finding the filename from the hash.
        # This is a simplification.
        return {"status": "success", "message": f"Paper with hash {paper_hash} deleted."}
    except Exception as e:
        print(f"Delete failed: {e}")  # debug
        raise HTTPException(status_code=500, detail=f"Failed to delete paper: {e}")
