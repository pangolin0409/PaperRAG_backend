from fastapi import APIRouter, BackgroundTasks
from app.services import rag_service
from app.services.embedding import EMBEDDING_MODEL
import os

router = APIRouter()
PAPERS_DIR = "papers/"

@router.get("/stats")
def get_database_stats():
    """
    Gets statistics for the vector database.
    """
    try:
        total_papers = rag_service.papers_collection.count()
        total_chunks = rag_service.chunks_collection.count()
        
        status = "ready"
        if total_papers == 0 and total_chunks == 0:
            status = "empty"
        
        return {
            "total_papers": total_papers,
            "total_chunks": total_chunks,
            "embedding_model": EMBEDDING_MODEL,
            "index_status": status,
        }

    except Exception as e:
        # Log the error and return an appropriate HTTP response
        return {"error": str(e)}

def rebuild_db():
    """
    Background task to rebuild the database.
    """
    rag_service.clear_all_data()

@router.post("/rebuild")
def rebuild_database(background_tasks: BackgroundTasks):
    """
    Triggers a complete rebuild of the vector database.
    """
    background_tasks.add_task(rebuild_db)
    return {"status": "success", "message": "Rebuilding started"}
