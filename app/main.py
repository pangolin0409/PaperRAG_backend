from fastapi import FastAPI
from app.routes import papers, search, models, prompts, database
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Paper RAG Backend",
    description="A FastAPI backend for a research paper assistant.",
    version="0.1.0",
    debug=True
)

# 允許的來源（開發時可直接放 *，但正式建議指定 domain）
origins = [
    "http://localhost:5173",   # Vite (React dev)
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        # 允許的來源
    allow_credentials=True,
    allow_methods=["*"],          # 允許所有 HTTP 方法 (GET, POST, DELETE...)
    allow_headers=["*"],          # 允許所有自訂 headers
)

# Include routers
app.include_router(papers.router, prefix="/papers", tags=["Paper Management"])
app.include_router(search.router, prefix="/search", tags=["Search & QA"])
app.include_router(models.router, prefix="/models", tags=["Model Management"])
app.include_router(prompts.router, prefix="/prompts", tags=["Prompt Management"])
app.include_router(database.router, prefix="/database", tags=["Database Management"])

@app.get("/", tags=["Health Check"])
def read_root():
    return {"status": "ok"}
