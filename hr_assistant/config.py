import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    raw_key = os.getenv("OPENAI_API_KEY", "")
    OPENAI_API_KEY = raw_key.strip() if raw_key else None
    
    EMBEDDING_MODEL = "text-embedding-3-small"
    LLM_MODEL = "gpt-4o-mini"
    
    CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chroma")
    RESUMES_DIR = os.path.join(os.path.dirname(__file__), "..", "resumes")

if Config.OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = Config.OPENAI_API_KEY
