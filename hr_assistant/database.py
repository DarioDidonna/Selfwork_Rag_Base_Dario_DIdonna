import os
import chromadb
from chromadb.utils import embedding_functions
from hr_assistant.config import Config

class Database:
    def __init__(self):
        os.makedirs(Config.CHROMA_PERSIST_DIR, exist_ok=True)
        self.client = chromadb.PersistentClient(path=Config.CHROMA_PERSIST_DIR)
        
        self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=Config.OPENAI_API_KEY,
            model_name=Config.EMBEDDING_MODEL
        )
        
        self.collection = self.client.get_or_create_collection(
            name="resumes_rag_base",
            embedding_function=self.embedding_fn
        )
