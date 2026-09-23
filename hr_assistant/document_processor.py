import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from hr_assistant.config import Config

class DocumentProcessor:
    @staticmethod
    def load_and_chunk_documents(db):
        if not os.path.exists(Config.RESUMES_DIR):
            os.makedirs(Config.RESUMES_DIR, exist_ok=True)
            return

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            length_function=len
        )

        all_chunks = []
        all_metadatas = []
        all_ids = []

        for file_name in os.listdir(Config.RESUMES_DIR):
            if file_name.startswith('.'):
                continue
                
            file_path = os.path.join(Config.RESUMES_DIR, file_name)
            if os.path.isfile(file_path):
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                chunks = text_splitter.split_text(content)

                for idx, chunk in enumerate(chunks):
                    all_chunks.append(chunk)
                    all_metadatas.append({"source": file_name, "chunk_index": idx})
                    all_ids.append(f"{file_name}_chunk_{idx}")

        if all_ids:
            db.collection.upsert(
                documents=all_chunks,
                metadatas=all_metadatas,
                ids=all_ids
            )
            print(f"Indicizzazione completata: generati {len(all_ids)} chunk da {len(os.listdir(Config.RESUMES_DIR))} file.")
