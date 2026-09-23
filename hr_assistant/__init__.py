import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import chainlit as cl
from openai import OpenAI
from hr_assistant.database import Database
from hr_assistant.document_processor import DocumentProcessor
from hr_assistant.config import Config

db = Database()
DocumentProcessor.load_and_chunk_documents(db)

@cl.on_chat_start
async def start():
    total_chunks = db.collection.count()
    await cl.Message(
        content=f"👋 **Assistente HR**\n\nFai una domanda per esaminare i dettagli dei candidati."
    ).send()

@cl.on_message
async def main(message: cl.Message):
    results = db.collection.query(
        query_texts=[message.content],
        n_results=4
    )
    
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    
    if docs:
        context_blocks = []
        for doc, meta in zip(docs, metas):
            source = meta.get("source", "Documento Sconosciuto")
            context_blocks.append(f"[Fonte: {source}]\n{doc}")
        context = "\n\n---\n\n".join(context_blocks)
    else:
        context = "Nessun dato trovato nei CV."

    prompt = f"""Sei un collaboratore HR esperto. Rispondi alla richiesta dell'utente in modo naturale, fluido e professionale, basandoti ESCLUSIVAMENTE sui dati reali contenuti nel contesto sottostante.

GUIDA PER LA RISPOSTA:
1. Esprimi il concetto con un discorso continuo e ben strutturato (evita elenchi puntati o schemi rigidi).
2. Menzione chiaramente il nome del candidato o il file di provenienza (es. `cv_mario_rossi.txt`).
3. Cita solo ed esclusivamente dati concreti: aziende, periodi lavorativi, ruoli e tecnologie esplicitamente menzionati.
4. Non aggiungere teorie, commenti generici o supposizioni personali.

CONTESTO dai CV:
{context}

RICHIESTA UTENTE:
{message.content}
"""

    client = OpenAI(api_key=Config.OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=Config.LLM_MODEL,
        messages=[
            {"role": "system", "content": "Sei un assistente HR professionale. Rispondi in modo discorsivo, fluido e rigorosamente basato sui fatti del contesto."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1  
    )
    
    await cl.Message(content=response.choices[0].message.content).send()