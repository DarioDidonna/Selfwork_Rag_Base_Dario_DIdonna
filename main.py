import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from openai import OpenAI

# Carica le variabili d'ambiente dal file .env
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

if not openai_key:
    raise ValueError("OPENAI_API_KEY non trovata. Verifica il file .env")

# Configurazione della funzione di embedding di OpenAI
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_key, model_name="text-embedding-3-small"
)

# Inizializzazione del client ChromaDB
chroma_client = chromadb.Client()

collection = chroma_client.get_or_create_collection(
    name="CVs", embedding_function=openai_ef
)

# Dati dei candidati
documents = [
    (
        "Esperto in Digital Marketing e Social Media Strategy. Gestisce"
        " campagne pubblicitarie, promozione di prodotti e brand awareness."
    ),
    (
        "Sviluppatore Full Stack specializzato in Python, Django e React."
        " Creazione di architetture web avanzate e database."
    ),
    (
        "Graphic Designer e Content Creator. Specializzato in brand identity,"
        " creazione di contenuti visivi e materiale promozionale."
    ),
]

metadatas = [
    {"source": "CV_Esperto_Marketing.txt"},
    {"source": "CV_Sviluppatore_Web.txt"},
    {"source": "CV_Graphic_Designer.txt"},
]

ids = ["id_1", "id_2", "id_3"]

# Inserimento dei documenti nel DB vettoriale
collection.add(documents=documents, metadatas=metadatas, ids=ids)

# Query utente
user_question = "mi serve qualcuno per promuovere il mio prodotto"

results = collection.query(query_texts=[user_question], n_results=1)

# Estrazione del contesto recuperato
source_file = results["metadatas"][0][0]["source"]
matched_text = results["documents"][0][0]

context = f"CONTESTO: nome file {source_file} ecco il paragrafo più significativo: {matched_text}"

prompt = f"""Dato il seguente contesto:
{context}

Rispondi alla domanda dell'utente: "{user_question}"
Spiega che nel file individuato c'è il profilo più adatto.
Argomenta la scelta utilizzando il contenuto del testo individuato nel contesto."""

# Inizializzazione del client OpenAI ed esecuzione della completion
client = OpenAI(api_key=openai_key)

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "developer",
            "content": (
                "Sei un assistente HR, specializzato nella ricerca di profili"
                " professionali."
            ),
        },
        {"role": "user", "content": prompt},
    ],
)

print(completion.choices[0].message.content)