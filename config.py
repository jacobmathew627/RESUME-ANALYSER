import os
import faiss
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# ✅ Load environment variables
load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# ✅ Configure Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)

# ✅ Load Sentence Transformer Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Initialize FAISS Index
EMBEDDING_DIM = 384  # Size of embeddings from all-MiniLM-L6-v2
faiss_index = faiss.IndexFlatL2(EMBEDDING_DIM)
document_store = []  # Stores text data

def add_to_faiss(text):
    """Adds text embeddings to FAISS index."""
    embedding = embedding_model.encode(text).astype("float32").reshape(1, -1)
    faiss_index.add(embedding)
    document_store.append(text)
