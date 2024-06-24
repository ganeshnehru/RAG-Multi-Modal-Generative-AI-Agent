import numpy as np
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
import os

class EmbeddingModels:
    def __init__(self):
        # Initialize Embedding Models
        self.document_embedder = NVIDIAEmbeddings(model="ai-embed-qa-4", model_type="passage")
        self.query_embedder = NVIDIAEmbeddings(model="ai-embed-qa-4", model_type="query")

        # Ensure directories exist
        self.DOCS_DIR = os.path.abspath("./uploaded_docs")
        self.EMBEDDINGS_DIR = os.path.abspath("./embeddings")
        if not os.path.exists(self.DOCS_DIR):
            os.makedirs(self.DOCS_DIR)
        if not os.path.exists(self.EMBEDDINGS_DIR):
            os.makedirs(self.EMBEDDINGS_DIR)

    def embed_documents(self, documents):
        """Generates embeddings for a list of documents."""
        return self.document_embedder.embed_documents(documents)

    def embed_query(self, query):
        """Generates an embedding for a single query."""
        return self.query_embedder.embed_query(query)

    def save_embedding(self, file_name, embedding):
        """Saves an embedding to a file."""
        embedding_path = os.path.join(self.EMBEDDINGS_DIR, f"{file_name}.npy")
        np.save(embedding_path, embedding)

    def load_embeddings(self):
        """Loads all stored document embeddings."""
        embeddings_files = [os.path.join(self.EMBEDDINGS_DIR, f) for f in os.listdir(self.EMBEDDINGS_DIR)]
        document_embeddings = [(file, np.load(file)) for file in embeddings_files]
        return document_embeddings

    def compare_embeddings(self, query_embedding, document_embeddings):
        """Compares a query embedding with stored document embeddings."""
        similarities = [(file, np.dot(query_embedding, doc_emb.T)) for file, doc_emb in document_embeddings]
        similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
        return similarities
