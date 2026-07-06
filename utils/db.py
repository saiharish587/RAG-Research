import os
import faiss
import numpy as np
import pdfplumber
import logging
from sentence_transformers import SentenceTransformer

# Suppress verbose pypdf warnings
logging.getLogger("pypdf").setLevel(logging.ERROR)
class RecursiveCharacterTextSplitter:
    def __init__(self, chunk_size=500, chunk_overlap=50, length_function=len):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text):
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if end < len(text):
                boundary = -1
                for char in ['\n', '.', ' ']:
                    pos = text.rfind(char, start, end)
                    if pos != -1 and pos > start + (self.chunk_size // 2):
                        boundary = pos + 1
                        break
                if boundary != -1:
                    end = boundary
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
                
            if end >= len(text):
                break
                
            new_start = end - self.chunk_overlap
            if new_start <= start:
                new_start = end
            start = new_start
            
            if start >= len(text) or self.chunk_size <= self.chunk_overlap:
                break
        return chunks

class VectorDBManager:
    def __init__(self, embedding_model_name="BAAI/bge-small-en-v1.5", device="cpu"):
        print(f"Loading embedding model: {embedding_model_name} on {device}...")
        self.model = SentenceTransformer(embedding_model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.doc_chunks = []  # Stores the actual text chunks corresponding to vector IDs

    def load_documents(self, documents_dir):
        """Loads and extracts text recursively from TXT and PDF documents in a directory."""
        documents = []
        if not os.path.exists(documents_dir):
            print(f"Documents directory '{documents_dir}' does not exist.")
            return documents
        
        print(f"Scanning directory '{documents_dir}' recursively...")
        for root, dirs, files in os.walk(documents_dir):
            for file in files:
                file_path = os.path.join(root, file)
                rel_source = os.path.relpath(file_path, documents_dir)
                
                if file.endswith(".txt"):
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            documents.append({"text": f.read(), "source": rel_source})
                    except Exception as e:
                        print(f"Error reading TXT {file_path}: {e}")
                elif file.endswith(".pdf"):
                    try:
                        import pypdf
                        text_content = []
                        reader = pypdf.PdfReader(file_path)
                        for page in reader.pages:
                            text = page.extract_text()
                            if text:
                                text_content.append(text)
                        documents.append({"text": "\n".join(text_content), "source": rel_source})
                    except Exception as e:
                        print(f"Error reading PDF {file_path} with pypdf: {e}. Falling back to pdfplumber...")
                        try:
                            text_content = []
                            with pdfplumber.open(file_path) as pdf:
                                for page in pdf.pages:
                                    text = page.extract_text()
                                    if text:
                                        text_content.append(text)
                            documents.append({"text": "\n".join(text_content), "source": rel_source})
                        except Exception as e2:
                            print(f"Error reading PDF {file_path} with pdfplumber fallback: {e2}")
        print(f"Successfully loaded {len(documents)} documents from '{documents_dir}'.")
        return documents

    def chunk_documents(self, documents, chunk_size=500, chunk_overlap=50):
        """Chunks documents using RecursiveCharacterTextSplitter."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        chunks = []
        for doc in documents:
            split_texts = splitter.split_text(doc["text"])
            for idx, text in enumerate(split_texts):
                chunks.append({
                    "text": text,
                    "metadata": {"source": doc["source"], "chunk_id": idx}
                })
        return chunks

    def build_index(self, chunks):
        """Generates embeddings and builds a FAISS index."""
        if not chunks:
            print("No chunks provided to build vector index.")
            return
        
        valid_chunks = []
        texts = []
        for c in chunks:
            t = c.get("text")
            if isinstance(t, str) and len(t.strip()) > 0:
                # Strip Unicode surrogate characters that crash Rust tokenizers (U+D800 to U+DFFF)
                t_clean = "".join(char for char in t if not (0xD800 <= ord(char) <= 0xDFFF))
                if len(t_clean.strip()) > 0:
                    c["text"] = t_clean
                    valid_chunks.append(c)
                    texts.append(t_clean)
                
        self.doc_chunks = valid_chunks
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True, batch_size=256)
        
        # Build IndexFlatIP (Inner Product/Cosine Similarity after normalization) or IndexFlatL2
        # We will use IndexFlatIP for Cosine/Dot Product similarity
        # First normalize embeddings
        faiss.normalize_L2(embeddings)
        
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)
        print("FAISS Index built successfully.")

    def save_index(self, folder_path="vector_db/faiss_index"):
        """Saves the FAISS index and the corresponding document chunks to disk."""
        os.makedirs(folder_path, exist_ok=True)
        if self.index is None:
            print("No index to save.")
            return
        
        faiss_file = os.path.join(folder_path, "index.faiss")
        chunks_file = os.path.join(folder_path, "chunks.npy")
        
        faiss.write_index(self.index, faiss_file)
        np.save(chunks_file, self.doc_chunks, allow_pickle=True)
        print(f"Index successfully saved to {folder_path}")

    def load_index(self, folder_path="vector_db/faiss_index"):
        """Loads the FAISS index and document chunks from disk."""
        faiss_file = os.path.join(folder_path, "index.faiss")
        chunks_file = os.path.join(folder_path, "chunks.npy")
        
        if not os.path.exists(faiss_file) or not os.path.exists(chunks_file):
            print(f"Failed to load index: files not found in {folder_path}")
            return False
        
        self.index = faiss.read_index(faiss_file)
        self.doc_chunks = np.load(chunks_file, allow_pickle=True).tolist()
        print(f"Index loaded from {folder_path} with {len(self.doc_chunks)} chunks.")
        return True

    def search(self, query, top_k=3):
        """Performs a semantic search on the index and returns matching chunks and scores."""
        if self.index is None:
            print("Search error: FAISS Index not loaded or built.")
            return []
        
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.doc_chunks):
                results.append({
                    "chunk": self.doc_chunks[idx],
                    "score": float(score)
                })
        return results
