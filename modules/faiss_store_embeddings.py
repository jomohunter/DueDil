import faiss
import numpy as np

def store_embeddings_in_faiss(embeddings):
    array = np.array(embeddings)
    index = faiss.IndexFlatL2(array.shape[1])
    index.add(array)
    return index

def save_faiss_index(index, path):
    faiss.write_index(index, path)
    print(f"✅ FAISS index saved to {path}")

def load_faiss_index(path):
    return faiss.read_index(path)