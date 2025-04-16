import json
import numpy as np
import faiss
import ollama
from pathlib import Path
from modules.faiss_store_embeddings import load_faiss_index

# === Configuration ===
QUESTIONS_FILE = "modules/due_diligence_questions_final_clean.json"
TOP_K = 5  # Number of top chunks to retrieve per question

def load_questions():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def load_chunks(chunks_file):
    with open(chunks_file, "r", encoding="utf-8") as f:
        return json.load(f)

def embed_question(text: str):
    """Generate a vector embedding for the input question using Ollama."""
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return np.array(response["embedding"]).astype("float32")

def match_questions_to_chunks(chunks_file="temp/chunks.json", faiss_index_path="data/faiss_index.index", output_path="modules/matched_questions.json"):
    print("🔍 Loading data...")
    questions = load_questions()
    chunks = load_chunks(chunks_file)
    index = load_faiss_index(faiss_index_path)

    chunk_id_map = {i: chunk for i, chunk in enumerate(chunks)}
    matched_results = []

    print("🚀 Matching questions to relevant chunks...\n")

    for q in questions:
        q_id = q.get("id")
        q_text = q.get("question")

        q_vector = embed_question(q_text).reshape(1, -1)
        distances, indices = index.search(q_vector, TOP_K)

        matches = []
        for score, idx in zip(distances[0], indices[0]):
            if idx in chunk_id_map:
                matches.append({
                    "chunk_id": chunk_id_map[idx].get("chunk_id"),
                    "text": chunk_id_map[idx].get("text"),
                    "score": float(score)
                })

        matched_results.append({
            "question_id": q_id,
            "question": q_text,
            "matches": matches
        })

        print(f"✅ Q{q_id}: matched with top {len(matches)} chunks")

    print("\n💾 Saving match results...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(matched_results, f, indent=4, ensure_ascii=False)

    print(f"🎯 Done! Results saved to: {output_path}")

if __name__ == "__main__":
    match_questions_to_chunks()
