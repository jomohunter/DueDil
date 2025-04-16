import json
import ollama
import os

# === LLM Configuration ===
MODEL_NAME = "llama3.1:latest"
SYSTEM_PROMPT = (
    "You are an expert due diligence analyst specialized in crypto funds.\n"
    "Answer each question using ONLY the provided context. "
    "Be concise and factual. If the answer isn't clearly supported, say: 'Insufficient data to answer.'"
)

def load_matched_questions(matched_file_path):
    with open(matched_file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_answer(question: str, context: str) -> str:
    prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"].strip()

def generate_all_answers(filename: str):
    matched_file_path = os.path.join("temp", f"matched_{filename}.json")
    output_dir = "generated_answers"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"answers_{filename}.json")

    print("🧠 Loading matched questions...")
    matched = load_matched_questions(matched_file_path)
    results = []

    for item in matched:
        qid = item["question_id"]
        question = item["question"]
        top_chunks = [match["text"] for match in item["matches"]]
        context = "\n---\n".join(top_chunks)

        print(f"💬 Generating answer for Q{qid}...")
        answer = generate_answer(question, context)

        results.append({
            "question_id": qid,
            "question": question,
            "answer": answer
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"✅ Done. Answers saved to: {output_path}")
    return output_path
