import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import shutil
import json

from modules.text_extractor import extract_data_from_file
from modules.text_cleaner import clean_combined_output
from modules.text_chunking_and_embedding import smart_chunk_text, preprocess_text
from modules.faiss_store_embeddings import store_embeddings_in_faiss, save_faiss_index
from modules.question_matcher import match_questions_to_chunks
from modules.llm_responder import generate_all_answers

# === Folder Paths ===
UPLOAD_FOLDER = "uploads/"
DATA_FOLDER = "data/"
TEMP_FOLDER = "temp/"
ANSWER_FOLDER = "generated_answers/"
HISTORY_FILE = os.path.join(DATA_FOLDER, "upload_history.json")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)
os.makedirs(ANSWER_FOLDER, exist_ok=True)

# === Streamlit UI ===
st.set_page_config(page_title="Crypto DD Automation", layout="wide")
st.title("🧠 Crypto Fund Due Diligence Automation")
st.sidebar.header("📤 Upload Your Document")

uploaded_file = st.sidebar.file_uploader(
    "Supported: PDF, DOCX, XLS(X), CSV, PNG, JPG",
    type=["pdf", "docx", "xls", "xlsx", "csv", "jpg", "png"]
)

# === Upload History Utility ===
def update_upload_history(file_name, answer_path):
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
    entry = {"file": file_name, "answers": answer_path}
    if entry not in history:
        history.append(entry)
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)

# === Upload History Sidebar ===
st.sidebar.markdown("## 📚 Upload History")
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "r") as f:
        history = json.load(f)
    for entry in history:
        if st.sidebar.button(f"📄 {entry['file']}"):
            st.subheader(f"📋 Answers for: {entry['file']}")
            with open(entry["answers"], "r", encoding="utf-8") as f_ans:
                answers = json.load(f_ans)
            for a in answers:
                st.markdown(f"**Q{a['question_id']}: {a['question']}**")
                st.success(a["answer"])
                st.markdown("---")

# === Main Processing Flow ===
if uploaded_file:
    filename = uploaded_file.name
    st.success(f"✅ Uploaded: {filename}")

    # STEP 1: Save uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # STEP 2: Extract text
    text_file_path, _ = extract_data_from_file(file_path)
    if not text_file_path or not os.path.exists(text_file_path):
        st.error("❌ Failed to extract readable text.")
        st.stop()

    with open(text_file_path, "r", encoding="utf-8") as tf:
        raw_text = tf.read()

    # STEP 3: Clean + preprocess
    cleaned = clean_combined_output(raw_text)
    preprocessed = preprocess_text(cleaned)

    # STEP 4: Chunk and embed
    st.info("🔍 Chunking and embedding text...")
    chunks = smart_chunk_text(preprocessed, max_tokens=600)
    embeddings = [chunk["embedding"] for chunk in chunks]

    # Save chunks temporarily
    chunk_json = [{"chunk_id": c["chunk_id"], "text": c["text"]} for c in chunks]
    chunks_file = os.path.join(TEMP_FOLDER, "chunks.json")
    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(chunk_json, f, indent=4, ensure_ascii=False)

    # STEP 5: Store embeddings in FAISS
    index = store_embeddings_in_faiss(embeddings)
    save_faiss_index(index, os.path.join(DATA_FOLDER, "faiss_index.index"))
    st.success("📦 FAISS index built and saved.")

    # STEP 6: Match questions
    matched_output_path = os.path.join(TEMP_FOLDER, f"matched_{filename}.json")
    st.info("🔗 Matching document to due diligence questions...")
    match_questions_to_chunks(output_path=matched_output_path, chunks_file=chunks_file)

    # STEP 7: Generate answers
    st.info("🤖 Generating LLM-powered answers...")
    generated_answer_path = generate_all_answers(filename)

    # Save to history
    update_upload_history(filename, generated_answer_path)

    # STEP 8: Display Answers
    st.subheader(f"📋 Final Answers for: {filename}")
    with open(generated_answer_path, "r", encoding="utf-8") as f:
        responses = json.load(f)
    for answer in responses:
        st.markdown(f"**Q{answer['question_id']}: {answer['question']}**")
        st.success(answer["answer"])
        st.markdown("---")
