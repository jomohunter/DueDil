import re
import spacy
import tiktoken
import ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter

nlp = spacy.load("en_core_web_lg")
IMPORTANT_ENTITIES = {"ORG", "GPE", "MONEY", "LAW", "EVENT", "DATE", "PRODUCT", "PERCENT", "CARDINAL"}

def extract_important_phrases(text):
    return {ent.text for ent in nlp(text).ents if ent.label_ in IMPORTANT_ENTITIES}

def count_tokens(text):
    return len(tiktoken.get_encoding("cl100k_base").encode(text))

def preprocess_text(text):
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\.{3,}", " ", text)
    toc_patterns = [
        r"(?i)(?:index|table of contents|contents)[\s\S]+?(?=\n[I1]\.)",
        r"(?i)(?:index|contents)\s*(?:\.\.\.\.+|\d+)+[\s\S]+?(?=\n[I1]\.)"
    ]
    for pattern in toc_patterns:
        text = re.sub(pattern, "", text)
    return text

def get_embedding(text):
    return ollama.embeddings(model="nomic-embed-text", prompt=text)["embedding"]

def smart_chunk_text(text, max_tokens=600):
    splitter = RecursiveCharacterTextSplitter(chunk_size=max_tokens, chunk_overlap=50)
    chunks = []
    for i, chunk in enumerate(splitter.split_text(text)):
        chunks.append({
            "chunk_id": i + 1,
            "text": chunk,
            "tokens": count_tokens(chunk),
            "embedding": get_embedding(chunk),
            "important_entities": list(extract_important_phrases(chunk))
        })
    return chunks
