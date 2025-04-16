# --- modules/text_extractor.py ---
import fitz  # PyMuPDF
import docx
import pandas as pd
import easyocr
from PIL import Image
import pdfplumber
import os

UPLOAD_FOLDER = "uploads/"
DATA_FOLDER = "data/"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

ocr_reader = easyocr.Reader(['en'])

def save_text_to_file(text, file_name):
    path = os.path.join(DATA_FOLDER, f"{file_name}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path

def save_tables_to_file(tables, file_name):
    ext = os.path.splitext(file_name)[1].lower()
    if isinstance(tables, dict):
        excel_path = os.path.join(DATA_FOLDER, f"{file_name}.xlsx")
        with pd.ExcelWriter(excel_path) as writer:
            for sheet, df in tables.items():
                df.to_excel(writer, sheet_name=sheet, index=False)
        return excel_path
    elif isinstance(tables, pd.DataFrame):
        path = os.path.join(DATA_FOLDER, f"{file_name}.xlsx")
        tables.to_excel(path, index=False)
        return path
    return None

def extract_text_from_pdf(pdf_path):
    with fitz.open(pdf_path) as doc:
        return "\n".join(page.get_text("text") for page in doc)

def extract_text_and_tables_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text, tables = "", []
        for page in pdf.pages:
            text += page.extract_text() or ""
            tables.extend(page.extract_tables())
        return text, tables

def extract_text_from_word(path):
    doc = docx.Document(path)
    return "\n".join(para.text for para in doc.paragraphs)

def extract_tables_from_excel_or_csv(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv": return pd.read_csv(path)
    elif ext in [".xls", ".xlsx"]: return pd.read_excel(path, sheet_name=None)
    raise ValueError("Unsupported file type")

def extract_text_from_image(image_path):
    result = ocr_reader.readtext(image_path)
    return "\n".join([d[1] for d in result])

def extract_data_from_file(path):
    ext = os.path.splitext(path)[1].lower()
    name = os.path.basename(path)
    if ext == ".pdf":
        text, tables = extract_text_and_tables_from_pdf(path)
        return save_text_to_file(text, name), save_tables_to_file(tables, name)
    elif ext == ".docx":
        return save_text_to_file(extract_text_from_word(path), name), None
    elif ext in [".xls", ".xlsx", ".csv"]:
        return None, save_tables_to_file(extract_tables_from_excel_or_csv(path), name)
    elif ext in [".jpg", ".jpeg", ".png"]:
        return save_text_to_file(extract_text_from_image(path), name), None
    else:
        raise ValueError("Unsupported file type")