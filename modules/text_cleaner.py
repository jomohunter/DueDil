import re

def clean_combined_output(raw_text: str) -> str:
    cleaned = raw_text

    # Remove PDF-specific metadata and objects
    cleaned = re.sub(r"/\w+", "", cleaned)  # remove PDF tags like /Type /Font
    cleaned = re.sub(r"%PDF-\d\.\d", "", cleaned)  # remove PDF version
    cleaned = re.sub(r"\b(obj|endobj|stream|endstream)\b", "", cleaned)  # remove object keywords
    cleaned = re.sub(r"<<.*?>>", "", cleaned, flags=re.DOTALL)  # remove metadata blocks

    # Remove binary/control characters (non-printable)
    cleaned = re.sub(r"[^\x20-\x7E\n]+", " ", cleaned)

    # Standardize known section titles
    section_markers = {
        r"\[TEXT CONTENT\]": "\n\n--- TEXT CONTENT ---\n\n",
        r"\[TABLES\]": "\n\n--- TABLES ---\n\n",
        r"\[EMBEDDED IMAGE OCR\]": "\n\n--- IMAGES / EMBEDDED OCR ---\n\n",
        r"\[TEXT FROM IMAGE\]": "\n\n--- IMAGE TEXT ---\n\n",
        r"\[CHART DATA\]": "\n\n--- CHART DATA ---\n\n",
        r"\[TABLE FROM IMAGE\]": "\n\n--- IMAGE TABLE ---\n\n"
    }
    for pattern, repl in section_markers.items():
        cleaned = re.sub(pattern, repl, cleaned, flags=re.IGNORECASE)

    # Remove common footer/header/page numbers
    cleaned = re.sub(r"Page\s?\d+", "", cleaned)

    # Replace weird bullet points and dashes with normal dash
    cleaned = re.sub(r"[•▪–—−•·]", "- ", cleaned)

    # Remove excessive blank lines (keep 2 max)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    # Remove leading/trailing whitespaces per line
    cleaned = "\n".join(line.strip() for line in cleaned.splitlines())

    # Redact sensitive info
    cleaned = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "<EMAIL>", cleaned)  # emails
    cleaned = re.sub(r"https?://[^\s]+", "<URL>", cleaned)  # URLs
    cleaned = re.sub(r"\b(\+?\d{1,3})?[\s\-]?\(?\d{2,4}\)?[\s\-]?\d{3,5}[\s\-]?\d{3,5}\b", "<PHONE>", cleaned)  # phone numbers

    # Remove leftover multiple spaces
    cleaned = re.sub(r"\s{2,}", " ", cleaned)

    # Final strip
    return cleaned.strip()
