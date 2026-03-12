from pathlib import Path
import pymupdf


def parse_pdf(pdf_path: str) -> list[dict]:
    """
    Opens a PDF and extracts cleaned text from each page.
    Returns a list of dicts with page number and text.
    """  
    doc = pymupdf.open(str(pdf_path))
    cleaned = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        clean_text = " ".join(lines)
        cleaned.append({"page": i+1, "text": clean_text})
    doc.close()
    return (cleaned)




