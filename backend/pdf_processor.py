import fitz
def extract_text_from_pdf(pdf_path: str) -> str:
    """
        Opens a PDF file and extracts all text content.
    """
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text("text")
            full_text += f"--- Page {page_num + 1} ---\n"
            full_text += page_text + "\n"
        doc.close()
        return full_text.strip()
    except Exception as e:
        return f"Error extracting PDF: {str(e)}"
