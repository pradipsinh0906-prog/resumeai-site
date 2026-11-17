import PyPDF2
import io

def extract_text_from_pdf(uploaded_file):
    """Extract raw text from an uploaded PDF file object."""
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()