import os
from pathlib import Path
import sys
import io

import fitz  # PyMuPDF (for PDFs)
from PIL import Image, ImageOps, ImageFilter
import pytesseract
import google.generativeai as genai

# ------------- CONFIG -------------
# If Tesseract isn't on PATH, set its full path here (yours):
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\dhana\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# Gemini API key (you can keep this or move to env var GEMINI_API_KEY)
API_KEY = "#Replace your key"
genai.configure(api_key=API_KEY)

# Tesseract OCR settings (tweak if needed)
TESS_LANG = "eng"  # add +tam or others if needed, e.g., "eng+tam"
TESS_CONFIG = "--oem 3 --psm 6"  # balanced defaults for printed text
# ----------------------------------


# --------- OCR HELPERS ----------
def preprocess_for_ocr(pil_img: Image.Image) -> Image.Image:
    """
    Light pre-processing to help Tesseract:
    - convert to grayscale
    - upscale a bit for small text
    - automatic contrast equalization
    - slight sharpening
    """
    img = pil_img.convert("L")  # grayscale
    # Upscale small images (helps OCR)
    if min(img.size) < 1000:
        scale = 1.5
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    # Auto-contrast and gentle sharpen
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
    return img


def ocr_pil_image(pil_img: Image.Image) -> str:
    try:
        prepped = preprocess_for_ocr(pil_img)
        text = pytesseract.image_to_string(prepped, lang=TESS_LANG, config=TESS_CONFIG)
        return (text or "").strip()
    except pytesseract.TesseractNotFoundError:
        raise RuntimeError(
            "Tesseract OCR not found. Check the pytesseract.pytesseract.tesseract_cmd path."
        )


# --------- EXTRACTORS ----------
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Hybrid extractor:
      - Use PyMuPDF to get embedded text.
      - For pages with no text, render and OCR with Tesseract.
    """
    p = Path(pdf_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise RuntimeError(f"Could not open PDF (password-protected or corrupt?): {e}")

    parts = []
    for i, page in enumerate(doc):
        # Try embedded text first
        text = (page.get_text("text") or "").strip()
        if not text:
            # OCR fallback @300 dpi
            pix = page.get_pixmap(dpi=300, alpha=False)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = ocr_pil_image(img)
        if text:
            parts.append(f"\n--- Page {i+1} ---\n{text}")

    doc.close()
    return "\n".join(parts).strip()


def extract_text_from_image(img_path: str) -> str:
    """
    Handle single images (JPG/PNG) and multi-page TIFFs.
    """
    p = Path(img_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    try:
        with Image.open(p) as im:
            texts = []
            # Multi-frame support (e.g., TIFF)
            try:
                i = 0
                while True:
                    im.seek(i)
                    page_img = im.copy()
                    page_text = ocr_pil_image(page_img)
                    if page_text:
                        texts.append(f"\n--- Image Page {i+1} ---\n{page_text}")
                    i += 1
            except EOFError:
                # No more frames
                pass
            return "\n".join(texts).strip()
    except Exception as e:
        raise RuntimeError(f"Could not open image: {e}")


# --------- GEMINI ----------
def get_prescribed_medicines(extracted_text: str) -> str:
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    prompt = f"""
You are a medical prescription analyzer. Extract the list of prescribed medicines from the following prescription text.
Output only the medicines in a bullet list format. If no medicines are found, say "No medicines prescribed."

Prescription text:
{extracted_text}
"""
    resp = model.generate_content(prompt)
    return (getattr(resp, "text", "") or "").strip()


# --------- CLI CHATBOT ----------
def is_pdf(path: str) -> bool:
    return Path(path).suffix.lower() == ".pdf"


def is_image(path: str) -> bool:
    return Path(path).suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def chatbot():
    print("Welcome to the Prescription Analyzer Chatbot!")
    print("Tip: You can enter a PDF or an image (JPG/PNG/TIFF). If your path has spaces, wrap it in quotes.")
    while True:
        pdf_or_img_path = input("Enter the path to your prescription file (or type 'exit' to quit): ").strip()
        if pdf_or_img_path.lower() == "exit":
            print("Goodbye!")
            break

        try:
            if is_pdf(pdf_or_img_path):
                extracted_text = extract_text_from_pdf(pdf_or_img_path)
            elif is_image(pdf_or_img_path):
                extracted_text = extract_text_from_image(pdf_or_img_path)
            else:
                print("Unsupported file type. Please provide a PDF or an image (jpg, png, tiff, bmp, webp).")
                continue

            if not extracted_text:
                print("No text could be extracted (even with OCR). "
                      "The file may be blank, very low resolution, or password-protected.")
                continue

            print(f"\n[Info] Extracted text length: {len(extracted_text)} characters")
            medicines = get_prescribed_medicines(extracted_text)
            print("\nPrescribed Medicines:")
            print(medicines if medicines else "No medicines prescribed.")

        except Exception as e:
            print(f"Error: {e}\nPlease check the file path and prerequisites (Tesseract installed).")


if __name__ == "__main__":
    chatbot()
