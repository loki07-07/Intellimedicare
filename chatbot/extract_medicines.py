import os
from pathlib import Path
import fitz  # PyMuPDF (for PDFs)
from PIL import Image, ImageOps, ImageFilter
import pytesseract
import google.generativeai as genai

# ------------- CONFIG -------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\dhana\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
API_KEY = "AIzaSyBx4KoDKcnUEqsiEpLRhQgvcqzEhixyB4U"
genai.configure(api_key=API_KEY)
TESS_LANG = "eng"
TESS_CONFIG = "--oem 3 --psm 6"

# --------- OCR HELPERS ----------
def preprocess_for_ocr(pil_img: Image.Image) -> Image.Image:
    img = pil_img.convert("L")
    if min(img.size) < 1000:
        scale = 1.5
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
    return img

def ocr_pil_image(pil_img: Image.Image) -> str:
    try:
        prepped = preprocess_for_ocr(pil_img)
        text = pytesseract.image_to_string(prepped, lang=TESS_LANG, config=TESS_CONFIG)
        return (text or "").strip()
    except pytesseract.TesseractNotFoundError:
        raise RuntimeError("Tesseract OCR not found. Check the tesseract_cmd path.")

# --------- EXTRACTORS ----------
def extract_text_from_pdf(pdf_path: str) -> str:
    p = Path(pdf_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise RuntimeError(f"Could not open PDF: {e}")
    parts = []
    for i, page in enumerate(doc):
        text = (page.get_text("text") or "").strip()
        if not text:
            pix = page.get_pixmap(dpi=300, alpha=False)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = ocr_pil_image(img)
        if text:
            parts.append(f"\n--- Page {i+1} ---\n{text}")
    doc.close()
    return "\n".join(parts).strip()

def extract_text_from_image(img_path: str) -> str:
    p = Path(img_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    try:
        with Image.open(p) as im:
            texts = []
            i = 0
            while True:
                try:
                    im.seek(i)
                    page_img = im.copy()
                    page_text = ocr_pil_image(page_img)
                    if page_text:
                        texts.append(f"\n--- Image Page {i+1} ---\n{page_text}")
                    i += 1
                except EOFError:
                    break
            return "\n".join(texts).strip()
    except Exception as e:
        raise RuntimeError(f"Could not open image: {e}")

# --------- GEMINI ----------
def get_prescribed_medicines(extracted_text: str) -> list:
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    prompt = f"""
You are a medical prescription analyzer. Extract the list of prescribed medicines from the following prescription text.
Output only the medicines in a bullet list format. If no medicines are found, return an empty list.

Prescription text:
{extracted_text}
"""
    resp = model.generate_content(prompt)
    text = (getattr(resp, "text", "") or "").strip()
    if text == "No medicines prescribed.":
        return []
    # Parse bullet list into a Python list
    medicines = [line.strip()[2:] for line in text.split("\n") if line.strip().startswith("- ")]
    return medicines

# --------- Extract Medicines Function ----------
def extract_medicines(file_path: str) -> list:
    try:
        if Path(file_path).suffix.lower() == ".pdf":
            extracted_text = extract_text_from_pdf(file_path)
        elif Path(file_path).suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}:
            extracted_text = extract_text_from_image(file_path)
        else:
            raise ValueError("Unsupported file type. Use PDF or image (jpg, png, tiff, bmp, webp).")
        if not extracted_text:
            return []
        return get_prescribed_medicines(extracted_text)
    except Exception as e:
        print(f"Error extracting medicines: {e}")
        return []