import os
import io
from pathlib import Path

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

import fitz  # PyMuPDF (for PDFs)
from PIL import Image, ImageOps, ImageFilter
import pytesseract
import google.generativeai as genai

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
import time
import random

# ---------------- CONFIG ----------------
# Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\dhana\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# Gemini API key (prefer env var: setx GEMINI_API_KEY "YOUR_KEY")
API_KEY = 'AIzaSyBx4KoDKcnUEqsiEpLRhQgvcqzEhixyB4U'
genai.configure(api_key=API_KEY)

TESS_LANG = "eng"  # add +tam if needed: "eng+tam"
TESS_CONFIG = "--oem 3 --psm 6"
ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"}

# Selenium / ChromeDriver
CHROMEDRIVER_PATH = r"C:\Users\dhana\Downloads\chromedriver-win64\chromedriver.exe"

# Flask
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
# ----------------------------------------


# ---------------- OCR HELPERS ----------------
def preprocess_for_ocr(pil_img: Image.Image) -> Image.Image:
    img = pil_img.convert("L")
    if min(img.size) < 1000:
        scale = 1.5
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
    return img


def ocr_pil_image(pil_img: Image.Image) -> str:
    prepped = preprocess_for_ocr(pil_img)
    text = pytesseract.image_to_string(prepped, lang=TESS_LANG, config=TESS_CONFIG)
    return (text or "").strip()
# ------------------------------------------------


# ---------------- EXTRACTORS ----------------
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
            # OCR fallback @300 dpi
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
        texts = []
        with Image.open(p) as im:
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
# ------------------------------------------------


# ---------------- GEMINI ----------------
def parse_bullets_to_list(raw: str) -> list:
    """
    Convert bullet/numbered/loose lines to a clean list of medicine strings.
    Accepts '-', '*', '•', '·', numbers like '1.' or '1)'.
    """
    meds = []
    for line in (raw or "").splitlines():
        s = line.strip()
        if not s:
            continue
        # Strip common bullet markers
        for prefix in ["- ", "* ", "• ", "· ", "— ", "-- "]:
            if s.startswith(prefix):
                s = s[len(prefix):].strip()
                break
        # Strip numbered bullets
        if len(s) > 2 and (s[0].isdigit() and (s[1] in {")", "."})):
            s = s[2:].strip()
        # Remove trailing punctuation commonly used in lists
        s = s.rstrip(" .;:,")
        if s:
            meds.append(s)
    # De-dup while preserving order
    seen = set()
    out = []
    for m in meds:
        if m.lower() not in seen:
            seen.add(m.lower())
            out.append(m)
    return out


def get_prescribed_medicines(extracted_text: str) -> list:
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    prompt = f"""
You are a medical prescription analyzer. Extract the list of prescribed medicines from the following prescription text.
- Output only the medicines in a simple bullet list (one per line).
- Do not include dosage, route, or instructions.
- If no medicines are found, reply with exactly: No medicines prescribed.

Prescription text:
{extracted_text}
"""
    resp = model.generate_content(prompt)
    text = (getattr(resp, "text", "") or "").strip()
    if text == "No medicines prescribed.":
        return []
    return parse_bullets_to_list(text)
# ------------------------------------------------


# ---------------- WRAPPER ----------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_medicines(file_path: str) -> list:
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        extracted_text = extract_text_from_pdf(file_path)
    elif ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}:
        extracted_text = extract_text_from_image(file_path)
    else:
        raise ValueError("Unsupported file type. Use PDF or image (jpg, png, tiff, bmp, webp).")
    if not extracted_text:
        return []
    return get_prescribed_medicines(extracted_text)
# ------------------------------------------------


# ---------------- SCRAPER (1mg) ----------------
def scrape_1mg(medicine_name: str) -> dict:
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)")

    driver = webdriver.Chrome(service=Service(CHROMEDRIVER_PATH), options=options)
    driver.set_page_load_timeout(30)

    result = {"Medicine Name": medicine_name}
    search_url = f"https://www.1mg.com/search/all?name={medicine_name}"

    try:
        driver.get(search_url)
        WebDriverWait(driver, 12).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "a[href^='/drugs/'], a[href^='/otc/']"))
        )
        time.sleep(random.uniform(1.2, 2.0))
        items = driver.find_elements(By.CSS_SELECTOR, "a[href^='/drugs/'], a[href^='/otc/']")
        href = None
        for item in items:
            link = item.get_attribute("href")
            if link:
                href = link
                break
        if not href:
            result["error"] = "No product found."
            return result

        driver.get(href)
        time.sleep(random.uniform(1.0, 1.8))

        # Be defensive: check existence before reading
        result["Brand Name"] = driver.find_element(By.CSS_SELECTOR, ".DrugHeader__title-content___2ZaPo").text if driver.find_elements(By.CSS_SELECTOR, ".DrugHeader__title-content___2ZaPo") else "N/A"
        result["Manufacturer"] = driver.find_element(By.XPATH, "//div[contains(text(),'Marketer')]/following-sibling::div").text if driver.find_elements(By.XPATH, "//div[contains(text(),'Marketer')]/following-sibling::div") else "N/A"
        result["Salt"] = driver.find_element(By.XPATH, "//div[contains(.,'SALT COMPOSITION')]/following-sibling::div").text if driver.find_elements(By.XPATH, "//div[contains(.,'SALT COMPOSITION')]/following-sibling::div") else "N/A"
        # Price selector changes often; try multiple:
        price_el = (driver.find_elements(By.CSS_SELECTOR, "span.PriceBoxPlanOption__offer-price___3v9x8") or
                    driver.find_elements(By.CSS_SELECTOR, "[class*='offer-price']") or
                    driver.find_elements(By.CSS_SELECTOR, "[data-testid='price']"))
        result["Price"] = price_el[0].text if price_el else "N/A"

        uses = [li.text for li in driver.find_elements(By.CSS_SELECTOR, "#uses_and_benefits li")]
        result["Uses"] = ", ".join(uses) if uses else "N/A"

        side_effects = [li.text for li in driver.find_elements(By.CSS_SELECTOR, "#side_effects ul li")]
        result["Side Effects"] = ", ".join(side_effects) if side_effects else "N/A"

        result["URL"] = href

    except (TimeoutException, WebDriverException) as e:
        result["error"] = str(e)
    finally:
        driver.quit()

    return result
# ------------------------------------------------


# ---------------- ROUTES ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    data = []
    error = None
    if request.method == "POST":
        if "file" not in request.files:
            error = "No file uploaded."
        else:
            file = request.files["file"]
            if file.filename == "":
                error = "No file selected."
            elif file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(file_path)
                try:
                    medicines = extract_medicines(file_path)
                    if not medicines:
                        error = "No medicines extracted from the file."
                    else:
                        for med in medicines:
                            data.append(scrape_1mg(med))
                finally:
                    # Optional cleanup
                    try:
                        os.remove(file_path)
                    except Exception:
                        pass
            else:
                error = "Invalid file type. Use PDF or image."
    return render_template("index.html", data=data, error=error)


if __name__ == "__main__":
    # For local dev
    app.run(debug=True)
