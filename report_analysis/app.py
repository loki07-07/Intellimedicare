# app.py
import os
from pathlib import Path
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

# --- Gemini (NEW client) ---
# pip install google-genai pydantic
from google import genai
from google.genai import types as gat

# ---------------- CONFIG ----------------
API_KEY = '#Replace your key'

client = genai.Client(api_key=API_KEY)

ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"}

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
# ----------------------------------------


# ---------------- HELPERS ----------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def upload_to_gemini(file_path: str):
    # Works for PDFs and images (Files API)
    return client.files.upload(file=Path(file_path))


SYSTEM_INSTRUCTION = (
    "You are a cautious clinical lab interpretation assistant. "
    "You are NOT a doctor and must NOT provide a diagnosis. "
    "Given ONLY the attached lab report file, identify up to five POSSIBLE medical "
    "conditions or nutrient deficiencies that could plausibly explain any abnormal values "
    "based strictly on the report's stated reference ranges (use age/sex-specific ranges if provided). "
    "If the report shows no clear abnormalities or lacks enough information, return an empty list."
)

USER_PROMPT = (
    "Task: Based ONLY on the attached laboratory report file and its stated reference ranges, "
    "return a list of 0–5 POSSIBLE conditions or deficiencies that could explain clear abnormalities.\n"
    "Rules:\n"
    "- Do NOT diagnose.\n"
    "- Output only concise names (e.g., 'iron deficiency anemia', 'vitamin B12 deficiency', 'hypothyroidism', 'diabetes mellitus').\n"
    "- No numbers or explanations.\n"
    "- If insufficient evidence or normal, return an empty list.\n"
    "Output strictly as JSON: an array of strings."
)


def analyze_blood_report(file_path: str) -> list[str]:
    uploaded = upload_to_gemini(file_path)

    resp = client.models.generate_content(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-pro"),
        contents=[uploaded, USER_PROMPT],
        config=gat.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=list[str],  # force a JSON array of strings
        ),
    )

    items = []
    if isinstance(getattr(resp, "parsed", None), list):
        items = [s.strip() for s in resp.parsed if isinstance(s, str) and s.strip()]
    else:
        # Fallback: parse text as JSON
        import json
        try:
            tmp = json.loads((resp.text or "[]").strip())
            if isinstance(tmp, list):
                items = [str(s).strip() for s in tmp if isinstance(s, str) and str(s).strip()]
        except Exception:
            items = []

    # Deduplicate (case-insensitive), cap at 5
    seen, out = set(), []
    for s in items:
        k = s.lower()
        if k not in seen:
            seen.add(k)
            out.append(s)
        if len(out) >= 5:
            break
    return out
# ----------------------------------------


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
                    issues = analyze_blood_report(file_path)
                    if issues:
                        data = issues
                    else:
                        error = "No issues found in the report."
                except Exception as e:
                    error = f"Processing failed: {e}"
                finally:
                    try:
                        os.remove(file_path)
                    except Exception:
                        pass
            else:
                error = "Invalid file type. Use PDF or image."
    return render_template("index.html", data=data, error=error)


if __name__ == "__main__":
    app.run(debug=True)
