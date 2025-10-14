import cv2
import pytesseract

# Path to tesseract.exe
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\dhana\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# Load image
image = cv2.imread(r"C:\Users\dhana\Pictures\Screenshots\Screenshot 2025-07-28 174459.png")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding
thresh = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV, 15, 8
)

# Denoise
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
processed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# Save for verification (optional)
cv2.imwrite("processed_handwritten.png", processed)

# OCR
custom_config = r'--oem 1 --psm 6'
text = pytesseract.image_to_string(processed, config=custom_config)
print("📝 Extracted Text:\n")
print(text)
