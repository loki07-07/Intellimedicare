from flask import Flask, render_template, request, url_for
from tensorflow import keras
from tensorflow.keras.utils import img_to_array  # use tf.keras utils
from PIL import Image
import numpy as np
import os
from datetime import datetime

# If the app.py file lives inside D:/Intellimedicare/Brain_tumor/,
# static should usually be just "static" (not "Brain_tumor/static")
app = Flask(__name__, static_folder='static')

# ---- Load the SavedModel via TFSMLayer (Keras 3 way) ----
# Signature you printed: input (None, 128, 128, 3) float32, output (None, 4)


tfsml = keras.layers.TFSMLayer(
    r"D:/Intellimedicare/Brain_tumor/model_saved",
    call_endpoint="serving_default"
)

inp = keras.Input(shape=(128, 128, 3), dtype="float32", name="input_2")

# IMPORTANT: explicitly select the 'dense_1' output so predict() returns (1, 4)
out = tfsml(inp)["dense_1"]

model = keras.Model(inp, out, name="wrapped_savedmodel")

class_labels = ['Glioma', 'Meningioma', 'notumor', 'Pituitary']

# Set a subfolder inside static for uploads
UPLOAD_FOLDER = os.path.join(app.static_folder, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def predict_image(image):
    image = image.convert('RGB').resize((128, 128))
    img = img_to_array(image).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)  # (1,128,128,3)

    preds = model.predict(img)  # expect (1, 4) after fix above

    # Coerce to a 2D array: (batch, classes)
    preds = np.asarray(preds)
    if preds.ndim == 0:
        raise ValueError(f"Unexpected scalar prediction: {preds!r}")
    if preds.ndim == 1:
        preds = preds.reshape(1, -1)  # (1, C)
    elif preds.ndim > 2:
        preds = preds.reshape(preds.shape[0], -1)  # collapse anything exotic

    probs = preds[0]              # (C,)
    pred_index = int(np.argmax(probs))
    confidence = float(np.max(probs))

    label = class_labels[pred_index]
    result = "🟢 No Tumor Detected" if label == "notumor" else f"🔴 Tumor Detected: {label}"
    return result, f"{confidence*100:.2f}%"

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    confidence = None
    image_file = None

    if request.method == 'POST':
        file = request.files.get('image')
        if file and file.filename:
            try:
                image = Image.open(file)
                result, confidence = predict_image(image)

                filename = f"upload_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
                image_path = os.path.join(UPLOAD_FOLDER, filename)
                image.convert('RGB').save(image_path, quality=92)
                image_file = f"uploads/{filename}"
            except Exception as e:
                print("❌ Image handling error:", e)

    return render_template('index.html',
                           result=result,
                           confidence=confidence,
                           image_file=image_file,
                           now=datetime.now())

if __name__ == '__main__':
    # Avoid double-start on Windows due to reloader
    app.run(debug=True, use_reloader=False)
