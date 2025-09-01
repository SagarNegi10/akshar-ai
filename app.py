from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import cv2
import base64

app = Flask(__name__)

# Load trained model safely
try:
    model = tf.keras.models.load_model("HindiModel.h5")
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Error loading model:", e)
    model = None

# Hindi labels (in the same order as training dataset)
labels = [
    u'\u091E',  # ञ
    u'\u091F',  # ट
    u'\u0920',  # ठ
    u'\u0921',  # ड
    u'\u0922',  # ढ
    u'\u0923',  # ण
    u'\u0924',  # त
    u'\u0925',  # थ
    u'\u0926',  # द
    u'\u0927',  # ध
    u'\u0915',  # क
    u'\u0928',  # न
    u'\u092A',  # प
    u'\u092B',  # फ
    u'\u092C',  # ब
    u'\u092D',  # भ
    u'\u092E',  # म
    u'\u092F',  # य
    u'\u0930',  # र
    u'\u0932',  # ल
    u'\u0935',  # व
    u'\u0916',  # ख
    u'\u0936',  # श
    u'\u0937',  # ष
    u'\u0938',  # स
    u'\u0939',  # ह
    u'\u0915\u094D\u0937',  # क्ष
    u'\u0924\u094D\u0930',  # त्र
    u'\u091C\u094D\u091E',  # ज्ञ
    u'\u0917',  # ग
    u'\u0918',  # घ
    u'\u0919',  # ङ
    u'\u091A',  # च
    u'\u091B',  # छ
    u'\u091C',  # ज
    u'\u091D',  # झ
    u'\u0966',  # ०
    u'\u0967',  # १
    u'\u0968',  # २
    u'\u0969',  # ३
    u'\u096A',  # ४
    u'\u096B',  # ५
    u'\u096C',  # ६
    u'\u096D',  # ७
    u'\u096E',  # ८
    u'\u096F'   # ९
]

def preprocess_image(image_b64):
    """Decode and preprocess base64 image for prediction."""
    # Decode base64 image
    image_data = image_b64.split(",")[1]
    nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    # Preprocess (match model input)
    img = cv2.resize(img, (32, 32))      # adjust to your model input size
    img = cv2.bitwise_not(img)           # invert (white char on black bg)
    img = img.astype("float32") / 255.0  # normalize
    img = np.expand_dims(img, axis=-1)   # add channel
    img = np.expand_dims(img, axis=0)    # add batch
    return img

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image received"}), 400

    # Preprocess
    img = preprocess_image(data["image"])

    # Predict
    pred = model.predict(img)
    pred = np.squeeze(pred)  # remove batch dim
    top_idx = int(np.argmax(pred))
    predicted_label = labels[top_idx]
    confidence = float(pred[top_idx]) * 100

    return jsonify({
        "prediction": predicted_label,
        "confidence": f"{confidence:.2f}%"
    })

if __name__ == "__main__":
    if model:
        print("Model input shape:", model.input_shape)
    app.run(debug=True)
