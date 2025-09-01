import cv2
import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import img_to_array

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Hindi character labels (Unicode codes + conjuncts)
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

def load_and_preprocess_image(image_path, target_size=(32, 32)):
    """Load and preprocess image for prediction"""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"❌ Image file not found: {image_path}")

    # Resize & normalize
    image = cv2.resize(image, target_size)
    image = image.astype("float32") / 255.0

    # Expand dimensions to match model input
    image = np.expand_dims(image, axis=-1)  # (32, 32, 1)
    image = np.expand_dims(image, axis=0)   # (1, 32, 32, 1)
    return image

def predict_character(model_path, image_path):
    """Predict Hindi character from image"""
    print("[INFO] Loading trained model...")
    model = tf.keras.models.load_model(model_path)

    image = load_and_preprocess_image(image_path)
    preds = model.predict(image, verbose=0)[0]

    # Best prediction
    best_idx = np.argmax(preds)
    predicted_label = labels[best_idx]
    confidence = preds[best_idx] * 100

    # Top-3 predictions (optional, for debugging)
    top3_idx = preds.argsort()[-3:][::-1]
    top3 = [(labels[i], preds[i] * 100) for i in top3_idx]

    return predicted_label, confidence, top3

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("⚠️ Usage: python predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = "HindiModel.h5"

    try:
        predicted_label, confidence, top3 = predict_character(model_path, image_path)
        print(f"✅ Recognized Character: {predicted_label} ({confidence:.2f}% confidence)")

        print("\n🔹 Top 3 Predictions:")
        for label, conf in top3:
            print(f"   {label} - {conf:.2f}%")

    except Exception as e:
        print(f"❌ Error: {e}")
