from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Create app
app = FastAPI()

# Enable CORS for all devices
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = tf.keras.models.load_model("fabric_model.h5")

# Class labels (must match training order)
class_names = [
    "chiffon",
    "cotton",
    "georgette",
    "kotanet",
    "linen",
    "organza",
    "silk",
    "tussar"
]

@app.get("/")
def home():
    return {"message": "Fabric API is running successfully"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image bytes
        contents = await file.read()

        # Open image
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Resize to model input size
        image = image.resize((224, 224))

        # Convert to array & normalize
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Predict
        predictions = model.predict(image_array)

        class_index = int(np.argmax(predictions))
        confidence = float(np.max(predictions)) * 100

        predicted_class = class_names[class_index]

        return {
            "predicted_class": predicted_class,
            "confidence": round(confidence, 2)
        }

    except Exception as e:
        return {
            "predicted_class": "Error",
            "confidence": 0,
            "error": str(e)
        }