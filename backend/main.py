# ==========================================
# BACKEND: FastAPI Inference Engine (main.py)
# ==========================================
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
import tensorflow as tf
import os

# 1. Initialize FastAPI Application
app = FastAPI(
    title="Diabetes Prediction API",
    description="High-performance serverless inference API powered by a Tiny TFLite ANN model.",
    version="1.0.0"
)

# 2. Configure CORS Middleware (Allow requests from Netlify frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your specific Netlify frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Load Scaler and Tiny TFLite Model
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
TFLITE_MODEL_PATH = os.path.join(MODEL_DIR, "diabetes_tiny_model.tflite")

try:
    scaler = joblib.load(SCALER_PATH)
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("Model and scaler loaded successfully into inference engine.")
except Exception as e:
    print(f"Error loading model assets: {e}")

# 4. Define Pydantic Request Body Schema
class PatientData(BaseModel):
    pregnancies: float
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree: float
    age: float

@app.get("/")
def read_root():
    return {
        "status": "Online",
        "service": "India Diabetes ANN Tiny Model Inference API",
        "docs_url": "/docs"
    }

@app.post("/predict")
def predict_diabetes(data: PatientData):
    try:
        # Extract features into a 2D numpy array
        features = np.array([[
            data.pregnancies,
            data.glucose,
            data.blood_pressure,
            data.skin_thickness,
            data.insulin,
            data.bmi,
            data.diabetes_pedigree,
            data.age
        ]], dtype=np.float32)

        # Scale features using the saved StandardScaler
        scaled_features = scaler.transform(features)

        # Run inference using the Tiny TFLite Interpreter
        interpreter.set_tensor(input_details[0]['index'], scaled_features)
        interpreter.invoke()
        prediction_tensor = interpreter.get_tensor(output_details[0]['index'])

        probability = float(prediction_tensor[0][0])
        diagnostic_result = "⚠️ High Risk: Diabetic Indicators Detected" if probability > 0.5 else "✅ Low Risk: Healthy Profile"

        return {
            "prediction_score": round(probability, 4),
            "confidence_percentage": f"{probability * 100:.2f}%",
            "diagnostic_result": diagnostic_result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference execution failed: {str(e)}")
