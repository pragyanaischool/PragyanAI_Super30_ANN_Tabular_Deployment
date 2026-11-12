import logging
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel
import tensorflow as tf

# Configure standard logging for Render
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("diabetes_api")

app = FastAPI(
    title="Diabetes Prediction API",
    description=(
        "FastAPI backend serving an optimized TFLite ANN model for Diabetes"
        " Prediction"
    ),
    version="1.0.0",
)

# Configure CORS for your Netlify frontend
origins = [
    "https://pragyanai-ann-tabular-tinymodel.netlify.app",[cite: 1]
    "http://localhost:3000",
    "http://localhost:3001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DiabetesInput(BaseModel):
  Pregnancies: float
  Glucose: float
  BloodPressure: float
  SkinThickness: float
  Insulin: float
  BMI: float
  DiabetesPedigreeFunction: float
  Age: float


# Load Scaler and TFLite Model at startup
try:
  logger.info("Initializing model assets and scaler...")
  scaler = joblib.load("model/scaler.pkl")
  interpreter = tf.lite.Interpreter(model_path="model/diabetes_tiny_model.tflite")
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  logger.info("Model and scaler loaded successfully!")
except Exception as e:
  logger.error(f"⚠️ Failed to load model or scaler assets: {e}")


@app.get("/")
def read_root():
  logger.info("Health check endpoint '/' accessed.")
  return {
      "status": "Online",
      "service": "Diabetes Prediction ANN TFLite API",
      "frontend_origin": "https://pragyanai-ann-tabular-tinymodel.netlify.app",
  }


@app.post("/predict")
def predict_diabetes(data: DiabetesInput):
  logger.info(f"Received prediction request payload: {data.dict()}")
  try:
    # 1. Convert input data into 2D numpy array
    input_data = np.array(
        [[
            data.Pregnancies,
            data.Glucose,
            data.BloodPressure,
            data.SkinThickness,
            data.Insulin,
            data.BMI,
            data.DiabetesPedigreeFunction,
            data.Age,
        ]],
        dtype=np.float32,
    )

    # 2. Scale features
    scaled_data = scaler.transform(input_data).astype(np.float32)

    # 3. Run TFLite inference
    interpreter.set_tensor(input_details[0]["index"], scaled_data)
    interpreter.invoke()

    # 4. Extract output probability
    output_data = interpreter.get_tensor(output_details[0]["index"])
    raw_prob = float(output_data[0][0])

    prediction = 1 if raw_prob >= 0.5 else 0
    result_text = (
        "Positive for Diabetes Risk"
        if prediction == 1
        else "Negative for Diabetes Risk"
    )

    logger.info(
        f"Inference complete. Probability: {raw_prob:.4f} -> Result:"
        f" {result_text}"
    )

    return {
        "prediction": prediction,
        "probability": round(raw_prob, 4),
        "result": result_text,
    }

  except Exception as e:
    logger.error(f"Inference execution failed: {str(e)}", exc_info=True)
    raise HTTPException(
        status_code=500, detail=f"Inference execution failed: {str(e)}"
    )
