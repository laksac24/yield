import pickle
import gdown
import os
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Initialize the FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow any origin to access the API
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Define input schema
class ModelInput(BaseModel):
    N: int
    P: int
    K: int
    temperature: float
    humidity: float
    ph: float
    rainfall: float
    crop: int

# Function to download the model from Google Drive
def download_model():
    file_id = "1_D9TH0QkDIHyVM8eUWsA58Ej3Ur09hEH"  # Your Google Drive file ID
    url = f"https://drive.google.com/uc?id={file_id}"  # Google Drive direct download URL
    output = "Yield_Prediction"  # Save the model file locally

    if not os.path.exists(output):
        print("Downloading the model from Google Drive...")
        gdown.download(url, output, quiet=False)
    else:
        print("Model already exists. Skipping download.")

# Call the download function when the app starts
download_model()

# Load the model
model = pickle.load(open("Yield_Prediction", "rb"))

# Prediction endpoint
@app.post("/predict")
def prediction(input_param: ModelInput):
    input_data = input_param.dict()

    # Extract features for the model
    nitrogen = input_data['N']
    phosphorous = input_data['P']
    potassium = input_data['K']
    temp = input_data['temperature']
    humid = input_data['humidity']
    phv = input_data['ph']
    rain = input_data['rainfall']
    crop = input_data['crop']

    # Prepare input for model prediction
    input_list = [nitrogen, phosphorous, potassium, temp, humid, phv, rain, crop]

    # Get the model's prediction
    prediction = model.predict([input_list])
    predicted_yield = float(prediction[0])
    return JSONResponse(content={"predicted_yield": predicted_yield})

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Yield Prediction API"}
