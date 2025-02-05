from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import dill as pickle
import numpy as np
import pandas as pd
import yaml
import sys
import os


app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

    
# Load the model

MODEL_FILE = config["model_file"]
with open(MODEL_FILE, 'rb') as file:
    model = pickle.load(file)



# Define the form fields
form_fields = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None, "image_url": None})

@app.post("/", response_class=HTMLResponse)
async def predict(request: Request, N: float = Form(...), P: float = Form(...), K: float = Form(...), temperature: float = Form(...), humidity: float = Form(...), ph: float = Form(...), rainfall: float = Form(...)):
    # Create a DataFrame from the form input
    input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=form_fields)
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    #LabelEncoder.inverse_transform(prediction)
    
    # Decode the prediction
    #if hasattr(model.named_steps['label_encoder'], 'inverse_transform'):
        #prediction = model.named_steps['label_encoder'].inverse_transform([prediction])[0]
    
    
    # Determine the image URL based on the prediction
    image_url = f"/static/images/{prediction}.png"
    
    return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction, "image_url": image_url})



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)