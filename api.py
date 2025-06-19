import os
import io
import json
import uvicorn
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
from enum import Enum
import shutil
import pickle
from pathlib import Path
from contextlib import asynccontextmanager
import tempfile

# Define the skin conditions based on your notebook
SKIN_CONDITIONS = [
    "ACK-Actinic Keratosis",
    "BCC-Basal Cell Carcinoma", 
    "MEL-Melanoma", 
    "NEV-Nevus", 
    "SCC-Squamous Cell Carcinoma", 
    "SEK-Seborrheic Keratosis"
]

# Define the ConvNeXt model with metadata
class ConvNeXtWithMetadata(nn.Module):
    def __init__(self, num_classes, metadata_features=5, convnext_variant='tiny'):
        super(ConvNeXtWithMetadata, self).__init__()
        
        # Import dynamically here to avoid loading at module level
        import torchvision.models as models
        
        # Load the appropriate ConvNeXt variant
        if convnext_variant == 'tiny':
            self.base_model = models.convnext_tiny(pretrained=True)
            num_features = 768
        elif convnext_variant == 'small':
            self.base_model = models.convnext_small(pretrained=True)
            num_features = 768
        elif convnext_variant == 'base':
            self.base_model = models.convnext_base(pretrained=True)
            num_features = 1024
        elif convnext_variant == 'large':
            self.base_model = models.convnext_large(pretrained=True)
            num_features = 1536
        else:
            raise ValueError(f"Unknown ConvNeXt variant: {convnext_variant}")
        
        # Modify the classifier to output features instead of classification
        self.base_model.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # Metadata processing layers
        self.metadata_layers = nn.Sequential(
            nn.Linear(metadata_features, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Combined layers
        self.combined_layers = nn.Sequential(
            nn.Linear(num_features + 64, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, images, metadata):
        # Process images through ConvNeXt
        img_features = self.base_model(images)
        
        # Process metadata
        metadata_features = self.metadata_layers(metadata)
        
        # Combine features
        combined = torch.cat((img_features, metadata_features), dim=1)
        
        # Final classification
        output = self.combined_layers(combined)
        
        return output

# Define input models
class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"

class YesNo(str, Enum):
    YES = "yes"
    NO = "no"

class PatientMetadata(BaseModel):
    gender: Gender
    age: int
    smoke: YesNo
    drink: YesNo
    skin_cancer_history: YesNo

class PredictionResponse(BaseModel):
    predicted_condition: str
    confidence: float
    all_probabilities: dict

# Global variables for model
model = None
device = None
transform = None

async def load_model():
    """Load the model during startup"""
    global model, device, transform
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    model_path = "model/final_convnext_tiny.pth"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please ensure the model file is uploaded.")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Files in current directory: {os.listdir('.')}")
        if os.path.exists("model"):
            print(f"Files in model directory: {os.listdir('model')}")
        else:
            print("Model directory does not exist")
        return
    
    # Initialize model
    try:
        model = ConvNeXtWithMetadata(
            num_classes=len(SKIN_CONDITIONS),
            metadata_features=5,
            convnext_variant='tiny'
        ).to(device)
        
        # Check model file integrity before loading
        print(f"Model file path: {model_path}")
        file_size = os.path.getsize(model_path)
        print(f"Model file size: {file_size} bytes")
        
        # Check if file is too small (likely corrupted)
        if file_size < 1000:  # Less than 1KB is definitely wrong
            print(f"ERROR: Model file is too small ({file_size} bytes). File is likely corrupted.")
            model = None
            transform = None
            return
        
        # Read first few bytes to check file format
        with open(model_path, 'rb') as f:
            first_bytes = f.read(20)
            print(f"First 20 bytes as hex: {first_bytes.hex()}")
            
            # Check if it starts with pickle protocol
            if not first_bytes.startswith(b'\x80'):
                print("ERROR: File does not appear to be a valid pickle/PyTorch file")
                print("Expected to start with \\x80 (pickle protocol), but got:", first_bytes[:4].hex())
                model = None
                transform = None
                return
        
        # Try to load model weights
        try:
            print("Attempting to load model with weights_only=False...")
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            print(f"Checkpoint loaded successfully. Type: {type(checkpoint)}")
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    print("Loading from checkpoint['state_dict']")
                    model.load_state_dict(checkpoint['state_dict'])
                elif 'model_state_dict' in checkpoint:
                    print("Loading from checkpoint['model_state_dict']")
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    print("Loading checkpoint directly as state_dict")
                    model.load_state_dict(checkpoint)
            else:
                print("Checkpoint is not a dictionary, attempting direct load")
                model.load_state_dict(checkpoint)
                
            model.eval()
            print("Model loaded and set to eval mode successfully")
            
            # Initialize transform ONLY after successful model loading
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            print("Transform initialized successfully")
            
        except pickle.UnpicklingError as pickle_error:
            print(f"PICKLE ERROR: {pickle_error}")
            print("This indicates the model file is corrupted or in wrong format.")
            print("Please re-upload your model file to Railway.")
            model = None
            transform = None
            
        except Exception as load_error:
            print(f"Error during model loading: {load_error}")
            print(f"Error type: {type(load_error)}")
            model = None
            transform = None
        
    except Exception as e:
        print(f"Error initializing model: {e}")
        print(f"Error type: {type(e)}")
        model = None
        transform = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern lifespan event handler"""
    # Startup
    print("Starting up...")
    await load_model()
    yield
    # Shutdown
    print("Shutting down...")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Skin Cancer Detection API",
    description="API for detecting skin cancer from images and patient metadata",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def process_image_from_upload(upload_file: UploadFile) -> Image.Image:
    """
    Process image directly from upload file without saving to disk
    This is more reliable on cloud platforms like Railway
    """
    try:
        # Read the file content
        contents = upload_file.file.read()
        
        # Reset file pointer for potential future reads
        upload_file.file.seek(0)
        
        # Create PIL Image from bytes
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

def predict_skin_condition(image: Image.Image, metadata):
    """
    Make prediction using the trained model
    
    Args:
        image: PIL Image object
        metadata: Dictionary containing patient metadata
    
    Returns:
        predicted_class: The predicted skin condition
        probabilities: Probabilities for each class
    """
    global model, device, transform
    
    # Check if model is loaded
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Preprocess the image
    try:
        image_tensor = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error transforming image: {str(e)}")
    
    # Process metadata
    try:
        # Convert string values to appropriate numeric values
        smoke = 1 if str(metadata.get('smoke', '')).lower() in ['yes', 'true', '1', 'y', 't'] else 0
        drink = 1 if str(metadata.get('drink', '')).lower() in ['yes', 'true', '1', 'y', 't'] else 0
        history = 1 if str(metadata.get('skin_cancer_history', '')).lower() in ['yes', 'true', '1', 'y', 't'] else 0
        
        # Process age
        try:
            age = float(metadata.get('age', 50.0))
        except (ValueError, TypeError):
            age = 50.0
        age_normalized = age / 100.0
        
        # Process gender
        if isinstance(metadata.get('gender'), (int, float)):
            gender = float(metadata.get('gender'))
        elif isinstance(metadata.get('gender'), str):
            gender = 1 if metadata.get('gender', '').lower() in ['male', 'm', '1'] else 0
        else:
            gender = 0
        
        # Create metadata tensor
        metadata_tensor = torch.tensor([
            smoke,
            drink,
            history,
            age_normalized,
            gender
        ], dtype=torch.float32).unsqueeze(0).to(device)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing metadata: {str(e)}")
    
    # Get prediction
    try:
        with torch.no_grad():
            outputs = model(image_tensor, metadata_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
        
        # Get the predicted class name
        predicted_class = SKIN_CONDITIONS[predicted.item()]
        probabilities_list = probabilities[0].cpu().numpy()
        
        # Create a dictionary of probabilities for each class
        class_probabilities = {SKIN_CONDITIONS[i]: float(probabilities_list[i]) for i in range(len(SKIN_CONDITIONS))}
        
        return predicted_class, class_probabilities
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    image: UploadFile = File(...),
    gender: Gender = Form(...),
    age: int = Form(...),
    smoke: YesNo = Form(...),
    drink: YesNo = Form(...),
    skin_cancer_history: YesNo = Form(...)
):
    """
    Predict skin cancer condition from an image and patient metadata.
    """
    # Validate file type
    if not image.content_type or not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Validate age
    if age <= 0 or age > 120:
        raise HTTPException(status_code=400, detail="Age must be between 1 and 120")
    
    # Create metadata dictionary
    metadata = {
        "gender": gender,
        "age": age,
        "smoke": smoke,
        "drink": drink,
        "skin_cancer_history": skin_cancer_history
    }
    
    try:
        # Process image directly from upload
        processed_image = process_image_from_upload(image)
        
        # Get prediction
        predicted_class, class_probabilities = predict_skin_condition(processed_image, metadata)
        
        # Get confidence score for predicted class
        confidence = class_probabilities[predicted_class]
        
        # Return response
        return PredictionResponse(
            predicted_condition=predicted_class,
            confidence=confidence,
            all_probabilities=class_probabilities
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None:
        return {"status": "error", "message": "Model not loaded"}
    return {"status": "ok", "message": "Service is up and running"}

@app.get("/conditions")
async def get_conditions():
    """Get the list of skin conditions the model can detect"""
    return {"conditions": SKIN_CONDITIONS}

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Skin Cancer Detection API", "status": "running"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
