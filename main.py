from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torchvision.transforms as transforms
import io

from model_def import SimpleNN  # Your model class

app = FastAPI()

# Load the trained model
model = SimpleNN()
model.load_state_dict(torch.load("cats_vs_dogs_model.pth", map_location=torch.device('cpu')))
model.eval()

# Define the image transforms to match 28x28 grayscale input
transform = transforms.Compose([
    transforms.Grayscale(),              # Convert to grayscale
    transforms.Resize((28, 28)),         # Resize to 28x28
    transforms.ToTensor(),               # Convert to tensor (1, 28, 28)
    transforms.Lambda(lambda x: x.view(-1))  # Flatten to (784,)
])

@app.get("/")
def root():
    return {"message": "Cat vs Dog Classifier is running."}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_t = transform(image).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            outputs = model(img_t)
            _, predicted = torch.max(outputs, 1)
            label = "Cat" if predicted.item() == 0 else "Dog"

        return JSONResponse(content={"prediction": label})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
