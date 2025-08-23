from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from torchvision import transforms
from PIL import Image
import io

# Import your trained Net class
from mainFile import Net  

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
model = Net()
model.load_state_dict(torch.load("mnist_model.pth", map_location=torch.device(device)))
model.eval()

# Preprocessing (28x28 grayscale like MNIST)
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("L")
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)
            pred = output.argmax(dim=1, keepdim=True).item()

        return JSONResponse(content={"prediction": pred})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
