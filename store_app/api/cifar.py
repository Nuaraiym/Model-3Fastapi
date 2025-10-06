from fastapi import FastAPI,APIRouter,UploadFile,File,HTTPException
import io
import torch
from torch.xpu import device
from torchvision import transforms
import torch.nn as nn
from PIL import Image


cifar_router = APIRouter(prefix='/cifar',tags=['Cifar'])


class CirafClassifaction(nn.Module):
    def __init__(self):
        super().__init__()

        self.first = nn.Sequential(
            nn.Conv2d(1,16,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16*14*14,64),
            nn.ReLU(),
            nn.Linear(64,10)
        )

    def forward(self,image):
        image = self.first(image)
        image = self.second(image)
        return image

transform = transforms.Compose({
    transforms.Resize((32,32)),
    transforms.ToTensor()
})

cifar_app = FastAPI()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CirafClassifaction()
model.load_state_dict(torch.load('model (3).pth',map_location=device))
model.to(device)
model.eval()

@cifar_app.post('cifar/predict/')
async def check_image(file: UploadFile = File(...)):
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400,detail='File not')

        img = Image.open(io.BytesIO(data))
        img_tensor = transform(img).unsqeeze(0).to(device)

        with torch.no_grad():
            pred = model(img_tensor)
            result = pred.argmax(dim=1).item()
            return {'class': result}

    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))

