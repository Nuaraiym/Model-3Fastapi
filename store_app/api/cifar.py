from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, Depends
import io
import torch
from torch.xpu import device
from torchvision import transforms
import torch.nn as nn
from PIL import Image
from store_app.db import models
from store_app.db.database import SessionLocal
from sqlalchemy.orm import Session



cifar_router = APIRouter(prefix='/cifar',tags=['Cifar'])


async def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

label = ['airplane','automobile','bird','cat',
 'deer','dog','frog','horse','ship','truck']

class CifarClassifaction(nn.Module):
  def __init__(self):
      super().__init__()
      self.first = nn.Sequential(
          nn.Conv2d(3,32,kernel_size=3,padding=1),#32x32
          nn.ReLU(),
          nn.MaxPool2d(2),

          nn.Conv2d(32,64,kernel_size=3,padding=1),#16x16
          nn.ReLU(),
          nn.MaxPool2d(2),

          nn.Conv2d(64,128,kernel_size=3,padding=1),#8x8
          nn.ReLU(),
          nn.MaxPool2d(2)
      )
      self.second = nn.Sequential(
          nn.Flatten(),#4x4
          nn.Linear(128*4*4,256),
          nn.ReLU(),
          nn.Linear(256,10)
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
model = CifarClassifaction()
model.load_state_dict(torch.load('model (3).pth',map_location=device))
model.to(device)
model.eval()

@cifar_router.post('cifar/predict')
async def check_image(file: UploadFile = File(...),db: Session = Depends(get_db)):
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400,detail='File not')

        img = Image.open(io.BytesIO(data))
        img_tensor = transform(img).unsqeeze(0).to(device)

        with torch.no_grad():
            pred = model(img_tensor)
            result = pred.argmax(dim=1).item()

            db_cifar = models.Cifar( image=data, predict=result)
            db.add(db_cifar)
            db.commit()
            db.refresh(db_cifar)

        return {'class': label[result]}

    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))

