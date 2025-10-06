from fastapi import FastAPI,APIRouter,UploadFile,File,HTTPException
import io
import torch
from torchvision import transforms
import torch.nn as nn
from PIL import Image

fashion_router = APIRouter(prefix='/fashion',tags=['Fashion'])

class CheckImage(nn.Module):
  def __init__(self):
      super().__init__()

      self.first = nn.Sequential(
          nn.Conv2d(1,16,kernel_size=3,padding=1),
          nn.ReLU(),
          nn.MaxPool2d(2)
      )
      self.second = nn.Sequential(
          nn.Flatten(),
          nn.Linear(16 * 14 * 14,64),
          nn.ReLU(),
          nn.Linear(64,10)
      )

  def forward(self,x):
      x = self.first(x)
      y = self.second(x)
      return y

transform = transforms.Compose({
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28,28)),
    transforms.ToTensor()
})

check_image_app = FastAPI()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CheckImage()
model.load_state_dict(torch.load('model (1).pth',map_location=device))
model.to(device)
model.eval()

classes = [
    'T-shirt/top','Trouser','Pullover','Dress',
    'Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot'
]
@check_image_app.post('fashion/predict/')
async def check_image(file: UploadFile = File(...)):
    try:
        image_data = await file.read()

        if not image_data:
            raise HTTPException(status_code=400,detail='Now file')

        img = Image.open(io.BytesIO(image_data))
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(img_tensor)
            pred = y_pred.argmax(dim=1).item()
            class_name = classes[pred]

        return {'Answer': pred, 'Название одежды': class_name}

    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))




