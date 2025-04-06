import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import torch.nn as nn
import threading

# Dummy CNN — replace with real NSFW/toxicity classifier later
class DummyCNN(nn.Module):
    def __init__(self):
        super(DummyCNN, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 2)  # 2 classes: safe, toxic

    def forward(self, x):
        return self.base_model(x)

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None
_lock = threading.Lock()

def get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                model = DummyCNN().to(_device)
                model.eval()
                _model = model
    return _model

# Preprocessing for image input
_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_toxicity_image(image: Image.Image):
    model = get_model()
    img_tensor = _transform(image).unsqueeze(0).to(_device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()

    return {
        "safe": round(float(probs[0]), 4),
        "toxic": round(float(probs[1]), 4),
        "verdict": "toxic" if probs[1] > 0.5 else "safe"
    }
