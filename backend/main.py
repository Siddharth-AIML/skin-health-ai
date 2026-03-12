import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import sys

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Classes (must match folder names)
classes = ['healthy', 'mild', 'moderate']

# CNN Model (same architecture as train.py)
class SkinCNN(nn.Module):
    def __init__(self):
        super(SkinCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(32 * 30 * 30, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load model
model = SkinCNN().to(device)
model.load_state_dict(torch.load("models/skin_cnn.pth", map_location=device))
model.eval()

# Image transform (same as training)
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return classes[predicted.item()]

if __name__ == "__main__":
    image_path = sys.argv[1]
    result = predict(image_path)
    print(f"Predicted Skin Condition: {result}")