from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder("../data/images", transform=transform)

loader = DataLoader(dataset, batch_size=8, shuffle=True)

print("Classes:", dataset.classes)
print("Total images:", len(dataset))