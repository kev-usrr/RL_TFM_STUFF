import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet18
from torch.utils.data import Dataset

IMAGE_SIZE = 224

# Dataset personalizado
# Dataset personalizado
class EnvObservationDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor()
        ])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)


# Modelo ResNet desde cero para imágenes RGB
class ResNetBinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # RGB
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)  # Binaria

    def forward(self, x):
        return self.resnet(x)

    def extract_embedding(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        return x  # Embedding antes de la capa final