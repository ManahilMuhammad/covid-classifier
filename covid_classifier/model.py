import torch
from torchvision.models import ResNet18_Weights, resnet18

from .config import CLASSES


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_model(num_classes=len(CLASSES), pretrained=True):
    """
    imagenet pretrained resnet18 with new classification head
    """
    model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    return model


def save_checkpoint(model, path, **meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'classes': CLASSES, **meta}, path)


def load_model(path, device=None):
    device = device or get_device()
    checkpoint = torch.load(path, map_location=device)
    
    model = build_model(len(checkpoint['classes']), pretrained=False)
    model.load_state_dict(checkpoint['state_dict'])
    return model.to(device).eval()
