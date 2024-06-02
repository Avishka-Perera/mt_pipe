from torchvision.models import resnet50 as pt_resnet50, ResNet50_Weights
from torch.nn import Linear


def resnet50(n_classes=1000):
    model = pt_resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = Linear(in_features=2048, out_features=n_classes, bias=True)
    return model
