import json
import logging
import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResNetClassifier(nn.Module):
    """ResNet-based image classifier (must match training model)."""
    
    def __init__(self, num_classes=10):
        super(ResNetClassifier, self).__init__()
        from torchvision import models
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)


# CIFAR-10 class names
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']


def model_fn(model_dir):
    """Load the PyTorch model from the model directory."""
    logger.info(f"Loading model from {model_dir}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetClassifier(num_classes=10)
    
    model_path = os.path.join(model_dir, 'model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    logger.info("Model loaded successfully")
    return model


def input_fn(request_body, request_content_type):
    """Deserialize and prepare the prediction input."""
    logger.info(f"Content type: {request_content_type}")
    
    if request_content_type == 'application/x-image':
        # Handle raw image bytes
        image = Image.open(io.BytesIO(request_body))
        return image
    elif request_content_type == 'application/json':
        # Handle JSON with base64 encoded image
        data = json.loads(request_body)
        if 'image' in data:
            import base64
            image_bytes = base64.b64decode(data['image'])
            image = Image.open(io.BytesIO(image_bytes))
            return image
        else:
            raise ValueError("JSON must contain 'image' field with base64 encoded image")
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    """Make predictions on the input data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define preprocessing transforms
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Preprocess image
    if input_data.mode != 'RGB':
        input_data = input_data.convert('RGB')
    
    input_tensor = transform(input_data).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
    
    return {
        'predicted_class': predicted_class.item(),
        'class_name': CLASS_NAMES[predicted_class.item()],
        'confidence': confidence.item(),
        'probabilities': probabilities.cpu().numpy().tolist()[0]
    }


def output_fn(prediction, response_content_type):
    """Serialize the prediction output."""
    if response_content_type == 'application/json':
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported response content type: {response_content_type}")
