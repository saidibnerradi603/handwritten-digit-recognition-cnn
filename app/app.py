from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io
import base64

app = Flask(__name__)



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # First Convolutional Block
        self.conv1 = nn.Conv2d(
            in_channels=1,       # Number of input channels (1 for grayscale images)
            out_channels=32,     # Number of filters (32 feature maps as output)
            kernel_size=3,       # Filter size (3x3)
            padding=1            # Padding added to preserve spatial dimensions
        )
        self.bn1 = nn.BatchNorm2d(
            num_features=32      # Normalization applied to 32 feature maps
        )
        self.pool1 = nn.MaxPool2d(
            kernel_size=2        # Size reduction by a factor of 2 (2x2 max pooling)
        )

        # Second Convolutional Block
        self.conv2 = nn.Conv2d(
            in_channels=32,      # Number of input channels (32 from previous block)
            out_channels=64,     # Number of filters (64 feature maps as output)
            kernel_size=3,       # Filter size (3x3)
            padding=1            # Padding added to preserve spatial dimensions
        )
        self.bn2 = nn.BatchNorm2d(
            num_features=64      # Normalization applied to 64 feature maps
        )
        self.pool2 = nn.MaxPool2d(
            kernel_size=2        # Size reduction by a factor of 2 (2x2 max pooling)
        )

        # Third Convolutional Block
        self.conv3 = nn.Conv2d(
            in_channels=64,      # Number of input channels (64 from previous block)
            out_channels=128,    # Number of filters (128 feature maps as output)
            kernel_size=3,       # Filter size (3x3)
            padding=1            # Padding added to preserve spatial dimensions
        )
        self.bn3 = nn.BatchNorm2d(
            num_features=128     # Normalization applied to 128 feature maps
        )
        self.pool3 = nn.MaxPool2d(
            kernel_size=2        # Size reduction by a factor of 2 (2x2 max pooling)
        )

        # Fully Connected Layers
        self.fc1 = nn.Linear(
            in_features=128 * 3 * 3, # Input size (flattened after convolutions)
            out_features=512         # Number of neurons in hidden layer
        )
        self.dropout = nn.Dropout(
            p=0.5             # Probability to disable a neuron to prevent overfitting
        )
        self.fc2 = nn.Linear(
            in_features=512,       # Input size (from previous layer)
            out_features=10        # Number of output classes (10-class classification)
        )

    def forward(self, x):
        # First block
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))

        # Second block
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))

        # Third block
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        # Flatten
        x = x.view(-1, 128 * 3 * 3)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

# Load the trained model
model = CNN()
model.load_state_dict(torch.load("./model/mnist_model.pth", 
                                map_location=torch.device('cpu')))
model.eval()

def preprocess_image(image_data):
    # Decode base64 image
    image_data = image_data.split(',')[1]
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    
    # Convert to grayscale
    image = image.convert('L')
    
    # Resize to 28x28
    image = image.resize((28, 28))
    
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    image_tensor = transform(image)
    return image_tensor.unsqueeze(0)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image data from the request
        image_data = request.json['image']
        
        # Preprocess the image
        image_tensor = preprocess_image(image_data)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            prediction = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[prediction].item() * 100
            
            # Get all probabilities
            prob_list = probabilities.tolist()
            
            return jsonify({
                'success': True,
                'prediction': prediction,
                'confidence': confidence,
                'probabilities': prob_list
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)