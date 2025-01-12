# Handwritten Digit Recognition with CNN

A modern web application that uses a CNN to recognize handwritten digits. The application provides an interactive UI where users can draw a single digit and get predictions with confidence scores upon clicking the predict button.

## Project Structure

```
handwritten-digit-recognition-cnn/
├── app/
│   ├── model/              # Trained model files
│   ├── static/
│   │   ├── css/           # Styling files
│   │   └── js/            # Frontend JavaScript
│   ├── templates/         # HTML templates
│   └── app.py            # Main Flask application
├── Notebook/             # Jupyter notebooks for model training
├── images/              # Project images and resources
└── requirements.txt     # Project dependencies
```

## Technical Components

### 1. Neural Network Architecture (CNN)
The project uses a Convolutional Neural Network with the following architecture:
- Input Layer: 1 channel (grayscale images 28*28 pixels)
- 3 Convolutional Blocks:
  - Conv1: 32 filters (3x3), BatchNorm, ReLU, MaxPool
  - Conv2: 64 filters (3x3), BatchNorm, ReLU, MaxPool
  - Conv3: 128 filters (3x3), BatchNorm, ReLU, MaxPool
- Fully Connected Layers:
  - FC1: 128*3*3 → 512 neurons
  - Dropout (p=0.5)
  - FC2: 512 → 10 neurons (output)

### 2. Web Application
- **Backend**: Flask server handling:
  - Model inference
  - Image preprocessing
  - API endpoints
- **Frontend**:
  - Interactive canvas for digit drawing
  - Real-time predictions
  - Probability distribution visualization
## Dependencies
- Python 3.x
- PyTorch 
- Flask 

## Features
- Draw a single digit on an interactive canvas
- Predict the drawn digit using CNN model
- View prediction results after clicking "Predict" button
- See probability distribution for all digits (0-9)
- Model confidence score display


## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/saidibnerradi603/handwritten-digit-recognition-cnn.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app/app.py
```

4. Open your browser and navigate to `http://localhost:5000`

## Usage
1. Draw a digit (0-9) on the canvas 
2. Click the "Predict" button to get the model's prediction
3. The system will display:
   - The predicted digit
   - Confidence score for the prediction
   - Probability distribution for all digits
4. Use "Clear Canvas" to erase and draw a new digit

## Demo

<video controls width="600">
  <source src="https://github.com/saidibnerradi603/handwritten-digit-recognition-cnn/blob/master/images/demo.mp4)" type="video/mp4">
</video>


## Model Performance
- Training accuracy: 99.2%
- Trained on the MNIST dataset

## License
This project is open source and available under the MIT License. Feel free to use, modify, and distribute the code for any purpose.
