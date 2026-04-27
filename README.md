# Sign-Language-to Text Converter
⚙️ Tech Stack

Programming Language = Python 3.10
Libraries & Frameworks : 
OpenCV 4.7.0.72 (Image processing & webcam handling)
MediaPipe 0.10.9 (Hand detection & landmark tracking)
CVZone 1.6.1 (Simplified hand tracking wrapper)
TensorFlow 2.10.1 (Model inference)
Keras 2.10.0 (Model framework)
NumPy 1.24.4 (Numerical computations)
Protobuf 3.20.3 (Dependency support)

sign-lan-converter/
│── Data/                  # Dataset images
│── Model/                 #trained in teachable machine from google
│   ├── keras_model.h5
│   ├── labels.txt
│── datacollection.py      # Data collection script
│── test.py                # Prediction script
