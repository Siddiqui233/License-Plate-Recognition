🚗 Number Plate Detection and Recognition

![image](https://github.com/user-attachments/assets/f5d1e145-cf4f-4b97-974a-2839a6935030)

This project focuses on automatic vehicle license plate detection and character recognition using computer vision and deep learning techniques. The pipeline is divided into two main stages:

Detection – Locate the license plate within a car image.

Recognition – Extract and recognize the alphanumeric characters from the detected license plate.

📌 Features
✅ Detects and localizes license plates in real-world vehicle images

✂️ Crops license plate regions from detected bounding boxes

🔡 Recognizes license plate text using OCR or deep learning-based models

🗂 Supports batch processing for folders of images

🧩 Easy-to-understand visualizations for debugging and presentations

🔁 Modular, clean, and reusable Python codebase

🗂️ Project Structure

number-plate-detection/
│
├── license_plates_detection_train/     # Car images (Detection dataset)
├── license_plates_recognition_train/   # Cropped license plate images (Recognition dataset)
├── Licplatesdetection_train.csv        # Bounding box annotations for detection
├── Licplatesrecognition_train.csv      # Plate text labels for recognition
├── test_images/                        # Test set for evaluation
├── src/                                # Source code (scripts and modules)
│   ├── detect_plate.py                 # Detection logic
│   ├── recognize_plate.py              # Recognition logic
│   ├── utils.py                        # Helper functions
│   └── visualize.py                    # Image display and annotation
├── outputs/                            # Predicted outputs and results
├── requirements.txt                    # List of Python dependencies
└── README.md                           # Project documentation

🚀 Quick Start
1. Clone the Repository
🔹 Training Set 1 – Detection

900 vehicle images

Each image annotated with bounding box coordinates for license plates

Format: (ymin, xmin, ymax, xmax)

🔹 Training Set 2 – Recognition
900 cropped license plate images

Each image is labeled with the corresponding plate number (e.g., MH12AB1234)

🔹 Test Set
201 car images

Task: detect and recognize license plates

Evaluation is based only on character recognition accuracy

⚙️ Methods Used
🔍 License Plate Detection
Object detection models such as:

YOLOv5

Faster R-CNN

Input: Full car image

Output: Bounding box coordinates for the plate

🔡 License Plate Recognition
OCR using:

Tesseract

CRNN (Convolutional Recurrent Neural Network)

TrOCR (Transformer-based OCR)

Input: Cropped plate image

Output: Alphanumeric text

📦 Requirements
Example Python dependencies in requirements.txt:

Copy
Edit
opencv-python
matplotlib
pandas
numpy
tqdm
torch
torchvision
pytesseract
Install them with:

bash
Copy
Edit
pip install -r requirements.txt
📈 Evaluation
Model performance is evaluated only on the accuracy of license plate text recognition.
Detection performance indirectly affects recognition (bad crops = bad text).
🧠 Future Improvements
🔧 Train a custom detection model for better plate localization
🤖 Improve recognition with advanced models like TrOCR or fine-tuned CNNs
🌐 Add a web demo using Flask or Streamlit
🚘 Add support for:
Multi-line plates
Multiple plates per image

📚 Acknowledgements
Dataset and task inspired by real-world vehicle identification challenges.
Libraries used:
OpenCV
PyTorch
pytesseract
matplotlib
pandas
numpy




