# Implementing SVM from Scratch

This repository contains an implementation of a Support Vector Machine (SVM) from scratch for classifying handwritten digits from the MNIST dataset. The project includes a simple frontend that allows users to either draw a digit or upload an image to classify it using the trained SVM model. The code for training the model can be found here:

https://colab.research.google.com/drive/10Dp40lSoZueDbOD-WoqVuCkp8XMDNuII?usp=sharing

## Features
- **SVM Implementation**: SVM is implemented from scratch using Python without relying on any high-level libraries like sklearn.
- **Frontend Interface**: A simple web-based interface to:
  - Draw a digit using a canvas.
  - Upload an image of a digit.
  - Display the classification result after SVM inference.

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/nijiyamaharjan/Implementing_SVM_from_Scratch.git
   cd <project-directory>
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
   
3. Run the backend:
   ```bash
   python app.py
   ```

4. Open your browser and navigate to http://localhost:5000 to access the frontend.
