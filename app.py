from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import pickle  
import numpy as np
import cv2
from skimage.feature import hog
import base64
from io import BytesIO
from PIL import Image
from tqdm import tqdm

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def enhance_contrast(image):
    """Enhance contrast using CLAHE"""
    img = image.reshape(28, 28).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(img)
    return enhanced.flatten()

def extract_hog_features(image):
    """Extract HOG features from an image"""
    img = image.reshape((28, 28))
    pixels_per_cell = (4, 4)
    cells_per_block = (2, 2)
    orientations = 9
    
    fd = hog(
        img,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        visualize=False
    )
    
    return fd

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def _init_weights_bias(self, n_features):
        self.weights = np.zeros(n_features)
        self.bias = 0

    def _hinge_loss(self, y, y_pred):
        return np.maximum(0, 1 - y * y_pred)

    def _gradient_descent(self, X, y):
        n_samples, n_features = X.shape
        y_pred = np.dot(X, self.weights) + self.bias
        dw = np.zeros(n_features)
        db = 0

        for idx, x_i in enumerate(X):
            if y[idx] * y_pred[idx] <= 1:
                dw += -y[idx] * x_i
                db += -y[idx]

        self.weights = self.weights - self.learning_rate * (2 * self.lambda_param * self.weights + dw / n_samples)
        self.bias = self.bias - self.learning_rate * (db / n_samples)

    def fit(self, X, y, verbose=True):
        n_samples, n_features = X.shape
        self._init_weights_bias(n_features)
        y_ = np.where(y <= 0, -1, 1)
        self.train_history = []

        iterator = range(self.n_iterations)
        if verbose:
            iterator = tqdm(iterator, desc="Training SVM")

        for _ in iterator:
            self._gradient_descent(X, y_)
            y_pred = np.dot(X, self.weights) + self.bias
            loss = np.mean(self._hinge_loss(y_, y_pred)) + self.lambda_param * np.sum(self.weights ** 2)
            self.train_history.append(loss)

        return self

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return np.sign(y_pred)

class OneVsRestSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.models = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        for c in self.classes:
            y_binary = np.where(y == c, 1, -1)
            svm = SVM(
                learning_rate=self.learning_rate,
                lambda_param=self.lambda_param,
                n_iterations=self.n_iterations
            )
            svm.fit(X, y_binary, verbose=False)
            self.models[c] = svm
        return self

    def predict(self, X):
        scores = np.zeros((X.shape[0], len(self.classes)))
        for idx, c in enumerate(self.classes):
            model = self.models[c]
            scores[:, idx] = np.dot(X, model.weights) + model.bias
        return self.classes[np.argmax(scores, axis=1)]

print("Loading model...")
try:
    with open("mnist_svm_model.pkl", "rb") as f:
        model_data = pickle.load(f)
    
    model = model_data["model"]
    scaler = model_data["scaler"]
    preprocessing_pipeline = model_data["preprocessing_pipeline"]
    
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

def preprocess_image(image):
    """Preprocess an image for prediction"""
    if image.shape != (28, 28):
        image = cv2.resize(image, (28, 28))
    flattened = image.flatten()
    enhanced = enhance_contrast(flattened)  
    hog_features = extract_hog_features(enhanced)  
    scaled_features = scaler.transform([hog_features])
    return scaled_features

@app.post("/predict")
async def predict(file: UploadFile = File(None), drawing: str = Form(None)):
    try:
        if file is not None:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        elif drawing is not None:
            image_data = drawing.split(",")[1]
            decoded_image = base64.b64decode(image_data)
            img = Image.open(BytesIO(decoded_image)).convert("L")
            img = np.array(img)
            if np.mean(img) > 128:
                img = 255 - img
        else:
            return JSONResponse(content={"error": "No image provided"}, status_code=400)

        processed_image = preprocess_image(img)
        prediction = model.predict(processed_image)[0]
        return {"prediction": int(prediction)}

    except Exception as e:
        print(f"Error: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)