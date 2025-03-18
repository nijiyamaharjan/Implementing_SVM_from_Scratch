import pickle
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import cv2
from skimage.feature import hog
from tqdm import tqdm

# Define the model classes
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
        for c in tqdm(self.classes, desc="Training models"):
            y_binary = np.where(y == c, 1, -1)
            svm = SVM(learning_rate=self.learning_rate,
                      lambda_param=self.lambda_param,
                      n_iterations=self.n_iterations)
            svm.fit(X, y_binary, verbose=False)
            self.models[c] = svm
        return self

    def predict(self, X):
        scores = np.zeros((X.shape[0], len(self.classes)))
        for idx, c in enumerate(self.classes):
            model = self.models[c]
            scores[:, idx] = np.dot(X, model.weights) + model.bias
        return self.classes[np.argmax(scores, axis=1)]

# Functions for feature extraction
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

# Load a small subset of MNIST for training
print("Loading MNIST dataset...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')
X = X.astype('float32')
y = y.astype('int')

# Take a small subset for faster training
n_samples = 10000
np.random.seed(42)
indices = np.random.permutation(X.shape[0])[:n_samples]
X_subset = X.iloc[indices].values
y_subset = y.iloc[indices].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_subset, y_subset, test_size=0.2, random_state=42
)

# Create preprocessing pipeline
print("Preprocessing data...")
preprocessing_pipeline = {
    'enhance_contrast': enhance_contrast,
    'extract_hog': extract_hog_features
}

X_train_preprocessed = []
for img in tqdm(X_train, desc="Preprocessing training data"):
    enhanced = preprocessing_pipeline['enhance_contrast'](img)
    hog_features = preprocessing_pipeline['extract_hog'](enhanced)
    X_train_preprocessed.append(hog_features)

X_train_preprocessed = np.array(X_train_preprocessed)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_preprocessed)

# Train model
print("Training model...")
model = OneVsRestSVM(
    learning_rate=0.01,
    lambda_param=0.0005,
    n_iterations=600
)
model.fit(X_train_scaled, y_train)

# Save everything needed for prediction
print("Saving model and preprocessing components...")

# Create a dedicated module for the classes
# This approach ensures pickle can find the classes when loading
import sys
from types import ModuleType

# Create a module called 'svm_module'
svm_module = ModuleType('svm_module')
svm_module.SVM = SVM
svm_module.OneVsRestSVM = OneVsRestSVM
svm_module.enhance_contrast = enhance_contrast
svm_module.extract_hog_features = extract_hog_features
sys.modules['svm_module'] = svm_module

# Pack model components
model_data = {
    "model": model,
    "scaler": scaler,
    "preprocessing_pipeline": preprocessing_pipeline
}

# Save the model
with open("mnist_svm_model.pkl", "wb") as f:
    pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Model saved successfully as 'mnist_svm_model.pkl'")