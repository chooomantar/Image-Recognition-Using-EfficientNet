import os
import cv2
import numpy as np
import pickle
from PIL import Image
from io import BytesIO
import base64
from sklearn.preprocessing import normalize
from keras.applications import efficientnet
from keras.applications.efficientnet import EfficientNetB7
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate, Dense
from keras.models import Model
import tensorflow as tf
from sklearn.model_selection import train_test_split
import sys
from collections import namedtuple
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
from zipfile import ZipFile
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Define the Case namedtuple
Case = namedtuple('Case', ['features', 'label', 'image_data'])

# Use this for preprocessing
preprocess_input = efficientnet.preprocess_input

# Data augmentation for training
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode='nearest'
)

# Global variables
unique_labels = []
model = None

# Define and compile the model
def create_model():
    global unique_labels
    base_model = EfficientNetB7(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x1 = GlobalAveragePooling2D()(x)
    x2 = GlobalMaxPooling2D()(x)
    x = Concatenate()([x1, x2])
    x = Dense(1024, activation='relu')(x)
    outputs = Dense(len(unique_labels), activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=outputs)

    # Freeze early layers
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to preprocess an image
async def preprocess_image(image):
    # Convert PIL image to numpy array if necessary
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Deblur
    deblurred = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    
    # Enhance contrast
    lab = cv2.cvtColor(deblurred, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced

# Function to extract features
async def extract_features(image):
    global model
    img = await preprocess_image(image)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    features = model.predict(img)
    features = features.flatten()
    features = normalize(features.reshape(1, -1))[0]  # L2 normalization
    return features

# Function to load the case base from a file
def load_case_base(case_base_file):
    try:
        with open(case_base_file, 'rb') as f:
            try:
                return pickle.load(f)
            except ModuleNotFoundError:
                # Create a fake 'train2' module
                class train2:
                    Case = Case
                sys.modules['train2'] = train2
                return pickle.load(f)
    except FileNotFoundError:
        logger.warning(f"{case_base_file} not found. Starting with an empty case base.")
        return []

# Function to calculate similarity
def calculate_similarity(X, Y):
    return 1 - np.arccos(np.clip(np.dot(X, Y), -1.0, 1.0)) / np.pi

# Function to retrieve similar cases from the case base
def retrieve_similar_cases(query_features, case_base, top_n=5):
    similarities = [calculate_similarity(query_features, case.features) for case in case_base]
    similar_cases = [(case.label, case.image_data, sim) for case, sim in zip(case_base, similarities)]
    
    # Sort by similarity score in descending order and return top_n
    return sorted(similar_cases, key=lambda x: x[2], reverse=True)[:top_n]

# Function to save the case base to a file
def save_case_base(case_base, case_base_file):
    with open(case_base_file, 'wb') as f:
        pickle.dump(case_base, f)

# Function to save the model
def save_model(model, filepath='model.h5'):
    model.save(filepath)

# Function to load the model
def load_model(filepath='model.h5'):
    return tf.keras.models.load_model(filepath)

async def unzipfile(zip_file: UploadFile):
    if zip_file.filename.endswith('.zip'):
        with open('temp.zip', 'wb') as f:
            f.write(await zip_file.read())
        
        with ZipFile('temp.zip', 'r') as zip_ref:
            zip_ref.extractall(path=os.getcwd())
        
        return os.path.join(os.getcwd(), "new_parts")
    return None

# Initialize case_base and model
case_base = load_case_base('case_base.pkl')
model = load_model('model.h5') if os.path.exists('model.h5') else create_model()

@app.post("/search_similar_images/")
async def search_similar_images(file: UploadFile = File(...)):
    global model
    if not case_base:
        return JSONResponse(content={"error": "Case base is empty. Please train the model first."})

    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        query_image_np = np.array(image)

        query_features = await extract_features(query_image_np)
        similar_cases = retrieve_similar_cases(query_features, case_base, top_n=5)

        results = []
        for label, image_data, similarity_score in similar_cases:
            accuracy = similarity_score * 100
            if accuracy > 50:
                results.append({
                    "label": label,
                    "similarity": f"{accuracy:.2f}%",
                })

        return JSONResponse(content={"similar_images": results})
    except Exception as e:
        logger.error(f"Error during image search: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/train/")
async def train(zip_file: UploadFile = File(...)):
    global case_base, model, unique_labels
    try:
        dataset_directory = await unzipfile(zip_file)
        if not dataset_directory:
            raise ValueError("Uploaded file is not a zip file")

        images = []
        labels = []
        for part_number in os.listdir(dataset_directory):
            part_path = os.path.join(dataset_directory, part_number)
            if not os.path.isdir(part_path):
                continue
            for file_name in os.listdir(part_path):
                if file_name.endswith((".jpg", ".jpeg", ".png", ".webp")):
                    image_path = os.path.join(part_path, file_name)
                    img = cv2.imread(image_path)
                    img = cv2.resize(img, (224, 224))
                    img = preprocess_input(img)
                    images.append(img)
                    labels.append(part_number)

        X = np.array(images)
        y = np.array(labels)

        unique_labels = np.unique(y)
        model = create_model()

        label_to_index = {label: index for index, label in enumerate(unique_labels)}
        y_encoded = np.array([label_to_index[label] for label in y])
        y_one_hot = tf.keras.utils.to_categorical(y_encoded)

        X_train, X_val, y_train, y_val = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)
        
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
        
        # Save model after training
        save_model(model)

        case_base = []
        for img, label in zip(X, y):
            features = await extract_features(img)
            _, buffer = cv2.imencode('.jpg', img)
            img_str = base64.b64encode(buffer).decode('utf-8')
            case = Case(features, label, img_str)
            case_base.append(case)

        save_case_base(case_base, 'case_base.pkl')
        return JSONResponse(content={"message": "Training completed successfully"})
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
