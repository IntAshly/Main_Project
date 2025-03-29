import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers 
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

from django.conf import settings

# Define file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'medicines.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'train_models')
MODEL_PATH = os.path.join(MODEL_DIR, 'medicine_cnn_model.h5')
ENCODER_PATH = os.path.join(MODEL_DIR, 'medicine_encoder.pkl')

# Define consistent image dimensions
IMAGE_SIZE = (64, 64, 1)  # Grayscale images


def read_csv_with_encoding():
    """Try different encodings to read the CSV file"""
    encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(CSV_PATH, encoding=encoding)
            df.columns = df.columns.str.strip().str.replace('\n', '').str.replace('"', '')
            
            for column in df.columns:
                if df[column].dtype == 'object':
                    df[column] = df[column].str.strip().fillna('')
            
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error with encoding {encoding}: {str(e)}")
            continue
    
    raise ValueError("Could not read the CSV file with any of the attempted encodings")


def preprocess_image(image_path):
    """Preprocess the image for CNN input"""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
        
        resized = cv2.resize(img, (IMAGE_SIZE[0], IMAGE_SIZE[1]))
        normalized = resized / 255.0  # Normalize pixel values
        return normalized.reshape(IMAGE_SIZE)
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")


def build_cnn_model(num_classes):
    """Build a simple CNN model"""
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=IMAGE_SIZE),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_medicine_model():
    """Train the medicine recognition model using CNN"""
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        df = read_csv_with_encoding()
        
        if 'Medicine Name' not in df.columns or 'Image Path' not in df.columns:
            raise ValueError("CSV must contain 'Medicine Name' and 'Image Path' columns")
        
        X, y = [], []
        for _, row in df.iterrows():
            image_path = os.path.join(BASE_DIR, row['Image Path'])
            try:
                features = preprocess_image(image_path)
                X.append(features)
                y.append(row['Medicine Name'])
            except Exception as e:
                print(f"Skipping {image_path}: {e}")
        
        if not X:
            raise ValueError("No valid images found for training")
        
        X = np.array(X)
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        model = build_cnn_model(len(le.classes_))
        model.fit(X, y_encoded, epochs=10, batch_size=16, validation_split=0.2)
        
        model.save(MODEL_PATH)
        joblib.dump(le, ENCODER_PATH)
        
        print("CNN model trained and saved successfully at:", MODEL_PATH)
        print("Encoder saved successfully at:", ENCODER_PATH)
        
        return True
    except Exception as e:
        print(f"Error training model: {str(e)}")
        return False


def predict_medicine_details(image_array):
    try:
        # Load trained CNN model
        model = load_model(MODEL_PATH)  # Ensure this is a CNN model
        encoder = joblib.load(ENCODER_PATH)

        # Make prediction
        prediction = model.predict(image_array)
        predicted_label_index = np.argmax(prediction)  # Get the index of the highest probability class
        predicted_label = encoder.inverse_transform([predicted_label_index])[0]

        # Fetch additional details from CSV
        df = read_csv_with_encoding()
        medicine_row = df[df['Medicine Name'] == predicted_label]

        if medicine_row.empty:
            return {'error': 'No details found for predicted medicine'}

        medicine_info = {
            'name': predicted_label,
            'usage': medicine_row['Usage'].values[0] if 'Usage' in medicine_row else 'Unknown',
            'advantages': medicine_row['Advantages'].values[0] if 'Advantages' in medicine_row else 'Unknown',
            'age_group': medicine_row['Age Group'].values[0] if 'Age Group' in medicine_row else 'Unknown',
            'side_effects': medicine_row['Side Effects'].values[0] if 'Side Effects' in medicine_row else 'Unknown',
        }

        return medicine_info
    except Exception as e:
        return {'error': f"Error in prediction: {str(e)}"}

if __name__ == "__main__":
    print("Current working directory:", os.getcwd())
    print("CSV file path:", os.path.abspath(CSV_PATH))
    
    try:
        with open(CSV_PATH, 'r', encoding='latin1') as f:
            print("\nFirst few lines of CSV:")
            for i, line in enumerate(f):
                if i < 5:
                    print(line.strip())
                else:
                    break
    except Exception as e:
        print("Error reading CSV:", str(e))
    
    train_medicine_model()