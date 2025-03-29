import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import pickle
from PIL import Image
import numpy as np
import cv2
import warnings
warnings.filterwarnings('ignore')

from django.conf import settings

# Define file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'medicines.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'train_models')
MODEL_PATH = os.path.join(MODEL_DIR, 'medicine_model.pkl')
ENCODER_PATH = os.path.join(MODEL_DIR, 'medicine_encoder.pkl')

# Define consistent image dimensions
IMAGE_SIZE = (64, 64)  # Changed to 64x64 for consistency
FEATURE_SIZE = IMAGE_SIZE[0] * IMAGE_SIZE[1]  # 4096 features

def read_csv_with_encoding():
    """Try different encodings to read the CSV file"""
    encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(CSV_PATH, encoding=encoding)
            # Clean column names
            df.columns = df.columns.str.strip().str.replace('\n', '').str.replace('"', '')
            
            # Clean data
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
    """Preprocess the image for model input"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        resized = cv2.resize(gray, IMAGE_SIZE)  # Resize to 64x64
        
        flattened = resized.flatten() / 255.0  # Normalize and flatten
        return flattened
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")


def train_medicine_model():
    """Train the medicine recognition model"""
    try:
        # Create model directory if it doesn't exist
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Read and clean data
        df = read_csv_with_encoding()
        
        if 'Medicine Name' not in df.columns or 'Image Path' not in df.columns:
            raise ValueError("CSV must contain 'Medicine Name' and 'Image Path' columns")

        # Extract features from images
        X, y = [], []
        for _, row in df.iterrows():
            image_path = os.path.join(BASE_DIR, row['Image Path'])
            try:
                features = preprocess_image(image_path)  # Extract real image features
                X.append(features)
                y.append(row['Medicine Name'])
            except Exception as e:
                print(f"Skipping {image_path}: {e}")

        if not X:
            raise ValueError("No valid images found for training")

        X = np.array(X)
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Train model
        model = SVC(kernel='linear', probability=True)
        model.fit(X, y_encoded)

        # Save model and encoder
        joblib.dump(model, MODEL_PATH)
        joblib.dump(le, ENCODER_PATH)

        print("Model trained and saved successfully at:", MODEL_PATH)
        print("Encoder saved successfully at:", ENCODER_PATH)

        return True
    except Exception as e:
        print(f"Error training model: {str(e)}")
        return False


import joblib

MEDICINE_IMAGES_DIR = os.path.join(settings.BASE_DIR, 'app', 'static', 'medicine_images')

def predict_medicine_details(image_path):
    """Load the trained model and make a prediction"""
    try:
        # Load model and encoder
        model = joblib.load(MODEL_PATH)
        encoder = joblib.load(ENCODER_PATH)

        # Preprocess image
        features = preprocess_image(image_path).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)
        predicted_label = encoder.inverse_transform(prediction)[0]

        # Fetch additional details from the CSV
        df = read_csv_with_encoding()
        medicine_row = df[df['Medicine Name'] == predicted_label]

        if medicine_row.empty:
            return {'error': 'No details found for predicted medicine'}

        # Get the correct image path for response
        medicine_image_filename = medicine_row['Image Path'].values[0]  # Assuming CSV contains filenames
        medicine_image_full_path = os.path.join(MEDICINE_IMAGES_DIR, os.path.basename(medicine_image_filename))

        medicine_info = {
            'name': predicted_label,
            'usage': medicine_row['Usage'].values[0] if 'Usage' in medicine_row else 'Unknown',
            'advantages': medicine_row['Advantages'].values[0] if 'Advantages' in medicine_row else 'Unknown',
            'age_group': medicine_row['Age Group'].values[0] if 'Age Group' in medicine_row else 'Unknown',
            'side_effects': medicine_row['Side Effects'].values[0] if 'Side Effects' in medicine_row else 'Unknown',
            'image_path': medicine_image_full_path
        }

        return medicine_info
    except Exception as e:
        return {'error': f"Error in prediction: {str(e)}"}



# When running the script directly
if __name__ == "__main__":
    # Print the current working directory
    print("Current working directory:", os.getcwd())
    
    # Print the full path to the CSV file
    print("CSV file path:", os.path.abspath(CSV_PATH))
    
    # Try to read and print the first few lines of the CSV
    try:
        with open(CSV_PATH, 'r', encoding='latin1') as f:
            print("\nFirst few lines of CSV:")
            for i, line in enumerate(f):
                if i < 5:  # Print first 5 lines
                    print(line.strip())
                else:
                    break
    except Exception as e:
        print("Error reading CSV:", str(e))
    
    # Train the model
    train_medicine_model() 