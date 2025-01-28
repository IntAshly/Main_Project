import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import pickle
from PIL import Image
import numpy as np

# Define the paths
BASE_PATH = r'C:\Users\Ashly\OneDrive\Documents\s9\miniproject\nurturenest\app\static\vaccine_images'
CSV_PATH = r'C:\Users\Ashly\OneDrive\Documents\s9\miniproject\nurturenest\app\vaccine_data.csv'

# Function to preprocess image into a feature vector
def preprocess_image(image_path):
    print(f"Trying to open image: {image_path}")
    
    # Check if the file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at: {image_path}")
    
    # Open and preprocess the image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((64, 64))  # Resize to 64x64 pixels
    img_array = np.array(img).flatten()  # Flatten to a 1D array
    return img_array

def train_model():
    # Load the dataset
    data = pd.read_csv(CSV_PATH)
    
    # Preprocess images
    try:
        data['image_features'] = data['image_path'].apply(preprocess_image)
        X = np.stack(data['image_features'].values)
    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        return
    
    y = data['name']
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # Train the model
    try:
        model = SVC(kernel='linear', probability=True)
        model.fit(X_train, y_train)
        
        # Save the trained model and label encoder
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open('label_encoder.pkl', 'wb') as f:
            pickle.dump(label_encoder, f)
        
        print("Model and label encoder saved successfully!")
    except Exception as e:
        print(f"Error during model training: {e}")

train_model()

# Predict vaccine details from the uploaded image
def predict_vaccine_details(image_path):
    # Load the trained model and label encoder
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Preprocess the image and make prediction
    image_features = preprocess_image(image_path).reshape(1, -1)

    try:
        # Get the predicted class
        prediction = model.predict(image_features)
        
        # Decode the prediction to get the vaccine name
        vaccine_name = label_encoder.inverse_transform(prediction)[0]
        
        # Fetch vaccine details
        vaccine_details = get_vaccine_details(vaccine_name)
        return vaccine_details
    
    except Exception as e:
        # Handle prediction failure
        return {'error': f"Prediction failed: {str(e)}"}

def get_vaccine_details(name):
    data = pd.read_csv(CSV_PATH)
    
    # Check if there is a matching row in the CSV
    details = data[data['name'] == name]
    
    if details.empty:
        return {
            'error': "The uploaded image does not match any known vaccines."
        }

    details = details.iloc[0]
    return {
        'name': details['name'],
        'age_group': details['age_group'],
        'purpose': details['purpose'],
        'disadvantages': details['disadvantages'],
    }
