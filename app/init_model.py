import os
from django.conf import settings

def initialize_model_directories():
    """Create necessary directories for model training and prediction"""
    
    # Create directories
    directories = [
        os.path.join(settings.BASE_DIR, 'app', 'train_models'),
        os.path.join(settings.BASE_DIR, 'media', 'medicine_images'),
        os.path.join(settings.BASE_DIR, 'app', 'static', 'medicine_images')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

if __name__ == "__main__":
    initialize_model_directories() 