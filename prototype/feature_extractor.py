"""
Feature extraction module
Uses MobileNet V2 to extract image features and performs color, texture, and shape analysis
"""

import os

# Suppress noisy TensorFlow / absl logs.
# Must be set before importing tensorflow.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # 0: all logs, 3: errors only
os.environ.setdefault("ABSL_LOGGING_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # disable GPU detection
# Reduce oneDNN verbose info
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tempfile

import numpy as np
import cv2
from typing import Dict, List, Tuple

# TensorFlow import can emit noisy logs to stderr (often bypassing Python sys.stderr).
# Redirect the OS-level stderr (fd=2) during import so the console stays clean.
_stderr_fd = os.dup(2)
_tmp_stderr = tempfile.TemporaryFile(mode="w+b")
os.dup2(_tmp_stderr.fileno(), 2)
try:
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from tensorflow.keras.models import Model
finally:
    os.dup2(_stderr_fd, 2)
    os.close(_stderr_fd)
    _tmp_stderr.close()


class FeatureExtractor:
    """Image feature extractor"""
    
    def __init__(self):
        """Initialize feature extractor, load MobileNet V2 model"""
        print("Loading MobileNet V2 model...")
        # Load pre-trained MobileNet V2 model (ImageNet weights)
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Use intermediate layer (layer_7_expand_relu) to extract features
        # This layer contains rich feature information
        self.model = Model(
            inputs=base_model.input,
            outputs=base_model.get_layer('block_7_expand_relu').output
        )
        
        # Freeze model parameters (no training)
        self.model.trainable = False
        print("✓ MobileNet V2 model loaded")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image: resize to 224x224, normalize
        
        Args:
            image: Original image (numpy array, RGB format)
            
        Returns:
            Preprocessed image
        """
        # Resize to 224x224 (MobileNet input size)
        resized = cv2.resize(image, (224, 224))
        
        # Convert to float32 and normalize (MobileNet preprocessing)
        preprocessed = preprocess_input(resized.astype(np.float32))
        
        return preprocessed
    
    def extract_mobilenet_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract feature vector using MobileNet
        
        Args:
            image: Preprocessed image
            
        Returns:
            Feature vector (numpy array)
        """
        # Add batch dimension
        image_batch = np.expand_dims(image, axis=0)
        
        # Extract features
        features = self.model.predict(image_batch, verbose=0)
        
        # Remove batch dimension and flatten
        return features.flatten()
    
    def analyze_color(self, image: np.ndarray) -> Dict[str, any]:
        """
        Analyze image color features
        
        Args:
            image: Original image (RGB format)
            
        Returns:
            Color feature dictionary
        """
        # Convert to HSV color space (better for color analysis)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Calculate color histograms
        hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])  # Hue
        hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])  # Saturation
        hist_v = cv2.calcHist([hsv], [2], None, [256], [0, 256])  # Value (brightness)
        
        # Find dominant hue
        dominant_hue = int(np.argmax(hist_h))
        
        # Calculate mean saturation
        mean_saturation = float(np.mean(hist_s))
        
        # Calculate mean brightness
        mean_brightness = float(np.mean(hist_v))
        
        # Determine hue type
        if 0 <= dominant_hue < 15 or 165 <= dominant_hue < 180:
            hue_type = "Red"
        elif 15 <= dominant_hue < 30:
            hue_type = "Orange"
        elif 30 <= dominant_hue < 60:
            hue_type = "Yellow"
        elif 60 <= dominant_hue < 90:
            hue_type = "Green"
        elif 90 <= dominant_hue < 120:
            hue_type = "Cyan"
        elif 120 <= dominant_hue < 150:
            hue_type = "Blue"
        else:
            hue_type = "Purple"
        
        # Determine saturation level
        if mean_saturation > 200:
            saturation_level = "High Saturation"
        elif mean_saturation > 100:
            saturation_level = "Medium Saturation"
        else:
            saturation_level = "Low Saturation"
        
        # Determine brightness level
        if mean_brightness > 200:
            brightness_level = "High Brightness"
        elif mean_brightness > 100:
            brightness_level = "Medium Brightness"
        else:
            brightness_level = "Low Brightness"
        
        return {
            "dominant_hue": dominant_hue,
            "hue_type": hue_type,
            "mean_saturation": mean_saturation,
            "saturation_level": saturation_level,
            "mean_brightness": mean_brightness,
            "brightness_level": brightness_level,
            "color_features": [hue_type, saturation_level, brightness_level]
        }
    
    def analyze_texture(self, image: np.ndarray) -> Dict[str, any]:
        """
        Analyze image texture features
        
        Args:
            image: Original image (RGB format)
            
        Returns:
            Texture feature dictionary
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Use LBP (Local Binary Pattern) to analyze texture
        # Simplified version: use standard deviation to measure texture complexity
        texture_std = float(np.std(gray))
        
        # Use Laplacian operator to detect edges (another way to measure texture)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        edge_variance = float(np.var(laplacian))
        
        # Determine texture type
        if texture_std < 20:
            texture_type = "Smooth Surface"
        elif texture_std < 40:
            texture_type = "Light Texture"
        elif texture_std < 60:
            texture_type = "Medium Texture"
        else:
            texture_type = "Rough Texture"
        
        return {
            "texture_std": texture_std,
            "edge_variance": edge_variance,
            "texture_type": texture_type,
            "texture_features": [texture_type]
        }
    
    def analyze_shape(self, image: np.ndarray) -> Dict[str, any]:
        """
        Analyze image shape features
        
        Args:
            image: Original image (RGB format)
            
        Returns:
            Shape feature dictionary
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Binarize
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return {
                "shape_type": "Unrecognized",
                "shape_features": ["Unrecognized"]
            }
        
        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate convex hull of contour
        hull = cv2.convexHull(largest_contour)
        
        # Calculate contour area and hull area
        contour_area = cv2.contourArea(largest_contour)
        hull_area = cv2.contourArea(hull)
        
        # Calculate compactness (contour area / hull area)
        compactness = contour_area / hull_area if hull_area > 0 else 0
        
        # Calculate aspect ratio
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = w / h if h > 0 else 1
        
        # Determine shape type
        if compactness > 0.9:
            if 0.9 <= aspect_ratio <= 1.1:
                shape_type = "Circular Design"
            elif aspect_ratio > 1.2:
                shape_type = "Horizontal Rectangle"
            else:
                shape_type = "Vertical Rectangle"
        elif compactness > 0.7:
            shape_type = "Rounded Design"
        else:
            # Analyze contour complexity
            perimeter = cv2.arcLength(largest_contour, True)
            if perimeter / contour_area > 0.1:
                shape_type = "Streamlined Design"
            else:
                shape_type = "Geometric Design"
        
        return {
            "compactness": float(compactness),
            "aspect_ratio": float(aspect_ratio),
            "shape_type": shape_type,
            "shape_features": [shape_type]
        }
    
    def extract_all_features(self, image: np.ndarray) -> Dict[str, any]:
        """
        Extract all features from image
        
        Args:
            image: Original image (numpy array, RGB format)
            
        Returns:
            Dictionary containing all features
        """
        # Preprocess image
        preprocessed = self.preprocess_image(image)
        
        # Extract MobileNet features
        mobilenet_features = self.extract_mobilenet_features(preprocessed)
        
        # Analyze color
        color_features = self.analyze_color(image)
        
        # Analyze texture
        texture_features = self.analyze_texture(image)
        
        # Analyze shape
        shape_features = self.analyze_shape(image)
        
        return {
            "mobilenet_features": mobilenet_features.tolist(),  # Convert to list for JSON serialization
            "color": color_features,
            "texture": texture_features,
            "shape": shape_features
        }


if __name__ == "__main__":
    # Simple test
    print("Testing FeatureExtractor module...")
    
    try:
        extractor = FeatureExtractor()
        print("✓ Feature extractor initialized successfully")
        
        # Create a test image (random RGB image)
        test_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        print("✓ Test image created successfully")
        
        # Extract features
        features = extractor.extract_all_features(test_image)
        print("✓ Feature extraction successful")
        print(f"  - Color features: {features['color']['color_features']}")
        print(f"  - Texture features: {features['texture']['texture_features']}")
        print(f"  - Shape features: {features['shape']['shape_features']}")
        print(f"  - MobileNet feature dimension: {len(features['mobilenet_features'])}")
        
        print("\nAll tests passed! FeatureExtractor module works correctly.")
        
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
