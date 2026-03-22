"""
Utility functions module
Provides basic functions for image loading, saving, and path handling
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image
import numpy as np


def ensure_dir(dir_path: str) -> None:
    """Ensure directory exists, create if it doesn't"""
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def load_image(image_path: str) -> np.ndarray:
    """
    Load image file
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image as numpy array (RGB format)
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image format is not supported
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    try:
        img = Image.open(image_path)
        # Convert to RGB format (handle RGBA, etc.)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return np.array(img)
    except Exception as e:
        raise ValueError(f"Failed to load image {image_path}: {str(e)}")


def save_json(data: Dict[Any, Any], file_path: str) -> None:
    """
    Save data as JSON file
    
    Args:
        data: Data to save (dictionary format)
        file_path: Save path
    """
    ensure_dir(os.path.dirname(file_path))
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(file_path: str) -> Dict[Any, Any]:
    """
    Load JSON file
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Data in dictionary format
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_image_files(directory: str, extensions: List[str] = None) -> List[str]:
    """
    Get all image file paths in directory
    
    Args:
        directory: Directory path
        extensions: List of allowed file extensions, default ['.jpg', '.jpeg', '.png', '.bmp']
        
    Returns:
        List of image file paths
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']
    
    image_files = []
    if os.path.exists(directory):
        for file in os.listdir(directory):
            if any(file.endswith(ext) for ext in extensions):
                image_files.append(os.path.join(directory, file))
    
    return sorted(image_files)


def format_time(seconds: float) -> str:
    """
    Format time display
    
    Args:
        seconds: Number of seconds
        
    Returns:
        Formatted time string (e.g., "3.2s")
    """
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    else:
        return f"{seconds:.2f}s"


if __name__ == "__main__":
    # Simple test
    print("Testing utils.py module...")
    
    # Test directory creation
    test_dir = "test_output"
    ensure_dir(test_dir)
    print(f"✓ Directory creation works")
    
    # Test JSON save and load
    test_data = {"test": "data", "number": 123}
    test_json_path = os.path.join(test_dir, "test.json")
    save_json(test_data, test_json_path)
    loaded_data = load_json(test_json_path)
    assert loaded_data == test_data
    print(f"✓ JSON save and load works")
    
    # Test time formatting
    assert format_time(3.2) == "3.20s"
    assert format_time(0.5) == "500.0ms"
    print(f"✓ Time formatting works")
    
    # Clean up test files
    import shutil
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    print("\nAll tests passed! utils.py module works correctly.")


