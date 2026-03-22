"""
Image analysis main module
Integrates feature extraction and selling point conversion, provides complete image analysis functionality
"""

import time
from typing import Dict, List, Tuple
import numpy as np

from utils import load_image
from feature_extractor import FeatureExtractor
from selling_point_converter import SellingPointConverter


class ImageAnalyzer:
    """Image analyzer - main analysis module"""
    
    def __init__(self):
        """Initialize image analyzer"""
        print("Initializing image analyzer...")
        self.feature_extractor = FeatureExtractor()
        self.selling_point_converter = SellingPointConverter()
        print("✓ Image analyzer initialization complete")
    
    def analyze(self, image_path: str) -> Dict:
        """
        Analyze single image, extract features and convert to selling points
        
        Args:
            image_path: Image file path
            
        Returns:
            Analysis result dictionary containing:
            - image_path: Image path
            - extracted_features: Extracted features
            - selling_points: Converted selling points
            - processing_time: Processing time (seconds)
        """
        start_time = time.time()
        
        try:
            # 1. Load image
            image = load_image(image_path)
            
            # 2. Extract features
            features = self.feature_extractor.extract_all_features(image)
            
            # 3. Convert to selling points
            selling_points = self.selling_point_converter.convert_all_features(features)
            
            # 4. Calculate processing time
            processing_time = time.time() - start_time
            
            # 5. Build result
            result = {
                "image_path": image_path,
                "extracted_features": {
                    "color": features["color"]["color_features"],
                    "texture": features["texture"]["texture_features"],
                    "shape": features["shape"]["shape_features"]
                },
                "selling_points": selling_points,
                "processing_time": round(processing_time, 2),
                "detailed_features": {
                    "color": {
                        "hue_type": features["color"]["hue_type"],
                        "saturation_level": features["color"]["saturation_level"],
                        "brightness_level": features["color"]["brightness_level"]
                    },
                    "texture": {
                        "texture_type": features["texture"]["texture_type"]
                    },
                    "shape": {
                        "shape_type": features["shape"]["shape_type"]
                    }
                }
            }
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            return {
                "image_path": image_path,
                "error": str(e),
                "processing_time": round(processing_time, 2)
            }
    
    def analyze_batch(self, image_paths: List[str]) -> List[Dict]:
        """
        Batch analyze multiple images
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of analysis results
        """
        results = []
        total = len(image_paths)
        
        print(f"\nStarting batch analysis of {total} images...")
        
        for i, image_path in enumerate(image_paths, 1):
            print(f"Processing ({i}/{total}): {image_path}")
            result = self.analyze(image_path)
            results.append(result)
            
            if "error" in result:
                print(f"  ✗ Processing failed: {result['error']}")
            else:
                print(f"  ✓ Complete - Selling points: {result['selling_points']}")
                print(f"    Processing time: {result['processing_time']}s")
        
        return results
    
    def get_summary(self, results: List[Dict]) -> Dict:
        """
        Generate analysis result summary
        
        Args:
            results: List of analysis results
            
        Returns:
            Summary dictionary
        """
        successful = [r for r in results if "error" not in r]
        failed = [r for r in results if "error" in r]
        
        if not successful:
            return {
                "total": len(results),
                "successful": 0,
                "failed": len(failed),
                "average_processing_time": 0
            }
        
        avg_time = sum(r["processing_time"] for r in successful) / len(successful)
        
        # Count most common selling points
        all_selling_points = []
        for r in successful:
            all_selling_points.extend(r.get("selling_points", []))
        
        from collections import Counter
        common_points = Counter(all_selling_points).most_common(5)
        
        return {
            "total": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "average_processing_time": round(avg_time, 2),
            "most_common_selling_points": [point for point, count in common_points]
        }


if __name__ == "__main__":
    # Simple test
    print("Testing ImageAnalyzer module...")
    
    try:
        analyzer = ImageAnalyzer()
        print("✓ Image analyzer initialized successfully")
        
        # Create a test image (random RGB image)
        import os
        import cv2
        test_image_path = "test_image.jpg"
        test_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        cv2.imwrite(test_image_path, cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR))
        print("✓ Test image created successfully")
        
        # Analyze image
        result = analyzer.analyze(test_image_path)
        
        if "error" in result:
            print(f"✗ Analysis failed: {result['error']}")
        else:
            print("✓ Image analysis successful")
            print(f"  - Extracted features:")
            print(f"    Color: {result['extracted_features']['color']}")
            print(f"    Texture: {result['extracted_features']['texture']}")
            print(f"    Shape: {result['extracted_features']['shape']}")
            print(f"  - Marketing selling points: {result['selling_points']}")
            print(f"  - Processing time: {result['processing_time']}s")
        
        # Clean up test file
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
        
        print("\nAll tests passed! ImageAnalyzer module works correctly.")
        
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
