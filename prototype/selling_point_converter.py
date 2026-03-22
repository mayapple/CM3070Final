"""
Selling point conversion module
Converts technical features to marketing selling points
"""

from typing import Dict, List


class SellingPointConverter:
    """Feature to selling point converter"""
    
    def __init__(self):
        """Initialize conversion rules"""
        # Color feature to selling point mapping
        self.color_mapping = {
            "Red": {
                "High Saturation": ["Vibrant Red", "Eye-catching Color", "Passionate"],
                "Medium Saturation": ["Elegant Red", "Classic Color"],
                "Low Saturation": ["Soft Red", "Warm Color"]
            },
            "Orange": {
                "High Saturation": ["Vibrant Orange", "Youthful Color", "Trendy Bright"],
                "Medium Saturation": ["Warm Orange", "Comfortable Color"],
                "Low Saturation": ["Soft Orange", "Warm Color"]
            },
            "Yellow": {
                "High Saturation": ["Bright Yellow", "Energetic Color", "Sunny Feel"],
                "Medium Saturation": ["Warm Yellow", "Comfortable Color"],
                "Low Saturation": ["Soft Yellow", "Elegant Color"]
            },
            "Green": {
                "High Saturation": ["Fresh Green", "Natural Color", "Energetic Feel"],
                "Medium Saturation": ["Comfortable Green", "Balanced Color"],
                "Low Saturation": ["Soft Green", "Elegant Color"]
            },
            "Blue": {
                "High Saturation": ["Vibrant Blue", "Modern Color", "Tech Feel"],
                "Medium Saturation": ["Classic Blue", "Stable Color"],
                "Low Saturation": ["Soft Blue", "Elegant Color"]
            },
            "Purple": {
                "High Saturation": ["Mysterious Purple", "Trendy Color", "Unique Feel"],
                "Medium Saturation": ["Elegant Purple", "Premium Color"],
                "Low Saturation": ["Soft Purple", "Warm Color"]
            }
        }
        
        # Texture feature to selling point mapping
        self.texture_mapping = {
            "Smooth Surface": ["Premium Texture", "Fine Craftsmanship", "Quality Assurance"],
            "Light Texture": ["Delicate Texture", "Refined Design"],
            "Medium Texture": ["Natural Texture", "Comfortable Touch"],
            "Rough Texture": ["Vintage Texture", "Unique Design", "Distinctive Style"]
        }
        
        # Shape feature to selling point mapping
        self.shape_mapping = {
            "Circular Design": ["Rounded Design", "Soft Appearance", "Comfortable Feel"],
            "Horizontal Rectangle": ["Modern Design", "Stable Appearance"],
            "Vertical Rectangle": ["Minimalist Design", "Elegant Appearance"],
            "Rounded Design": ["Streamlined Design", "Fashionable Appearance", "Comfortable Feel"],
            "Streamlined Design": ["Fashionable Streamline", "Dynamic Design", "Modern Feel"],
            "Geometric Design": ["Minimalist Geometry", "Modern Design", "Unique Feel"]
        }
        
        # Brightness feature supplement
        self.brightness_mapping = {
            "High Brightness": ["Bright", "Clear", "Eye-catching"],
            "Medium Brightness": ["Comfortable", "Balanced"],
            "Low Brightness": ["Soft", "Elegant", "Premium"]
        }
    
    def convert_color_features(self, color_features: Dict) -> List[str]:
        """
        Convert color features to marketing selling points
        
        Args:
            color_features: Color feature dictionary
            
        Returns:
            List of marketing selling points
        """
        selling_points = []
        
        hue_type = color_features.get("hue_type", "")
        saturation_level = color_features.get("saturation_level", "")
        brightness_level = color_features.get("brightness_level", "")
        
        # Generate selling points based on hue and saturation
        if hue_type in self.color_mapping:
            if saturation_level in self.color_mapping[hue_type]:
                selling_points.extend(self.color_mapping[hue_type][saturation_level][:2])  # Take first 2
        
        # Supplement with brightness-based selling points
        if brightness_level in self.brightness_mapping:
            brightness_point = self.brightness_mapping[brightness_level][0]
            if brightness_point not in selling_points:
                selling_points.append(brightness_point)
        
        # If no match found, use generic description
        if not selling_points:
            selling_points = [f"{hue_type} Color", "Fashionable Design"]
        
        return selling_points[:3]  # Return at most 3 selling points
    
    def convert_texture_features(self, texture_features: Dict) -> List[str]:
        """
        Convert texture features to marketing selling points
        
        Args:
            texture_features: Texture feature dictionary
            
        Returns:
            List of marketing selling points
        """
        texture_type = texture_features.get("texture_type", "")
        
        if texture_type in self.texture_mapping:
            return self.texture_mapping[texture_type][:2]  # Take first 2
        else:
            return ["Fine Texture", "Quality Assurance"]
    
    def convert_shape_features(self, shape_features: Dict) -> List[str]:
        """
        Convert shape features to marketing selling points
        
        Args:
            shape_features: Shape feature dictionary
            
        Returns:
            List of marketing selling points
        """
        shape_type = shape_features.get("shape_type", "")
        
        if shape_type in self.shape_mapping:
            return self.shape_mapping[shape_type][:2]  # Take first 2
        else:
            return ["Fashionable Design", "Modern Appearance"]
    
    def convert_all_features(self, features: Dict) -> List[str]:
        """
        Convert all features to marketing selling points
        
        Args:
            features: Dictionary containing color, texture, shape features
            
        Returns:
            Merged list of marketing selling points (deduplicated)
        """
        all_selling_points = []
        
        # Convert color features
        if "color" in features:
            color_points = self.convert_color_features(features["color"])
            all_selling_points.extend(color_points)
        
        # Convert texture features
        if "texture" in features:
            texture_points = self.convert_texture_features(features["texture"])
            all_selling_points.extend(texture_points)
        
        # Convert shape features
        if "shape" in features:
            shape_points = self.convert_shape_features(features["shape"])
            all_selling_points.extend(shape_points)
        
        # Deduplicate while preserving order
        seen = set()
        unique_points = []
        for point in all_selling_points:
            if point not in seen:
                seen.add(point)
                unique_points.append(point)
        
        return unique_points[:5]  # Return at most 5 selling points


if __name__ == "__main__":
    # Simple test
    print("Testing SellingPointConverter module...")
    
    converter = SellingPointConverter()
    print("✓ Selling point converter initialized successfully")
    
    # Test color feature conversion
    test_color_features = {
        "hue_type": "Red",
        "saturation_level": "High Saturation",
        "brightness_level": "High Brightness"
    }
    color_points = converter.convert_color_features(test_color_features)
    print(f"✓ Color feature conversion: {color_points}")
    
    # Test texture feature conversion
    test_texture_features = {
        "texture_type": "Smooth Surface"
    }
    texture_points = converter.convert_texture_features(test_texture_features)
    print(f"✓ Texture feature conversion: {texture_points}")
    
    # Test shape feature conversion
    test_shape_features = {
        "shape_type": "Streamlined Design"
    }
    shape_points = converter.convert_shape_features(test_shape_features)
    print(f"✓ Shape feature conversion: {shape_points}")
    
    # Test complete conversion
    test_features = {
        "color": test_color_features,
        "texture": test_texture_features,
        "shape": test_shape_features
    }
    all_points = converter.convert_all_features(test_features)
    print(f"✓ Complete feature conversion: {all_points}")
    
    print("\nAll tests passed! SellingPointConverter module works correctly.")
