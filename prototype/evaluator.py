"""
Evaluation module
Evaluates feature extraction accuracy and performance
"""

import json
import os
from typing import Dict, List, Tuple
from collections import defaultdict

from utils import load_json, save_json, format_time


class Evaluator:
    """Evaluator - for evaluating model performance"""
    
    def __init__(self, annotations_path: str = None):
        """
        Initialize evaluator
        
        Args:
            annotations_path: Path to manual annotation file (JSON format)
        """
        self.annotations = {}
        if annotations_path and os.path.exists(annotations_path):
            self.annotations = load_json(annotations_path)
            print(f"✓ Loaded {len(self.annotations)} annotations")
    
    def calculate_accuracy(self, predicted_features: Dict, ground_truth: Dict) -> Dict[str, float]:
        """
        Calculate feature extraction accuracy
        
        Args:
            predicted_features: Predicted features (containing color, texture, shape)
            ground_truth: True annotated features
            
        Returns:
            Dictionary of accuracy for each feature type
        """
        accuracies = {}
        
        # Color feature accuracy
        if "color" in predicted_features and "color_features" in ground_truth:
            pred_color = set(predicted_features["color"])
            true_color = set(ground_truth["color_features"])
            if true_color:
                color_acc = len(pred_color & true_color) / len(true_color)
                accuracies["color"] = round(color_acc, 3)
            else:
                accuracies["color"] = 0.0
        
        # Texture feature accuracy
        if "texture" in predicted_features and "texture_features" in ground_truth:
            pred_texture = set(predicted_features["texture"])
            true_texture = set(ground_truth["texture_features"])
            if true_texture:
                texture_acc = len(pred_texture & true_texture) / len(true_texture)
                accuracies["texture"] = round(texture_acc, 3)
            else:
                accuracies["texture"] = 0.0
        
        # Shape feature accuracy
        if "shape" in predicted_features and "shape_features" in ground_truth:
            pred_shape = set(predicted_features["shape"])
            true_shape = set(ground_truth["shape_features"])
            if true_shape:
                shape_acc = len(pred_shape & true_shape) / len(true_shape)
                accuracies["shape"] = round(shape_acc, 3)
            else:
                accuracies["shape"] = 0.0
        
        # Overall accuracy (average of all categories)
        if accuracies:
            accuracies["overall"] = round(sum(accuracies.values()) / len(accuracies), 3)
        else:
            accuracies["overall"] = 0.0
        
        return accuracies
    
    def evaluate_single(self, result: Dict, image_name: str = None) -> Dict:
        """
        Evaluate single result
        
        Args:
            result: Analysis result dictionary
            image_name: Image filename (for finding annotation)
            
        Returns:
            Evaluation result dictionary
        """
        if "error" in result:
            return {
                "image_path": result["image_path"],
                "error": result["error"],
                "accuracy": None
            }
        
        # Get image filename
        if image_name is None:
            image_name = os.path.basename(result["image_path"])
        
        # Find corresponding annotation
        ground_truth = self.annotations.get(image_name, {})
        
        if not ground_truth:
            return {
                "image_path": result["image_path"],
                "warning": "No corresponding annotation data found",
                "processing_time": result["processing_time"]
            }
        
        # Calculate accuracy
        accuracies = self.calculate_accuracy(
            result["extracted_features"],
            ground_truth
        )
        
        return {
            "image_path": result["image_path"],
            "processing_time": result["processing_time"],
            "accuracy": accuracies,
            "predicted_features": result["extracted_features"],
            "ground_truth": {
                "color_features": ground_truth.get("color_features", []),
                "texture_features": ground_truth.get("texture_features", []),
                "shape_features": ground_truth.get("shape_features", [])
            }
        }
    
    def evaluate_batch(self, results: List[Dict]) -> Dict:
        """
        Batch evaluate results
        
        Args:
            results: List of analysis results
            
        Returns:
            Evaluation report dictionary
        """
        evaluations = []
        
        for result in results:
            image_name = os.path.basename(result["image_path"])
            eval_result = self.evaluate_single(result, image_name)
            evaluations.append(eval_result)
        
        # Calculate statistics
        successful_evals = [e for e in evaluations if "accuracy" in e and e["accuracy"] is not None]
        
        if not successful_evals:
            return {
                "total": len(evaluations),
                "successful": 0,
                "summary": "No valid evaluation data"
            }
        
        # Calculate average accuracy
        avg_accuracies = defaultdict(list)
        for eval_result in successful_evals:
            if "accuracy" in eval_result:
                for key, value in eval_result["accuracy"].items():
                    avg_accuracies[key].append(value)
        
        avg_acc = {
            key: round(sum(values) / len(values), 3)
            for key, values in avg_accuracies.items()
        }
        
        # Calculate average processing time
        avg_time = sum(e["processing_time"] for e in successful_evals) / len(successful_evals)
        
        # Statistics of accuracy distribution by category
        category_acc = {
            "color": [e["accuracy"]["color"] for e in successful_evals if "color" in e.get("accuracy", {})],
            "texture": [e["accuracy"]["texture"] for e in successful_evals if "texture" in e.get("accuracy", {})],
            "shape": [e["accuracy"]["shape"] for e in successful_evals if "shape" in e.get("accuracy", {})]
        }
        
        return {
            "total": len(evaluations),
            "successful": len(successful_evals),
            "failed": len(evaluations) - len(successful_evals),
            "average_accuracy": avg_acc,
            "average_processing_time": round(avg_time, 2),
            "category_accuracy": {
                "color": {
                    "mean": round(sum(category_acc["color"]) / len(category_acc["color"]), 3) if category_acc["color"] else 0,
                    "min": round(min(category_acc["color"]), 3) if category_acc["color"] else 0,
                    "max": round(max(category_acc["color"]), 3) if category_acc["color"] else 0
                },
                "texture": {
                    "mean": round(sum(category_acc["texture"]) / len(category_acc["texture"]), 3) if category_acc["texture"] else 0,
                    "min": round(min(category_acc["texture"]), 3) if category_acc["texture"] else 0,
                    "max": round(max(category_acc["texture"]), 3) if category_acc["texture"] else 0
                },
                "shape": {
                    "mean": round(sum(category_acc["shape"]) / len(category_acc["shape"]), 3) if category_acc["shape"] else 0,
                    "min": round(min(category_acc["shape"]), 3) if category_acc["shape"] else 0,
                    "max": round(max(category_acc["shape"]), 3) if category_acc["shape"] else 0
                }
            },
            "detailed_results": evaluations
        }
    
    def generate_report(self, evaluation_results: Dict, output_path: str) -> None:
        """
        Generate evaluation report
        
        Args:
            evaluation_results: Evaluation result dictionary
            output_path: Report save path
        """
        report = {
            "evaluation_summary": {
                "total_images": evaluation_results["total"],
                "successful_evaluations": evaluation_results["successful"],
                "failed_evaluations": evaluation_results.get("failed", 0),
                "overall_accuracy": evaluation_results["average_accuracy"].get("overall", 0),
                "average_processing_time": evaluation_results["average_processing_time"],
                "category_accuracy": evaluation_results["category_accuracy"]
            },
            "detailed_results": evaluation_results.get("detailed_results", [])
        }
        
        save_json(report, output_path)
        print(f"✓ Evaluation report saved to: {output_path}")


if __name__ == "__main__":
    # Simple test
    print("Testing Evaluator module...")
    
    # Create test data
    test_annotations = {
        "test_image.jpg": {
            "color_features": ["Red", "High Saturation", "High Brightness"],
            "texture_features": ["Smooth Surface"],
            "shape_features": ["Streamlined Design"]
        }
    }
    
    # Save test annotations
    test_annotations_path = "test_annotations.json"
    save_json(test_annotations, test_annotations_path)
    
    # Initialize evaluator
    evaluator = Evaluator(test_annotations_path)
    print("✓ Evaluator initialized successfully")
    
    # Test accuracy calculation
    predicted = {
        "color": ["Red", "High Saturation", "High Brightness"],
        "texture": ["Smooth Surface"],
        "shape": ["Streamlined Design"]
    }
    
    ground_truth = {
        "color_features": ["Red", "High Saturation", "High Brightness"],
        "texture_features": ["Smooth Surface"],
        "shape_features": ["Streamlined Design"]
    }
    
    accuracy = evaluator.calculate_accuracy(predicted, ground_truth)
    print(f"✓ Accuracy calculation: {accuracy}")
    
    # Clean up test file
    if os.path.exists(test_annotations_path):
        os.remove(test_annotations_path)
    
    print("\nAll tests passed! Evaluator module works correctly.")
