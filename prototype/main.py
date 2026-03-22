"""
Main program entry
Provides command-line interface, supports single image analysis and batch analysis
"""

import argparse
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import get_image_files, save_json, format_time, ensure_dir
from image_analyzer import ImageAnalyzer
from evaluator import Evaluator


def analyze_single_image(image_path: str, output_dir: str = None):
    """
    Analyze single image
    
    Args:
        image_path: Image file path
        output_dir: Result output directory (optional)
    """
    print(f"\n{'='*60}")
    print(f"Analyzing image: {image_path}")
    print(f"{'='*60}\n")
    
    analyzer = ImageAnalyzer()
    result = analyzer.analyze(image_path)
    
    if "error" in result:
        print(f"✗ Analysis failed: {result['error']}")
        return result
    
    # Display results
    print("✓ Analysis complete")
    print(f"\nExtracted features:")
    print(f"  Color: {', '.join(result['extracted_features']['color'])}")
    print(f"  Texture: {', '.join(result['extracted_features']['texture'])}")
    print(f"  Shape: {', '.join(result['extracted_features']['shape'])}")
    print(f"\nMarketing selling points:")
    for i, point in enumerate(result['selling_points'], 1):
        print(f"  {i}. {point}")
    print(f"\nProcessing time: {format_time(result['processing_time'])}")
    
    # Save results
    if output_dir:
        ensure_dir(output_dir)
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{image_name}_result.json")
        save_json(result, output_path)
        print(f"\nResults saved to: {output_path}")
    
    return result


def analyze_batch(image_dir: str, output_dir: str = None, annotations_path: str = None):
    """
    Batch analyze images
    
    Args:
        image_dir: Image directory path
        output_dir: Result output directory (optional)
        annotations_path: Annotation file path (optional, for evaluation)
    """
    print(f"\n{'='*60}")
    print(f"Batch analyzing image directory: {image_dir}")
    print(f"{'='*60}\n")
    
    # Get all image files
    image_files = get_image_files(image_dir)
    
    if not image_files:
        print(f"✗ No image files found in directory {image_dir}")
        return
    
    print(f"Found {len(image_files)} images\n")
    
    # Analyze images
    analyzer = ImageAnalyzer()
    results = analyzer.analyze_batch(image_files)
    
    # Generate summary
    summary = analyzer.get_summary(results)
    print(f"\n{'='*60}")
    print("Analysis Summary")
    print(f"{'='*60}")
    print(f"Total: {summary['total']} images")
    print(f"Successful: {summary['successful']} images")
    print(f"Failed: {summary['failed']} images")
    print(f"Average processing time: {format_time(summary['average_processing_time'])}")
    
    if summary.get('most_common_selling_points'):
        print(f"\nMost common selling points:")
        for i, point in enumerate(summary['most_common_selling_points'], 1):
            print(f"  {i}. {point}")
    
    # Save results
    if output_dir:
        ensure_dir(output_dir)
        
        # Save detailed results
        results_path = os.path.join(output_dir, "analysis_results.json")
        save_json(results, results_path)
        print(f"\nDetailed results saved to: {results_path}")
        
        # Save summary
        summary_path = os.path.join(output_dir, "analysis_summary.json")
        save_json(summary, summary_path)
        print(f"Analysis summary saved to: {summary_path}")
        
        # If annotation file exists, perform evaluation
        if annotations_path and os.path.exists(annotations_path):
            print(f"\nStarting evaluation...")
            evaluator = Evaluator(annotations_path)
            evaluation_results = evaluator.evaluate_batch(results)
            
            # Display evaluation results
            print(f"\n{'='*60}")
            print("Evaluation Results")
            print(f"{'='*60}")
            print(f"Successful evaluations: {evaluation_results['successful']}/{evaluation_results['total']}")
            
            if "average_accuracy" in evaluation_results:
                acc = evaluation_results["average_accuracy"]
                print(f"\nAverage accuracy:")
                print(f"  Overall: {acc.get('overall', 0):.1%}")
                print(f"  Color: {acc.get('color', 0):.1%}")
                print(f"  Texture: {acc.get('texture', 0):.1%}")
                print(f"  Shape: {acc.get('shape', 0):.1%}")
            
            # Save evaluation report
            report_path = os.path.join(output_dir, "evaluation_report.json")
            evaluator.generate_report(evaluation_results, report_path)
    
    print(f"\n{'='*60}\n")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="MobileNet image feature extraction prototype - Extract marketing selling points from product images"
    )
    
    parser.add_argument(
        "input",
        help="Input image file path or image directory path"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Result output directory (optional)",
        default=None
    )
    
    parser.add_argument(
        "-a", "--annotations",
        help="Annotation file path (for evaluation, optional)",
        default=None
    )
    
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch processing mode (input is directory)"
    )
    
    args = parser.parse_args()
    
    # Check input path
    if not os.path.exists(args.input):
        print(f"✗ Error: Path does not exist: {args.input}")
        return
    
    # Determine output directory
    if args.output:
        output_dir = args.output
    else:
        # Default output to results directory
        base_dir = Path(__file__).parent.parent
        output_dir = str(base_dir / "results")
    
    # Execute analysis
    if args.batch or os.path.isdir(args.input):
        analyze_batch(args.input, output_dir, args.annotations)
    else:
        analyze_single_image(args.input, output_dir)


if __name__ == "__main__":
    main()
