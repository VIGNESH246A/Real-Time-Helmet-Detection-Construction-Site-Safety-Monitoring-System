"""
Model evaluation and testing script
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import logging
import cv2
import numpy as np
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluate helmet detection model performance
    """
    
    def __init__(self, model_path: str):
        """
        Initialize evaluator
        
        Args:
            model_path: Path to model weights
        """
        self.model_path = model_path
        self.model = YOLO(model_path)
        logger.info(f"Model loaded: {model_path}")
    
    def evaluate_on_dataset(
        self,
        data_yaml: str,
        split: str = "val",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ) -> Dict:
        """
        Evaluate model on dataset
        
        Args:
            data_yaml: Path to dataset YAML
            split: Dataset split to evaluate ('val' or 'test')
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            
        Returns:
            Evaluation metrics
        """
        logger.info(f"Evaluating on {split} split...")
        
        results = self.model.val(
            data=data_yaml,
            split=split,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=True
        )
        
        # Extract metrics
        metrics = {
            'mAP50': results.box.map50,
            'mAP50-95': results.box.map,
            'precision': results.box.p.mean(),
            'recall': results.box.r.mean(),
            'f1_score': 2 * (results.box.p.mean() * results.box.r.mean()) / 
                       (results.box.p.mean() + results.box.r.mean() + 1e-10)
        }
        
        logger.info("=" * 70)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 70)
        for key, value in metrics.items():
            logger.info(f"{key}: {value:.4f}")
        logger.info("=" * 70)
        
        return metrics
    
    def test_on_images(
        self,
        image_dir: str,
        output_dir: str = "outputs/evaluation",
        conf_threshold: float = 0.5
    ) -> List[Dict]:
        """
        Test model on individual images
        
        Args:
            image_dir: Directory containing test images
            output_dir: Output directory for annotated images
            conf_threshold: Confidence threshold
            
        Returns:
            List of results per image
        """
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = [
            f for f in image_dir.iterdir() 
            if f.suffix.lower() in image_extensions
        ]
        
        logger.info(f"Testing on {len(image_files)} images...")
        
        results_list = []
        
        for img_file in image_files:
            # Run inference
            results = self.model.predict(
                str(img_file),
                conf=conf_threshold,
                verbose=False
            )
            
            # Get annotated image
            annotated = results[0].plot()
            
            # Save
            output_path = output_dir / f"result_{img_file.name}"
            cv2.imwrite(str(output_path), annotated)
            
            # Extract detections
            boxes = results[0].boxes
            detections = []
            if boxes is not None:
                for box in boxes:
                    detections.append({
                        'class': int(box.cls[0]),
                        'confidence': float(box.conf[0]),
                        'bbox': box.xyxy[0].cpu().numpy().tolist()
                    })
            
            results_list.append({
                'image': str(img_file),
                'detections': detections,
                'num_detections': len(detections)
            })
            
            logger.info(f"Processed {img_file.name}: {len(detections)} detections")
        
        logger.info(f"Results saved to: {output_dir}")
        return results_list
    
    def benchmark_speed(
        self,
        image_size: int = 640,
        num_iterations: int = 100
    ) -> Dict:
        """
        Benchmark inference speed
        
        Args:
            image_size: Input image size
            num_iterations: Number of iterations
            
        Returns:
            Speed metrics
        """
        logger.info(f"Benchmarking speed ({num_iterations} iterations)...")
        
        # Create dummy image
        dummy_image = np.random.randint(
            0, 255, (image_size, image_size, 3), dtype=np.uint8
        )
        
        # Warmup
        for _ in range(10):
            self.model.predict(dummy_image, verbose=False)
        
        # Benchmark
        import time
        times = []
        
        for _ in range(num_iterations):
            start = time.time()
            self.model.predict(dummy_image, verbose=False)
            times.append(time.time() - start)
        
        times = np.array(times)
        
        metrics = {
            'mean_time': times.mean(),
            'std_time': times.std(),
            'min_time': times.min(),
            'max_time': times.max(),
            'fps': 1.0 / times.mean()
        }
        
        logger.info("=" * 70)
        logger.info("SPEED BENCHMARK")
        logger.info("=" * 70)
        logger.info(f"Mean inference time: {metrics['mean_time']*1000:.2f} ms")
        logger.info(f"Std deviation: {metrics['std_time']*1000:.2f} ms")
        logger.info(f"FPS: {metrics['fps']:.2f}")
        logger.info("=" * 70)
        
        return metrics
    
    def analyze_errors(
        self,
        data_yaml: str,
        output_dir: str = "outputs/error_analysis"
    ) -> None:
        """
        Analyze model errors and failure cases
        
        Args:
            data_yaml: Path to dataset YAML
            output_dir: Output directory
        """
        logger.info("Analyzing model errors...")
        
        # Run validation
        results = self.model.val(
            data=data_yaml,
            save_json=True,
            save_hybrid=True
        )
        
        logger.info("Error analysis complete")
        logger.info(f"Check outputs in model prediction directory")


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description='Evaluate Helmet Detection Model')
    
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model weights')
    parser.add_argument('--data', type=str,
                        help='Path to dataset YAML for evaluation')
    parser.add_argument('--images', type=str,
                        help='Directory of images for testing')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run speed benchmark')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold')
    parser.add_argument('--output', type=str, default='outputs/evaluation',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        logger.error(f"Model not found: {args.model}")
        return
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.model)
    
    # Run evaluations
    if args.data:
        evaluator.evaluate_on_dataset(
            data_yaml=args.data,
            conf_threshold=args.conf
        )
    
    if args.images:
        evaluator.test_on_images(
            image_dir=args.images,
            output_dir=args.output,
            conf_threshold=args.conf
        )
    
    if args.benchmark:
        evaluator.benchmark_speed()


if __name__ == "__main__":
    main()