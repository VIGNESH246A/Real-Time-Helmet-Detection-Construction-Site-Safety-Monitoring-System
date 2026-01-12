"""
Training script for helmet detection model
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_helmet_detector(
    data_yaml: str = "data/helmet_dataset.yaml",
    model_size: str = "n",
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = "0",
    project: str = "runs/train",
    name: str = "helmet_detector"
):
    """
    Train YOLO model for helmet detection
    
    Args:
        data_yaml: Path to dataset YAML file
        model_size: Model size (n, s, m, l, x)
        epochs: Number of training epochs
        batch_size: Batch size
        img_size: Input image size
        device: Device to train on (cuda device id or 'cpu')
        project: Project directory
        name: Experiment name
    """
    
    logger.info("=" * 70)
    logger.info("Starting Helmet Detection Model Training")
    logger.info("=" * 70)
    
    # Load pre-trained YOLO model
    model_name = f"yolov8{model_size}.pt"
    logger.info(f"Loading model: {model_name}")
    model = YOLO(model_name)
    
    # Training configuration
    logger.info("Training Configuration:")
    logger.info(f"  Data: {data_yaml}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch Size: {batch_size}")
    logger.info(f"  Image Size: {img_size}")
    logger.info(f"  Device: {device}")
    
    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=project,
        name=name,
        pretrained=True,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        pose=12.0,
        kobj=1.0,
        label_smoothing=0.0,
        nbs=64,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        verbose=True,
        save=True,
        save_period=-1,
        cache=False,
        plots=True,
        overlap_mask=True,
        mask_ratio=4,
    )
    
    logger.info("=" * 70)
    logger.info("Training completed!")
    logger.info("=" * 70)
    logger.info(f"Best model saved to: {Path(project) / name / 'weights' / 'best.pt'}")
    
    return results


def create_dataset_yaml():
    """Create example dataset YAML configuration"""
    
    yaml_content = """# Helmet Detection Dataset Configuration

# Dataset paths (absolute or relative to this file)
path: ../data/training  # Dataset root directory
train: images/train     # Train images (relative to 'path')
val: images/val         # Val images (relative to 'path')
test: images/test       # Test images (optional)

# Classes
names:
  0: helmet
  1: no-helmet
  2: person

# Number of classes
nc: 3

# Dataset info (optional)
download: false  # Set to URL if dataset needs to be downloaded
"""
    
    output_path = Path("data/helmet_dataset.yaml")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(yaml_content)
    
    logger.info(f"Example dataset YAML created: {output_path}")
    logger.info("Please update the paths and configure according to your dataset")


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description='Train Helmet Detection Model')
    
    parser.add_argument('--data', type=str, default='data/helmet_dataset.yaml',
                        help='Path to dataset YAML file')
    parser.add_argument('--model', type=str, default='n',
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='Model size (n=nano, s=small, m=medium, l=large, x=extra-large)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Input image size')
    parser.add_argument('--device', type=str, default='0',
                        help='Device to train on (cuda device id or cpu)')
    parser.add_argument('--project', type=str, default='runs/train',
                        help='Project directory')
    parser.add_argument('--name', type=str, default='helmet_detector',
                        help='Experiment name')
    parser.add_argument('--create-yaml', action='store_true',
                        help='Create example dataset YAML and exit')
    
    args = parser.parse_args()
    
    # Create example YAML if requested
    if args.create_yaml:
        create_dataset_yaml()
        return
    
    # Check if data file exists
    if not Path(args.data).exists():
        logger.error(f"Dataset YAML not found: {args.data}")
        logger.info("Run with --create-yaml to create an example configuration")
        return
    
    # Train model
    train_helmet_detector(
        data_yaml=args.data,
        model_size=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size,
        device=args.device,
        project=args.project,
        name=args.name
    )


if __name__ == "__main__":
    main()