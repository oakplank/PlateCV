#!/usr/bin/env python3
"""
Debug script to test YOLO model inference directly
This helps diagnose issues with license plate detection
"""

import sys
from pathlib import Path
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

def debug_model():
    """Debug the YOLO model loading and inference"""
    
    print("🔧 === YOLO MODEL DEBUG ===")
    
    # Check model path
    model_path = Path('runs/train/weights/best.pt')
    print(f"📁 Model path: {model_path}")
    print(f"📦 Model exists: {model_path.exists()}")
    
    if not model_path.exists():
        print("❌ Model file not found!")
        print("Make sure you have trained the model first.")
        return False
    
    # Load model
    print(f"🚀 Loading YOLO model...")
    try:
        model = YOLO(str(model_path))
        print(f"✅ Model loaded successfully!")
        
        # Print model info
        print(f"🏷️  Model names/classes: {model.names}")
        print(f"💾 Model device: {model.device}")
        
        # Check CUDA
        print(f"🖥️  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # Test with a sample image from dataset
    print(f"\n🖼️  === TESTING WITH SAMPLE IMAGE ===")
    
    # Look for a sample image
    sample_paths = [
        'dataset/val/images',
        'dataset/train/images',
        'dataset/images'
    ]
    
    sample_image = None
    for sample_path in sample_paths:
        sample_dir = Path(sample_path)
        if sample_dir.exists():
            # Get first image
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                images = list(sample_dir.glob(ext))
                if images:
                    sample_image = images[0]
                    break
            if sample_image:
                break
    
    if sample_image is None:
        print("❌ No sample images found in dataset directories")
        return False
        
    print(f"📎 Using sample image: {sample_image}")
    
    # Test inference
    try:
        print(f"🚀 Running inference on sample image...")
        
        # Test with different confidence thresholds
        confidence_levels = [0.1, 0.3, 0.5, 0.7]
        
        for conf in confidence_levels:
            print(f"\n🎯 Testing with confidence threshold: {conf}")
            
            results = model.predict(
                source=str(sample_image),
                conf=conf,
                verbose=True,
                save=False,
                show=False
            )
            
            if results and len(results) > 0:
                result = results[0]
                
                if hasattr(result, 'boxes') and result.boxes is not None:
                    num_detections = len(result.boxes)
                    print(f"   📦 Detections found: {num_detections}")
                    
                    for i, box in enumerate(result.boxes):
                        confidence = box.conf[0].cpu().numpy()
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        print(f"   📍 Detection {i+1}: conf={confidence:.4f}, bbox=({x1},{y1},{x2},{y2})")
                else:
                    print(f"   ❌ No boxes in result")
            else:
                print(f"   ❌ No results returned")
                
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n✅ Model debug complete!")
    return True

if __name__ == '__main__':
    debug_model() 