import os
import io
import base64
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
import torch

app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
UPLOAD_FOLDER = Path('uploads')
UPLOAD_FOLDER.mkdir(exist_ok=True)

# Global model variable
model = None

def load_model():
    """Load the trained YOLO model"""
    global model
    try:
        # Path to the trained model
        model_path = Path(__file__).parent.parent / 'runs' / 'train' / 'weights' / 'best.pt'
        
        print(f"🔍 Looking for model at: {model_path}")
        print(f"📁 Model file exists: {model_path.exists()}")
        
        if not model_path.exists():
            print(f"❌ Error: Model file not found at {model_path}")
            return False
            
        print(f"📦 Loading YOLO model from {model_path}")
        model = YOLO(str(model_path))
        
        # Print model information
        print(f"✅ Model loaded successfully!")
        print(f"🏷️  Model type: {type(model)}")
        print(f"📊 Model names: {model.names}")
        print(f"💾 Model device: {model.device}")
        
        # Check if CUDA is available
        device = '0' if torch.cuda.is_available() else 'cpu'
        print(f"🖥️  Inference device: {device}")
        if torch.cuda.is_available():
            print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
        
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

def process_image(image):
    """Process image with YOLO model and return image with bounding boxes"""
    try:
        print(f"\n🖼️  === STARTING IMAGE PROCESSING ===")
        
        # Convert PIL image to numpy array
        img_array = np.array(image)
        print(f"📐 Original image shape: {img_array.shape}")
        print(f"🎨 Original image dtype: {img_array.dtype}")
        
        # Convert RGB to BGR for OpenCV
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            print(f"🔄 Converted RGB to BGR for OpenCV")
        
        # Check device and confidence settings
        device = '0' if torch.cuda.is_available() else 'cpu'
        confidence_threshold = 0.25  # Even lower threshold for high-res images
        
        print(f"🖥️  Inference device: {device}")
        print(f"🎯 Confidence threshold: {confidence_threshold}")
        print(f"📏 Image size for inference: {img_array.shape[:2]}")
        
        # Calculate what the license plate size would be after scaling
        original_height, original_width = img_array.shape[:2]
        print(f"📊 Original dimensions: {original_width}x{original_height}")
        
        # Estimate license plate size (assuming it's roughly 1-3% of image width)
        estimated_plate_width = original_width * 0.02  # 2% of image width
        print(f"🔍 Estimated license plate width in original image: ~{estimated_plate_width:.0f}px")
        
        # See what it becomes after standard YOLO scaling (640px)
        scale_factor_640 = 640 / max(original_width, original_height)
        scaled_plate_width_640 = estimated_plate_width * scale_factor_640
        print(f"📉 After scaling to 640px: license plate would be ~{scaled_plate_width_640:.0f}px wide")
        
        # See what it becomes after larger scaling (1280px)
        scale_factor_1280 = 1280 / max(original_width, original_height)
        scaled_plate_width_1280 = estimated_plate_width * scale_factor_1280
        print(f"📈 After scaling to 1280px: license plate would be ~{scaled_plate_width_1280:.0f}px wide")
        
        # Run inference with larger input size
        print(f"🚀 Starting YOLO inference...")
        
        # Try different input sizes to handle high-resolution images
        input_sizes = [640, 1280]  # Try larger input size first
        best_detections = []
        best_results = None
        
        for imgsz in input_sizes:
            print(f"🔍 Trying inference with input size: {imgsz}x{imgsz}")
            
            results = model.predict(
                source=img_array,
                conf=confidence_threshold,
                device=device,
                verbose=True,
                save=False,
                show=False,
                imgsz=imgsz  # Specify input size
            )
            
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and result.boxes is not None:
                    num_detections = len(result.boxes)
                    print(f"   📦 Found {num_detections} detections at size {imgsz}")
                    
                    if num_detections > len(best_detections):
                        best_detections = result.boxes
                        best_results = results
                        print(f"   ✅ New best result with {num_detections} detections")
                else:
                    print(f"   ❌ No detections at size {imgsz}")
            
            # If we found detections at this size, we can break
            if best_detections and len(best_detections) > 0:
                print(f"🎯 Using results from input size {imgsz}")
                break
        
        # Use the best results
        results = best_results if best_results else results
        
        print(f"✅ Inference completed!")
        print(f"📊 Number of results: {len(results) if results else 0}")
        
        # Process results
        detections = []
        annotated_img = img_array.copy()
        
        if results and len(results) > 0:
            result = results[0]
            print(f"🔍 Processing result object...")
            print(f"📦 Result type: {type(result)}")
            
            # Debug: print all available attributes
            print(f"🏷️  Result attributes: {[attr for attr in dir(result) if not attr.startswith('_')]}")
            
            if hasattr(result, 'boxes') and result.boxes is not None:
                print(f"📦 Boxes found: {len(result.boxes)}")
                
                for i, box in enumerate(result.boxes):
                    print(f"\n📍 Processing box {i+1}:")
                    
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy()) if hasattr(box, 'cls') else 0
                    
                    print(f"   📍 Coordinates: ({x1}, {y1}) to ({x2}, {y2})")
                    print(f"   🎯 Confidence: {confidence:.4f}")
                    print(f"   🏷️  Class ID: {class_id}")
                    print(f"   📝 Class name: {model.names.get(class_id, 'unknown')}")
                    
                    # Draw rectangle
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    
                    # Add label
                    label = f"License Plate {confidence:.3f}"
                    cv2.putText(annotated_img, label, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(confidence),
                        'class': 'license_plate',
                        'class_id': class_id
                    })
                    
                print(f"✅ Found {len(detections)} detections above threshold {confidence_threshold}")
            else:
                print(f"❌ No boxes found in result")
                print(f"📦 Result.boxes: {result.boxes}")
                
                # Check if there are any detections below threshold
                if hasattr(result, 'boxes') and result.boxes is not None:
                    print(f"🔍 Checking all detections (including below threshold)...")
                    all_boxes = result.boxes
                    if len(all_boxes) > 0:
                        for i, box in enumerate(all_boxes):
                            conf = box.conf[0].cpu().numpy()
                            print(f"   Detection {i+1}: confidence = {conf:.4f}")
        else:
            print(f"❌ No results returned from model.predict()")
        
        # Convert back to RGB for web display
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        
        print(f"📊 Final detection count: {len(detections)}")
        print(f"🖼️  === IMAGE PROCESSING COMPLETE ===\n")
        
        return annotated_img, detections
        
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return None, []

def image_to_base64(image_array):
    """Convert numpy array to base64 string for web display"""
    try:
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(image_array)
        
        # Save to bytes buffer
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=95)
        
        # Encode to base64
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/jpeg;base64,{image_base64}"
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    try:
        print(f"\n🌐 === NEW PREDICTION REQUEST ===")
        
        # Check if model is loaded
        if model is None:
            print(f"❌ Model not loaded!")
            return jsonify({
                'success': False, 
                'error': 'Model not loaded. Please restart the server.'
            })
        
        print(f"✅ Model is loaded and ready")
        
        # Check if file was uploaded
        if 'file' not in request.files:
            print(f"❌ No file in request")
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            print(f"❌ Empty filename")
            return jsonify({'success': False, 'error': 'No file selected'})
        
        print(f"📎 Received file: {file.filename}")
        print(f"📏 File content type: {file.content_type}")
        
        # Validate file type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
        file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        print(f"📝 File extension: {file_ext}")
        
        if file_ext not in allowed_extensions:
            print(f"❌ Invalid file extension: {file_ext}")
            return jsonify({
                'success': False, 
                'error': 'Invalid file type. Please upload an image file.'
            })
        
        # Open and process image
        print(f"🖼️  Opening image file...")
        image = Image.open(file.stream)
        print(f"📐 Image size: {image.size}")
        print(f"🎨 Image mode: {image.mode}")
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            print(f"🔄 Converting {image.mode} to RGB")
            image = image.convert('RGB')
        
        # Process image with model
        print(f"🤖 Starting model processing...")
        processed_image, detections = process_image(image)
        
        if processed_image is None:
            print(f"❌ Image processing failed")
            return jsonify({
                'success': False, 
                'error': 'Error processing image'
            })
        
        print(f"✅ Image processing completed")
        print(f"📊 Detections found: {len(detections)}")
        
        # Convert to base64 for web display
        print(f"🔄 Converting to base64...")
        image_base64 = image_to_base64(processed_image)
        
        if image_base64 is None:
            print(f"❌ Base64 encoding failed")
            return jsonify({
                'success': False, 
                'error': 'Error encoding image'
            })
        
        # Prepare response
        response = {
            'success': True,
            'image': image_base64,
            'detections': detections,
            'message': f"Found {len(detections)} license plate(s)" if detections else "No license plates detected"
        }
        
        print(f"📤 Sending response with {len(detections)} detections")
        print(f"🌐 === REQUEST COMPLETE ===\n")
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Error in predict endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False, 
            'error': f'Server error: {str(e)}'
        })

if __name__ == '__main__':
    # Load model on startup
    if load_model():
        print("Model loaded successfully!")
        print("Starting Flask server...")
        app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False)
    else:
        print("Failed to load model. Server not started.") 