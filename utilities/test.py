import argparse
from pathlib import Path
from ultralytics import YOLO
import torch
import sys

def test_model(image_path: Path):
    """
    Tests the trained YOLO model on a single image.

    Args:
        image_path (Path): Path to the image file for testing.
    """
    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / 'runs' / 'train' / 'weights' / 'best.pt'

    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        print("Please make sure you have trained the model first by running the train.py script.")
        return

    if not image_path.exists():
        print(f"Error: Image file not found at {image_path}")
        return

    # Load the trained model
    model = YOLO(model_path)
    
    # Set device for inference
    device = '0' if torch.cuda.is_available() else 'cpu'

    # Run inference on the image
    results = model.predict(
        source=str(image_path),
        save=True,      # Save the image with bounding boxes
        show=True,      # Display the image with bounding boxes in a new window
        conf=0.5,       # Confidence threshold for detections
        device=device
    )
    
    print("Prediction complete. The output image with bounding boxes is saved in the 'runs/detect/' directory.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test a trained YOLO model on a custom image.")
    parser.add_argument('image_path', type=str, nargs='?', default='', help='Path to the image you want to test.')
    args = parser.parse_args()

    if not args.image_path:
        # Provide guidance if no image path is given.
        print("Usage: python utilities/test.py <path_to_your_image>")
        print("\nPlease provide the path to an image to test the model.")
        
        # Suggest a command with an example image from the validation set
        project_root = Path(__file__).resolve().parent.parent
        validation_images_path = project_root / 'dataset' / 'val' / 'images'
        if validation_images_path.exists():
            # Find the first image in the validation directory
            try:
                example_image = next(validation_images_path.glob('*.*'))
                print("\nFor example, you can use an image from your validation set:")
                print(f"python utilities/test.py \"{example_image}\"")
            except StopIteration:
                # This will happen if the directory is empty
                print("\nYour validation image folder seems to be empty. Please provide a path to any image.")
        sys.exit(1)

    image_to_test = Path(args.image_path)
    test_model(image_to_test) 