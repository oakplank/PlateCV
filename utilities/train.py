import torch
import yaml
from ultralytics import YOLO
import os
from pathlib import Path

def load_hyperparameters():
    """
    Load training hyperparameters from config file.
    If file doesn't exist, use default values.
    """
    project_root = Path(__file__).parent.parent
    config_path = project_root / 'config.yaml'
    
    default_config = {
        'model': 'yolov12s.pt',      # Use pre-trained weights
        'epochs': 100,
        'patience': 20,
        'image_size': 640,
        'batch_size': 16,
        'workers': 8,
        'val_frequency': 1,
        'lr0': 0.001,  # Set a smaller initial learning rate for fine-tuning
    }

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            if not config:  # Handle empty config file
                config = default_config
    except FileNotFoundError:
        config = default_config
    
    # Ensure all essential keys are present, update if missing
    updated = False
    for key, value in default_config.items():
        if key not in config:
            config[key] = value
            updated = True

    # If the file was updated, save the changes
    if updated or not config_path.exists():
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
            
    return config

def train_model():
    project_root = Path(__file__).parent.parent
    config = load_hyperparameters()

    # Initialize YOLOv12 detection model
    model = YOLO(config['model'])

    # Train with specified hyperparameters
    results = model.train(
        data=str(project_root / 'dataset.yaml'),
        epochs=config['epochs'],
        batch=config['batch_size'],
        imgsz=config['image_size'],
        workers=config['workers'],
        patience=config['patience'],
        lr0=config['lr0'],  # Use learning rate from config
        val=True,
        # val_every is deprecated, use val=True and it runs every epoch by default
        plots=True,
        save=True,
        project=str(project_root / 'runs'),
        name='train',
        device='0' if torch.cuda.is_available() else 'cpu',
        augment=True,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
    )

    return results

if __name__ == '__main__':
    # It's best practice to run scripts from the project root
    # The paths inside the functions are now absolute, so no need for os.chdir
    results = train_model()
    print("Training completed. The results and best model are saved in the 'runs/train' directory.") 