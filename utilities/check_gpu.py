import torch

def check_gpu():
    """
    Checks for available GPUs and prints their details.
    """
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"Found {gpu_count} GPU(s).")
        for i in range(gpu_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        # Set the default device to GPU
        torch.set_default_device('cuda')
    else:
        print("No GPU available. Running on CPU.")

if __name__ == '__main__':
    check_gpu() 