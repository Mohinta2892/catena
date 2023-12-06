"""
Credits: adapted from wandb.ai
"""
import torch


def find_cuda_devices():
    # get number of GPUs available
    device_count = torch.cuda.device_count()  # returns 1 in my case

    for d in range(device_count):
        # setting device on GPU if available, else CPU
        device = torch.device(f'cuda:{d}' if torch.cuda.is_available() else 'cpu')
        # Additional Info when using cuda
        if device.type == 'cuda':
            print('Using device:', device)
            print(torch.cuda.get_device_name(d))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(d) / 1024 ** 3, 1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(d) / 1024 ** 3, 1), 'GB')


if __name__ == '__main__':
    find_cuda_devices()
