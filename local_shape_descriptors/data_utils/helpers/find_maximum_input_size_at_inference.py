import torch
import torch.nn as nn
import torch.optim as optim

# Make a model here

def find_maximum_input_size(parameter, model_class, device='cuda'):
    input_channels = 1
    output_channels = 4  # Example: 3 for vectors + 1 for indicator
    fmap_num = parameter['fmap_num']
    fmap_inc_factor = parameter['fmap_inc_factor']
    downsample_factors = parameter['downsample_factors']

    # Start with a small size and grow it
    input_size = [16, 64, 64]  # Example starting size
    max_size = input_size.copy()
    model = None

    while True:
        try:
            # Create a new model for the current input size
            model = model_class(input_channels, output_channels, fmap_num, fmap_inc_factor, downsample_factors)
            model.to(device)

            # Create a random input tensor
            input_tensor = torch.randn((1, input_channels, *input_size), device=device)

            # Forward pass
            output = model(input_tensor)

            # If successful, increase size
            max_size = input_size.copy()
            input_size = [dim + 16 for dim in input_size]  # Increment each dimension
            del model, input_tensor, output
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"Out of memory at input size: {input_size}")
                break
            else:
                raise e

    print(f"Maximum input size that fits: {max_size}")
    return max_size

# Example parameter dictionary
parameter = {
    'fmap_num': 12,
    'fmap_inc_factor': 2,
    'downsample_factors': [[2, 2, 2], [2, 2, 2]],
}

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_input_size = find_maximum_input_size(parameter, UNet, device=device)
