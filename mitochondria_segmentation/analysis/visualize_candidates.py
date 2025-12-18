import neuroglancer
import numpy as np
import zarr
import sys
import os
import time
import threading

# --- CONFIGURATION ---
# Default resolution to match your analysis (z, y, x)
# This is crucial for converting the nm coordinates back to voxels
VOXEL_RES = (8, 8, 8)

# Host address (use 'localhost' for local, '0.0.0.0' for remote server)
BIND_ADDRESS = 'localhost'
PORT = 9999


def add_layer(txn, zarr_root, key, name, layer_type):
    """
    Adds a Zarr array as a layer to Neuroglancer.
    """
    path = os.path.join(zarr_root, key)

    if not os.path.exists(path):
        print(f"Warning: Could not find {path}")
        return

    print(f"Loading {name} from {path}...")

    # We use LocalVolume to serve data directly from disk
    # This requires the Zarr to be accessible by the machine running this script

    # Note: 'dimensions' maps the axes. Standard Zarr 3D is usually (z, y, x)
    # We define the voxel_size to allow physical coordinate navigation
    volume = neuroglancer.LocalVolume(
        volume_type=layer_type,
        data=zarr.open(path, mode='r'),
        dimensions=neuroglancer.CoordinateSpace(
            names=['z', 'y', 'x'],
            units='nm',
            scales=VOXEL_RES
        ),
    )

    if layer_type == 'segmentation':
        txn.layers[name] = neuroglancer.SegmentationLayer(source=volume)
    else:
        txn.layers[name] = neuroglancer.ImageLayer(source=volume)


def start_viewer(zarr_path):
    neuroglancer.set_server_bind_address(BIND_ADDRESS, bind_port=PORT)
    viewer = neuroglancer.Viewer()

    # Clean up zarr path string
    zarr_path = zarr_path.strip().rstrip('/')
    sample_name = os.path.basename(zarr_path).replace('.zarr', '')

    with viewer.txn() as s:
        # Add Raw Volume
        add_layer(s, zarr_path, 'volumes/raw', 'Raw Image', 'image')

        # Add Labels
        add_layer(s, zarr_path, 'volumes/labels/neuron_ids', 'Mitochondria', 'segmentation')

        # Set a sane initial view
        # We can't know the exact center without loading data, so we start at 0,0,0
        # or defaults.
        print("Layers added.")

    print(f"\n{'-' * 60}")
    print(f"VIWER RUNNING FOR: {sample_name}")
    print(f"{'-' * 60}")
    print(f"Link: {viewer}")
    print(f"{'-' * 60}\n")

    return viewer


def navigation_loop(viewer):
    """
    Interactive loop to accept coordinates from the user.
    """
    print("READY TO NAVIGATE.")
    print("Paste the coordinates from your analysis output (e.g., [1234, 5678, 9000])")
    print("Type 'q' to quit.\n")

    while True:
        try:
            user_input = input("Enter Position (nm) >> ").strip()

            if user_input.lower() in ['q', 'quit', 'exit']:
                print("Closing viewer...")
                break

            # Clean input (remove brackets if copied directly from list print)
            clean_input = user_input.replace('[', '').replace(']', '').replace(',', ' ')
            coords = list(map(float, clean_input.split()))

            if len(coords) != 3:
                print("Error: Please enter 3 numbers (z, y, x)")
                continue

            # Since we defined the coordinate space in the layer as 'nm',
            # we can pass physical coordinates directly!
            with viewer.txn() as s:
                s.position = coords

            print(f"Jumped to: {coords}")

        except ValueError:
            print("Invalid input. Please enter numbers.")
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_candidates.py /path/to/your/sample.zarr")
        sys.exit(1)

    zarr_path = sys.argv[1]

    if not os.path.exists(zarr_path):
        print(f"Error: Path not found: {zarr_path}")
        sys.exit(1)

    # Launch
    viewer = start_viewer(zarr_path)

    # Start Interaction
    navigation_loop(viewer)
