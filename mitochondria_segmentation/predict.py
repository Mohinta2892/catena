"""
Run prediction on held-out data.
Prediction can be run with trained models 
"""

# Inference Configuration and Execution

def predict(infer_args):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(infer_args.output_dir, exist_ok=True)

    # --- Load Model ---
    print("\n--- Loading Model for Inference ---")
    # Instantiate the model (must match the architecture used for training)
    # If you used RSUNet:
    model = RSUNet(in_ch=1, out_ch=1).to(device)
    # If you used MONAI UNet:
    # model = UNet(
    #     spatial_dims=3,
    #     in_channels=1,
    #     out_channels=1,
    #     channels=(28, 36, 48, 64, 80), # Matching your RSUNet's channel progression
    #     strides=(2, 2, 2, 2),        # 4 downsampling steps, each by factor of 2
    #     kernel_size=3,               # Default kernel size for conv layers in UNet (3x3x3 for 3D)
    #     up_kernel_size=3,            # Default kernel size for transpose conv (2x2x2 for 3D)
    #     num_res_units=0,             # Set to 0 for standard UNet (no residual units beyond default blocks)
    #                                  # Set to 1 or 2 for a ResUNet type behavior (more residual blocks)
    #     norm='batch',                # Use BatchNorm
    #     dropout=0.1                  # No dropout
    # ).to(device)

    # Load trained weights
    if not os.path.exists(infer_args.model_path):
        raise FileNotFoundError(f"Model weights not found at: {infer_args.model_path}")

    checkpoint = torch.load(infer_args.model_path, map_location=device)
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    # model.load_state_dict(torch.load(infer_args.model_path, map_location=device))
    model.eval()  # Set model to evaluation mode
    print(f"Model loaded from {infer_args.model_path} and set to evaluation mode.")

    # --- Setup Test Dataset and DataLoader ---
    print("\n--- Setting up Test Dataset and DataLoader ---")
    test_dataset = EMDataset(
        zarr_data_dirs=infer_args.test_zarr_dirs,
        label_type=infer_args.label_type,
        patch_size=infer_args.patch_size,
        stride=infer_args.stride,  # Use same stride as training for consistent patching
        original_resolution_nm=infer_args.original_res,
        target_resolution_nm=infer_args.target_res,
        clahe=infer_args.clahe,
        subsample_frac=1.0,  # Predict on all test patches
        subsample_number=0,
        balance_patches=False,  # No need for balancing during inference
        min_positive_pixels=0,  # No need for min_positive_pixels during inference
        expect_labels=False  # For infering on volumes with no labels
    )
    test_loader = DataLoader(test_dataset, batch_size=infer_args.batch_size, shuffle=False,
                             num_workers=infer_args.num_workers, pin_memory=True)
    print(f"Test DataLoader created with {len(test_dataset)} patches.")

    # --- Prepare Output Volume ---
    print("\n--- Preparing Output Volume ---")
    # Get the shape of the *first* test Zarr volume to pre-allocate output
    # Assuming all test Zarr volumes are of the same shape and are processed sequentially
    if not test_dataset._volume_metadata:
        raise ValueError("No test Zarr volumes found for inference.")

    first_volume_metadata = test_dataset._volume_metadata[0]
    full_volume_shape_raw = first_volume_metadata['shape']  # Z, Y, X of the raw Zarr volume

    # Calculate the effective shape of the output volume at target_res
    scale_z = infer_args.original_res[0] / infer_args.target_res[0]
    scale_y = infer_args.original_res[1] / infer_args.target_res[1]
    scale_x = infer_args.original_res[2] / infer_args.target_res[2]

    output_volume_shape = (
        int(full_volume_shape_raw[0] / scale_z),
        int(full_volume_shape_raw[1] / scale_y),
        int(full_volume_shape_raw[2] / scale_x)
    )
    print(f"Predicted output volume shape (at target resolution): {output_volume_shape}")

    # Initialize an empty NumPy array for the full prediction volume
    # Use float32 to store probabilities/logits before binarization, or uint8 for binary output
    full_prediction_volume = np.zeros(output_volume_shape, dtype=np.float32)
    # Keep track of how many predictions contributed to each pixel for averaging overlaps
    overlap_count_volume = np.zeros(output_volume_shape, dtype=np.uint8)

    # --- Run Inference and Merge Predictions ---
    print("\n--- Running Inference and Merging Predictions ---")
    start_inference_time = datetime.now()

    with torch.no_grad():
        for batch_idx, sample_batch_X in enumerate(test_loader):  # <--- CHANGE: Only expect sample_batch_X
            # If batch_size > 1 and dataset returns (image,) for multiple images,
            # sample_batch_X would be a list of tensors. If batch_size=1, it's a single tensor.
            # Ensure it's a tensor if num_workers > 0 and batch_size > 1
            # For batch_size=1, next(iter(dataloader)) directly gives the tensor, no need for tuple unpacking
            if isinstance(sample_batch_X,
                          list):  # If DataLoader returns a list (e.g., if batch_size > 1 and only one item is returned by __getitem__)
                sample_batch_X = sample_batch_X[0]  # Take the first (and only) item from the list

            sample_batch_X = sample_batch_X.to(device)

            # Get prediction (logits)
            pred_logits = model(sample_batch_X)
            pred_probs = torch.sigmoid(pred_logits)  # Convert logits to probabilities (0-1)

            # Move prediction to CPU and convert to NumPy
            pred_probs_np = pred_probs.squeeze(0).squeeze(0).cpu().numpy()  # Remove batch and channel dims

            # Get the original patch coordinates (in effective resolution)
            # This requires accessing the _patch_global_indices from the dataset
            # Since DataLoader shuffles, we need to get the original index from the sampler
            # For sequential loader, batch_idx directly maps to _patch_global_indices
            volume_idx, z_start_eff, y_start_eff, x_start_eff = test_dataset._patch_global_indices[batch_idx]

            # Calculate end coordinates for pasting
            z_end_eff = z_start_eff + infer_args.patch_size[0]
            y_end_eff = y_start_eff + infer_args.patch_size[1]
            x_end_eff = x_start_eff + infer_args.patch_size[2]

            # Paste prediction into the full volume
            # For simplicity, we're just adding. For true averaging, you'd need to divide by overlap_count_volume later.
            # For now, this is a simple "sum and then binarize" approach for overlaps.
            full_prediction_volume[z_start_eff:z_end_eff, y_start_eff:y_end_eff, x_start_eff:x_end_eff] += pred_probs_np
            overlap_count_volume[z_start_eff:z_end_eff, y_start_eff:y_end_eff, x_start_eff:x_end_eff] += 1

            if (batch_idx + 1) % 100 == 0:
                print(f"Processed {batch_idx + 1}/{len(test_loader)} patches.")

    # Final averaging for overlapping regions
    # Avoid division by zero where overlap_count_volume is 0 (i.e., regions not covered by any patch)
    # This assumes full coverage of the volume. If not, these regions will remain 0.
    full_prediction_volume = np.divide(full_prediction_volume, overlap_count_volume,
                                       out=np.zeros_like(full_prediction_volume),
                                       where=overlap_count_volume != 0)

    # Binarize the final prediction volume (0 or 1)
    final_binary_prediction = (full_prediction_volume >= 0.5).astype(np.uint8)

    end_inference_time = datetime.now()
    print(f"Inference and merging complete. Time taken: {end_inference_time - start_inference_time}")

    # --- Save Merged Prediction ---
    output_path = os.path.join(infer_args.output_dir, infer_args.output_filename)

    if infer_args.output_format == "zarr":
        # Save as Zarr
        zarr_output_path = output_path + ".zarr"
        print(f"Saving merged prediction to Zarr: {zarr_output_path}")
        # Create a new Zarr array
        zarr.save_array(zarr_output_path, final_binary_prediction, chunks=infer_args.patch_size,
                        compressor=zarr.Blosc())
        print("Zarr saved successfully.")
    elif infer_args.output_format == "tiff":
        # Save as multi-page TIFF
        tiff_output_path = output_path + ".tif"
        print(f"Saving merged prediction to TIFF: {tiff_output_path}")
        tff.imwrite(tiff_output_path, final_binary_prediction)
        print("TIFF saved successfully.")
    else:
        print(f"Unsupported output format: {infer_args.output_format}. Prediction not saved.")

    print("Inference script finished.")


# --- Inference Configuration ---
class InferenceArgs:
    def __init__(self):
        # Path to the trained model weights (.pth file)
        self.model_path = "/media/samia/DATA/PhD/codebases/MitoEM/mito_rsunet_from_scratch/best_model_IoU0.8577.pth"  # <--- IMPORTANT: Update this path!

        # Directory containing test Zarr files (can be multiple)
        self.test_zarr_dirs = [
            "/media/samia/DATA/mounts/fibserver1/smohinta_data/catena_data/HEMI_MITO/data_3d/test/"]  # <--- IMPORTANT: Update this path!
        self.label_type = 'neuron'  # The type of label the model was trained for

        # Model and Dataset parameters (must match training)
        self.patch_size = [128, 128, 128]  # Z Y X
        self.stride = [64, 64, 64]  # Z Y X (for inference tiling)
        self.original_res = [8.0, 8.0, 8.0]
        self.target_res = [8.0, 8.0, 8.0]  # Prediction will be at this resolution
        self.clahe = False  # Must match training preprocessing
        self.batch_size = 1  # Keep at 1 for simplicity in merging, or handle batch merging logic
        self.num_workers = 0  # Use 0 for debugging, can increase for speed

        # Output settings
        self.output_dir = "inference_results_rsunet"
        self.output_filename = "merged_predictions_hemi"  # merged_predictions_AL_crop_octo
        self.output_format = "tiff"  # "zarr" or "tiff"


if __name__ == '__main__':
    # Instantiate inference arguments
    infer_args = InferenceArgs()
    predict(infer_args)
